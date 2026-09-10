"""Извлечение нормализованного аудио: WAV PCM s16le, 16 кГц, моно.

Нормализация делается один раз здесь, чтобы VAD и ASR не зависели от
кодека, частоты дискретизации и раскладки каналов исходного файла.
"""

from __future__ import annotations

import contextlib
import logging
import wave
from pathlib import Path

from ..contracts import AudioArtifact, AudioTrackInfo, MediaInfo, MissingAudioTrackError
from ._ffmpeg import FFMPEG, MediaProcessingError, atomic_file, run, stderr_tail

logger = logging.getLogger(__name__)

__all__ = ["extract_audio", "select_audio_track", "TARGET_SAMPLE_RATE"]

#: Частота дискретизации, которую ждут VAD (silero) и ASR-бэкенды.
TARGET_SAMPLE_RATE = 16000

#: Фильтр выравнивания начала аудио по нулю контейнера.
#: Аудиопоток может стартовать не в нуле (``start_time`` > 0). Без этого
#: фильтра ffmpeg отбрасывает начальное смещение: содержимое WAV съезжает к
#: нулю, а длительность выхода становится меньше длительности записи — то
#: есть все метки ASR оказываются сдвинутыми относительно кадров.
#: ``first_pts=0`` дополняет начало тишиной до нуля контейнера.
_ALIGN_FILTER = "aresample=async=1:first_pts=0"

#: Допуск на расхождение длительностей аудио и записи, секунды.
#: AAC несёт encoder delay/padding порядка 1024–2048 сэмплов (21–43 мс при
#: 48 кГц), плюс длительность контейнера округляется по таймбейсу. 0.25 с —
#: на порядок выше наблюдаемого и на порядок ниже интервала выборки кадров.
DURATION_TOLERANCE_S = 0.25


def select_audio_track(media: MediaInfo, track_index: int | None = None) -> AudioTrackInfo:
    """Выбрать аудиодорожку и залогировать фактический выбор.

    `track_index` — индекс потока в контейнере (то же, что `AudioTrackInfo.index`
    и `ffprobe stream index`), а не порядковый номер среди аудиодорожек.
    При `None` берётся первая аудиодорожка файла.
    """
    if not media.audio_tracks:
        raise MissingAudioTrackError(
            f"во входном файле нет аудиодорожки: {media.path} — транскрипт речи получить невозможно"
        )

    if track_index is None:
        track = media.audio_tracks[0]
        reason = "выбор не задан, взята первая"
    else:
        found = next((t for t in media.audio_tracks if t.index == track_index), None)
        if found is None:
            available = ", ".join(str(t.index) for t in media.audio_tracks)
            raise MissingAudioTrackError(
                f"аудиодорожка с индексом {track_index} отсутствует в {media.path}; "
                f"доступные индексы: {available}"
            )
        track = found
        reason = "выбор задан конфигурацией"

    logger.info(
        "аудиодорожка: индекс %d (%s) — %s: codec=%s, %d Гц, каналов %d, язык=%s, title=%s",
        track.index,
        reason,
        f"всего дорожек {len(media.audio_tracks)}",
        track.codec,
        track.sample_rate,
        track.channels,
        track.language or "—",
        track.title or "—",
    )
    return track


def _wav_duration(path: Path) -> tuple[int, int, float]:
    """(sample_rate, channels, duration_s) готового WAV — по его же заголовку."""
    with contextlib.closing(wave.open(str(path), "rb")) as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.getnframes()
    return rate, channels, (frames / rate if rate else 0.0)


def extract_audio(
    media: MediaInfo,
    out_path: Path | str,
    track_index: int | None = None,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> AudioArtifact:
    """Извлечь аудиодорожку в WAV PCM s16le, моно, `sample_rate` Гц.

    Многоканальная дорожка сводится в моно средствами ffmpeg (`-ac 1`):
    микширование каналов, а не выбор одного из них, — чтобы речь,
    присутствующая хотя бы в одном канале, не потерялась.

    Файл пишется во временный путь рядом с целевым и переименовывается
    только после успешного завершения ffmpeg: при ошибке `out_path` не
    создаётся.
    """
    out_path = Path(out_path)
    track = select_audio_track(media, track_index)

    with atomic_file(out_path) as tmp:
        args = [
            FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-i",
            str(media.path),
            "-map",
            f"0:{track.index}",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-af",
            _ALIGN_FILTER,
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(tmp),
        ]
        result = run(args)
        if result.returncode != 0:
            raise MediaProcessingError(
                f"не удалось извлечь аудио из {media.path}: {stderr_tail(result)}"
            )

    actual_rate, channels, duration_s = _wav_duration(out_path)
    logger.info(
        "аудио извлечено: %s — %.3f с, %d Гц, каналов %d",
        out_path,
        duration_s,
        actual_rate,
        channels,
    )
    if media.duration_s > 0 and media.duration_s - duration_s > DURATION_TOLERANCE_S:
        # ffmpeg возвращает 0 и на подбитом посередине контейнере, поэтому
        # недобор длительности виден только при явном сравнении.
        logger.warning(
            "аудио короче записи: %.3f с против %.3f с (недобор %.3f с) — "
            "часть речи в транскрипт не попадёт: %s",
            duration_s,
            media.duration_s,
            media.duration_s - duration_s,
            media.path,
        )
    return AudioArtifact(
        path=out_path,
        sample_rate=actual_rate,
        duration_s=duration_s,
        source_track_index=track.index,
    )
