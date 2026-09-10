"""Чтение нормализованного WAV без внешних зависимостей.

Стадия распознавания работает с артефактом media-ingest: WAV PCM, моно.
Здесь только stdlib `wave` + numpy — ни одна тяжёлая библиотека не нужна,
поэтому модуль импортируется всегда и его можно тестировать на синтетике.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..contracts import PipelineError

__all__ = [
    "UnreadableAudioError",
    "clear_audio_cache",
    "read_wav_mono",
    "write_wav_mono",
]


class UnreadableAudioError(PipelineError):
    """WAV не читается или содержит неподдерживаемый формат сэмплов."""


#: Кэш последнего прочитанного файла: (ключ файла) -> (сэмплы, частота).
#: За прогон стадии одно и то же аудио читают VAD и ASR; на 91-минутной
#: записи повторное чтение — это лишние ~350 МБ и лишний проход по диску.
#: Кэш ровно на один файл: память ограничена одной записью и освобождается
#: либо чтением другого файла, либо явным clear_audio_cache().
_CACHE_KEY: tuple[object, ...] | None = None
_CACHE_VALUE: tuple[np.ndarray, int] | None = None


def clear_audio_cache() -> None:
    """Забыть закэшированные сэмплы (освободить память между стадиями, D7)."""
    global _CACHE_KEY, _CACHE_VALUE
    _CACHE_KEY = None
    _CACHE_VALUE = None


def _cache_key(path: Path) -> tuple[object, ...] | None:
    """Ключ файла: путь, inode, mtime, ctime, размер. None — файл не опросить.

    Одних mtime и размера мало: WAV одной длительности одного размера, а
    ``cp -p`` / ``os.utime`` возвращают прежний mtime. ctime так не подделать —
    любая запись в файл его обновляет.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (
        str(path.resolve()),
        stat.st_ino,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
    )


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Прочитать WAV как моно float32 в диапазоне [-1.0, 1.0].

    Возвращает (сэмплы, частота дискретизации). Многоканальный вход
    сводится в моно усреднением каналов.

    Результат последнего чтения кэшируется по (путь, inode, mtime, ctime, размер):
    повторный вызов на том же файле возвращает **тот же** массив, поэтому
    вызывающий не должен менять его на месте.
    """
    global _CACHE_KEY, _CACHE_VALUE

    key = _cache_key(path)
    if key is not None and key == _CACHE_KEY and _CACHE_VALUE is not None:
        return _CACHE_VALUE

    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, OSError) as exc:  # noqa: PERF203 — точная диагностика важнее
        raise UnreadableAudioError(f"не удалось прочитать WAV {path}: {exc}") from exc

    raw = _decode_pcm(frames, width, path)
    del frames  # сырые байты больше не нужны: на 91 мин это ~170 МБ
    if channels > 1:
        usable = (raw.size // channels) * channels
        raw = raw[:usable].reshape(-1, channels).mean(axis=1, dtype=np.float32)
    samples = np.ascontiguousarray(raw, dtype=np.float32)

    if key is not None:
        _CACHE_KEY = key
        _CACHE_VALUE = (samples, sample_rate)
    return samples, sample_rate


def _decode_pcm(frames: bytes, width: int, path: Path) -> np.ndarray:
    """Развернуть сырые PCM-байты в float32 [-1.0, 1.0].

    Нормировка делается на месте (``/=``): промежуточный результат деления —
    ещё одна копия всей записи, а она тут ни к чему.
    """
    if width == 1:  # 8-bit PCM беззнаковый
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data -= 128.0
        data /= 128.0
        return data
    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        data /= 32768.0
        return data
    if width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        data /= 2147483648.0
        return data
    raise UnreadableAudioError(
        f"неподдерживаемая разрядность сэмплов {width * 8} бит в {path}; "
        "ожидается PCM 8/16/32 бит"
    )


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Записать моно WAV PCM s16. Нужна для синтетики в тестах и отладки."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path
