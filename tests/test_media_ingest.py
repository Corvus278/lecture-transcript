"""Проверки стадии media-ingest (задачи 2.1–2.4).

Тесты намеренно не зависят от `tests/conftest.py`: эталонная запись берётся
из env `LECTURE_REFERENCE_MP4`, короткие клипы и синтетические файлы
(две аудиодорожки, стерео, видео без звука, аудио без видео, битый файл)
нарезаются здесь же через ffmpeg lavfi.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from lecture_transcript.contracts import (
    AudioTrackInfo,
    MediaInfo,
    MissingAudioTrackError,
    MissingVideoTrackError,
    UnreadableMediaError,
    VideoStreamInfo,
)
from lecture_transcript.media_ingest import _ffmpeg as ffmpeg_util
from lecture_transcript.media_ingest import (
    HwaccelUnavailableError,
    MediaProcessingError,
    extract_audio,
    extract_frames,
    probe,
    resolve_hwaccel,
)

REFERENCE_DEFAULT = "~/Downloads/wr_20260909_1150.mp4"

# Допуск на длительность извлечённого аудио.
# Обоснование: AAC несёт encoder delay/padding порядка 1024–2048 сэмплов
# (21–43 мс при 48 кГц), плюс длительность контейнера округляется по
# таймбейсу. Измеренное расхождение на эталонной записи — 18 мс на 5482 с.
# 0.25 с — на порядок выше наблюдаемого и на порядок ниже интервала выборки
# кадров (1 с), то есть достаточно строго, чтобы поймать реальный обрыв
# дорожки, и достаточно мягко, чтобы не ловить артефакты кодека.
AUDIO_DURATION_TOLERANCE_S = 0.25


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------


def _reference_path() -> Path:
    """Путь к эталонной записи; ``~`` разворачивается и в env, и в дефолте."""
    return Path(
        os.environ.get("LECTURE_REFERENCE_MP4") or REFERENCE_DEFAULT
    ).expanduser()


def _ffmpeg(*args: str) -> None:
    """Собрать синтетический файл; падение ffmpeg — ошибка самого теста."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"не удалось подготовить фикстуру: {result.stderr[-500:]}")


def _dominant_frequency(path: Path) -> float:
    """Доминирующая частота WAV-файла по FFT — проверка фактического содержимого."""
    with contextlib.closing(wave.open(str(path), "rb")) as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    signal = samples.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    return float(np.fft.rfftfreq(len(signal), 1.0 / rate)[int(np.argmax(spectrum))])


def _rms(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _leftovers(directory: Path) -> list[Path]:
    """Временные пути стадии, которые не должны пережить ошибку."""
    return [p for p in directory.iterdir() if p.name.startswith(".") and ".tmp-" in p.name]


# --------------------------------------------------------------------------
# Синтетические входные файлы
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media_ingest_synth")


@pytest.fixture(scope="session")
def two_audio_mp4(synth_dir: Path) -> Path:
    """Видео + две аудиодорожки: первая 440 Гц, вторая 1200 Гц."""
    path = synth_dir / "two_audio.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=1200:sample_rate=48000:duration=5",
        "-map", "0:v", "-map", "1:a", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    )
    return path


@pytest.fixture(scope="session")
def stereo_wav(synth_dir: Path) -> Path:
    """Стерео 48 кГц: слева 440 Гц, справа тишина."""
    path = synth_dir / "stereo_left_only.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000:duration=4",
        "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]",
        "-map", "[a]", "-c:a", "pcm_s16le", "-t", "4", str(path),
    )
    return path


@pytest.fixture(scope="session")
def video_without_audio(synth_dir: Path) -> Path:
    path = synth_dir / "video_only.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
    )
    return path


@pytest.fixture(scope="session")
def audio_without_video(synth_dir: Path) -> Path:
    path = synth_dir / "audio_only.m4a"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3",
        "-c:a", "aac", "-vn", str(path),
    )
    return path


@pytest.fixture(scope="session")
def broken_media(synth_dir: Path) -> Path:
    path = synth_dir / "broken.mp4"
    path.write_bytes(os.urandom(64 * 1024))
    return path


@pytest.fixture(scope="session")
def short_clip(synth_dir: Path) -> Path:
    """Короткий клип 12 с @ 25 fps — быстрый аналог эталона для кадров."""
    path = synth_dir / "clip_12s.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=12",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    )
    return path


@pytest.fixture(scope="session")
def offset_audio_mp4(synth_dir: Path) -> Path:
    """Видео с нуля, аудиодорожка со смещением ~3 с; тон в абсолютных 8–9 с.

    Такое смещение (``start_time`` != 0) встречается в записях вебинаров.
    Тон стоит в 5–6 с самой дорожки, то есть в 8–9 с записи.
    """
    path = synth_dir / "offset_audio.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=13",
        "-itsoffset", "3",
        "-f", "lavfi", "-i",
        "aevalsrc=0.4*sin(2*PI*440*t)*between(t\\,5\\,6):d=10:s=48000",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    )
    return path


@pytest.fixture(scope="session")
def timecoded_clip(synth_dir: Path) -> Path:
    """Клип 10 с, в котором содержимое кадра кодирует его секунду.

    Кадр разбит на 10 колонок по 64 px; в секунду T залиты белым первые
    ``floor(T) + 1`` колонок. Кодируется геометрией, а не яркостью,
    поэтому восстановление не зависит от диапазона (limited/full) и от
    потерь кодека.
    """
    path = synth_dir / "timecoded.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=640x64:rate=25:duration=10",
        "-vf", "geq=lum='255*lt(X\\,64*(floor(T)+1))':cb=128:cr=128",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path),
    )
    return path


def _second_encoded_in_frame(path: Path) -> int:
    """Секунда, закодированная в кадре `timecoded_clip`: белых колонок минус 1."""
    from PIL import Image

    with Image.open(path) as img:
        gray = img.convert("L")
        white = sum(1 for x in range(32, 640, 64) if gray.getpixel((x, 32)) > 128)
    return white - 1


# ==========================================================================
# 2.1 — извлечение аудио в WAV 16 кГц моно
# ==========================================================================


def test_extract_audio_normalizes_to_16k_mono(short_clip: Path, tmp_path: Path) -> None:
    media = probe(short_clip)
    artifact = extract_audio(media, tmp_path / "audio.wav")

    assert artifact.sample_rate == 16000
    with contextlib.closing(wave.open(str(artifact.path), "rb")) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2  # PCM s16
        assert wav.getframerate() == 16000
    assert abs(artifact.duration_s - media.duration_s) <= AUDIO_DURATION_TOLERANCE_S


def test_extract_audio_downmixes_multichannel(stereo_wav: Path, tmp_path: Path) -> None:
    """Речь, присутствующая только в одном канале, переживает сведение в моно."""
    media = probe(stereo_wav)
    assert media.audio_tracks[0].channels == 2

    artifact = extract_audio(media, tmp_path / "mono.wav")

    assert artifact.sample_rate == 16000
    assert _rms(artifact.path) > 1000.0  # канал не потерян и не занулён
    assert abs(_dominant_frequency(artifact.path) - 440.0) < 5.0


@pytest.mark.reference
def test_extract_audio_reference_duration_matches(tmp_path: Path) -> None:
    """2.1 на эталонной записи: длительность, частота и число каналов."""
    reference = _reference_path()
    if not reference.exists():
        pytest.skip(f"эталонная запись недоступна: {reference}")

    media = probe(reference)
    artifact = extract_audio(media, tmp_path / "reference.wav")

    assert artifact.sample_rate == 16000
    with contextlib.closing(wave.open(str(artifact.path), "rb")) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
    assert abs(artifact.duration_s - media.duration_s) <= AUDIO_DURATION_TOLERANCE_S
    assert media.duration_s > 5000.0  # это действительно длинная запись, а не обрезок


def _per_second_rms(path: Path) -> list[float]:
    """RMS по односекундным окнам WAV — где именно в файле лежит звук."""
    with contextlib.closing(wave.open(str(path), "rb")) as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    signal = samples.astype(np.float64)
    return [
        float(np.sqrt(np.mean(signal[i * rate : (i + 1) * rate] ** 2)))
        for i in range(len(signal) // rate)
    ]


def test_audio_track_offset_preserved(offset_audio_mp4: Path, tmp_path: Path) -> None:
    """Смещение аудиодорожки не съедается: WAV покрывает всю запись.

    Дорожка начинается в ~3 с контейнера, тон стоит в абсолютных 8–9 с.
    Без выравнивания по нулю контейнера содержимое съезжает к нулю (тон
    оказывается в 5-й секунде), а длительность выхода недобирает 3 с.
    """
    media = probe(offset_audio_mp4)
    assert media.duration_s == pytest.approx(13.0, abs=0.1)

    artifact = extract_audio(media, tmp_path / "offset.wav")

    assert abs(artifact.duration_s - media.duration_s) <= AUDIO_DURATION_TOLERANCE_S
    energy = _per_second_rms(artifact.path)
    loudest = max(range(len(energy)), key=energy.__getitem__)
    assert loudest == 8, f"тон уехал в секунду {loudest}, ожидалась 8: {energy}"


def test_short_audio_is_reported(
    short_clip: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Недобор длительности не проходит молча: ffmpeg возвращает 0 и на обрыве."""
    media = probe(short_clip)
    longer = dataclasses.replace(media, duration_s=media.duration_s + 5.0)

    with caplog.at_level(logging.WARNING, logger="lecture_transcript.media_ingest.audio"):
        extract_audio(longer, tmp_path / "short.wav")

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("короче записи" in m for m in warnings), warnings


# ==========================================================================
# 2.2 — выбор аудиодорожки при нескольких дорожках
# ==========================================================================


def test_probe_detects_two_audio_tracks(two_audio_mp4: Path) -> None:
    media = probe(two_audio_mp4)
    assert len(media.audio_tracks) == 2
    assert [t.index for t in media.audio_tracks] == [1, 2]
    assert media.has_video and media.video is not None
    assert (media.video.width, media.video.height) == (320, 240)


def test_default_track_is_the_first_one(two_audio_mp4: Path, tmp_path: Path) -> None:
    media = probe(two_audio_mp4)
    artifact = extract_audio(media, tmp_path / "default.wav")

    assert artifact.source_track_index == media.audio_tracks[0].index
    # первая дорожка — sine 440 Гц; проверяем содержимое, а не только метаданные
    assert abs(_dominant_frequency(artifact.path) - 440.0) < 5.0


def test_explicit_track_index_is_honoured(two_audio_mp4: Path, tmp_path: Path) -> None:
    media = probe(two_audio_mp4)
    artifact = extract_audio(media, tmp_path / "second.wav", track_index=2)

    assert artifact.source_track_index == 2
    assert abs(_dominant_frequency(artifact.path) - 1200.0) < 5.0


def test_track_selection_is_logged(
    two_audio_mp4: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    media = probe(two_audio_mp4)
    with caplog.at_level(logging.INFO, logger="lecture_transcript.media_ingest.audio"):
        extract_audio(media, tmp_path / "logged.wav", track_index=2)

    messages = [r.getMessage() for r in caplog.records]
    selection = [m for m in messages if "аудиодорожка: индекс 2" in m]
    assert selection, f"выбор дорожки не залогирован: {messages}"
    assert "всего дорожек 2" in selection[0]


def test_unknown_track_index_rejected(two_audio_mp4: Path, tmp_path: Path) -> None:
    media = probe(two_audio_mp4)
    out = tmp_path / "nope.wav"
    with pytest.raises(MissingAudioTrackError):
        extract_audio(media, out, track_index=42)
    assert not out.exists()


# ==========================================================================
# 2.3 — извлечение кадров с настраиваемой частотой и таймкодами
# ==========================================================================


def _assert_frames_consistent(artifact, duration_s: float, fps: float) -> None:
    frames = artifact.frames
    assert frames, "кадры не извлечены"
    assert artifact.fps == pytest.approx(fps)

    # число кадров соответствует длительности при заданной частоте выборки;
    # ±2 кадра — краевые эффекты сетки fps на первом и последнем узле
    assert abs(len(frames) - duration_s * fps) <= 2

    timestamps = [f.timestamp_s for f in frames]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:])), "метки не монотонны"
    assert timestamps[0] == pytest.approx(0.0)
    assert timestamps[-1] <= duration_s

    # индексы плотные, файлы существуют, лексикографический порядок = временной
    assert [f.index for f in frames] == list(range(len(frames)))
    assert all(f.path.exists() for f in frames)
    assert [f.path.name for f in frames] == sorted(f.path.name for f in frames)
    assert all(f.path.parent == artifact.directory for f in frames)


def test_extract_frames_1fps_short_clip(short_clip: Path, tmp_path: Path) -> None:
    media = probe(short_clip)
    artifact = extract_frames(media, tmp_path / "frames", fps=1.0)
    _assert_frames_consistent(artifact, media.duration_s, 1.0)


def test_extract_frames_sampling_rate_is_configurable(
    short_clip: Path, tmp_path: Path
) -> None:
    media = probe(short_clip)
    dense = extract_frames(media, tmp_path / "frames_2fps", fps=2.0)
    _assert_frames_consistent(dense, media.duration_s, 2.0)
    assert dense.frames[1].timestamp_s == pytest.approx(0.5)

    sparse = extract_frames(media, tmp_path / "frames_half", fps=0.5)
    _assert_frames_consistent(sparse, media.duration_s, 0.5)
    assert len(sparse.frames) < len(dense.frames)


def test_extract_frames_png_format(short_clip: Path, tmp_path: Path) -> None:
    media = probe(short_clip)
    artifact = extract_frames(media, tmp_path / "png", fps=0.5, image_format="png")
    assert all(f.path.suffix == ".png" for f in artifact.frames)
    assert artifact.frames[0].path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_extract_frames_scale_width(short_clip: Path, tmp_path: Path) -> None:
    from PIL import Image

    media = probe(short_clip)
    artifact = extract_frames(media, tmp_path / "small", fps=0.5, scale_width=160)
    with Image.open(artifact.frames[0].path) as img:
        assert img.width == 160


def test_invalid_fps_rejected(short_clip: Path, tmp_path: Path) -> None:
    media = probe(short_clip)
    out = tmp_path / "bad_fps"
    with pytest.raises(ValueError):
        extract_frames(media, out, fps=0.0)
    assert not out.exists()


@pytest.mark.parametrize("fps", [1.0, 2.0])
def test_frame_timestamp_matches_frame_content(
    timecoded_clip: Path, tmp_path: Path, fps: float
) -> None:
    """Метка кадра совпадает с позицией, из которой кадр реально взят.

    Содержимое кадра кодирует его секунду, поэтому сдвиг сетки выборки или
    проигнорированный фильтр ``fps`` тест ломают — в отличие от проверки
    «метки монотонны и начинаются с нуля», которая верна по построению.
    """
    media = probe(timecoded_clip)
    artifact = extract_frames(
        media, tmp_path / f"timecoded_{fps}", fps=fps, image_format="png", hwaccel="none"
    )

    for frame in artifact.frames:
        encoded = _second_encoded_in_frame(frame.path)
        assert 0 <= encoded <= 9, f"кадр {frame.path.name} нечитаем: {encoded}"
        # точность не хуже интервала выборки (spec.md, «Точность временны́х меток»)
        assert abs(encoded - frame.timestamp_s) <= 1.0 / fps, (
            f"кадр {frame.index}: содержимое из секунды {encoded}, "
            f"метка {frame.timestamp_s}"
        )
        # секунда содержимого = floor(метки): сетка fps идёт от начала записи
        assert encoded == int(frame.timestamp_s)


def test_frames_output_dir_with_percent_sign(short_clip: Path, tmp_path: Path) -> None:
    """``%`` в пути каталога — не printf-шаблон для muxer'а image2."""
    media = probe(short_clip)
    out_dir = tmp_path / "Лекция 100% готово" / "frames"

    artifact = extract_frames(media, out_dir, fps=1.0, hwaccel="none")

    _assert_frames_consistent(artifact, media.duration_s, 1.0)
    assert artifact.frames[0].path.name == "frame_000000.jpg"


def test_frames_output_dir_with_printf_specifier(
    short_clip: Path, tmp_path: Path
) -> None:
    """``%d`` в имени каталога не превращается в номер кадра."""
    media = probe(short_clip)
    out_dir = tmp_path / "lecture_%d_run"

    artifact = extract_frames(media, out_dir, fps=0.5, hwaccel="none")

    assert artifact.directory == out_dir
    assert all(f.path.parent == out_dir for f in artifact.frames)
    assert len(list(out_dir.iterdir())) == len(artifact.frames)


@pytest.mark.reference
@pytest.mark.slow
def test_extract_frames_reference_1fps(tmp_path: Path) -> None:
    """2.3 на эталонной записи 1:31:22 при 1 fps.

    Кадры масштабируются до ширины 640: проверяются число кадров и
    монотонность меток, а не качество изображения, — так тест не пишет
    полтора гигабайта во временный каталог.
    """
    reference = _reference_path()
    if not reference.exists():
        pytest.skip(f"эталонная запись недоступна: {reference}")

    media = probe(reference)
    artifact = extract_frames(
        media, tmp_path / "reference_frames", fps=1.0, scale_width=640, quality=8
    )
    _assert_frames_consistent(artifact, media.duration_s, 1.0)
    assert len(artifact.frames) > 5000


# ==========================================================================
# 2.4 — состав дорожек, отсутствующие дорожки, атомарность
# ==========================================================================


def test_video_without_audio_raises(video_without_audio: Path, tmp_path: Path) -> None:
    media = probe(video_without_audio)
    assert media.has_video
    assert not media.has_audio

    out = tmp_path / "should_not_exist.wav"
    with pytest.raises(MissingAudioTrackError):
        extract_audio(media, out)
    assert not out.exists()
    assert not _leftovers(tmp_path)


def test_audio_without_video_works_in_audio_mode(
    audio_without_video: Path, tmp_path: Path
) -> None:
    media = probe(audio_without_video)
    assert media.has_audio
    assert media.has_video is False
    assert media.video is None

    artifact = extract_audio(media, tmp_path / "audio_mode.wav")
    assert artifact.sample_rate == 16000
    assert abs(artifact.duration_s - media.duration_s) <= AUDIO_DURATION_TOLERANCE_S

    # кадры без видеотрека не извлекаются и каталог не создаётся
    frames_dir = tmp_path / "frames_should_not_exist"
    with pytest.raises(MissingVideoTrackError):
        extract_frames(media, frames_dir, fps=1.0)
    assert not frames_dir.exists()
    assert not _leftovers(tmp_path)


def test_missing_video_track_warns_about_incomplete_result(
    audio_without_video: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Аудио-режим — не молчаливый: пользователь предупреждён о неполноте.

    spec.md, сценарий «Файл без видеотрека»: пайплайн продолжает работу,
    слайды пропускаются, **и пользователь предупреждён**.
    """
    with caplog.at_level(logging.WARNING, logger="lecture_transcript.media_ingest"):
        media = probe(audio_without_video)

    assert media.video is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "предупреждения об отсутствии видео нет"
    assert any("нет видеодорожки" in m and "неполным" in m for m in warnings), warnings


def test_broken_file_raises_unreadable(broken_media: Path) -> None:
    with pytest.raises(UnreadableMediaError):
        probe(broken_media)


def test_missing_file_raises_unreadable(tmp_path: Path) -> None:
    with pytest.raises(UnreadableMediaError):
        probe(tmp_path / "нет-такого.mp4")


def test_empty_file_raises_unreadable(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(UnreadableMediaError):
        probe(empty)


def test_audio_failure_leaves_no_partial_artifact(
    two_audio_mp4: Path, tmp_path: Path
) -> None:
    """ffmpeg падает посреди работы -> целевого файла и временных путей нет."""
    broken_media_info = MediaInfo(
        path=two_audio_mp4,
        duration_s=5.0,
        audio_tracks=(AudioTrackInfo(index=99, codec="aac", sample_rate=48000, channels=1),),
    )
    out = tmp_path / "partial.wav"
    with pytest.raises(MediaProcessingError):
        extract_audio(broken_media_info, out)

    assert not out.exists()
    assert not _leftovers(tmp_path)


def test_frames_failure_leaves_no_partial_directory(
    short_clip: Path, tmp_path: Path
) -> None:
    broken_media_info = MediaInfo(
        path=short_clip,
        duration_s=12.0,
        video=VideoStreamInfo(index=99, codec="h264", width=640, height=360, fps=25.0),
    )
    out_dir = tmp_path / "partial_frames"
    with pytest.raises(MediaProcessingError):
        extract_frames(broken_media_info, out_dir, fps=1.0, hwaccel="none")

    assert not out_dir.exists()
    assert not _leftovers(tmp_path)


def test_frames_failure_preserves_existing_directory(
    short_clip: Path, tmp_path: Path
) -> None:
    """Неудачный повторный прогон не портит уже собранный каталог кадров."""
    media = probe(short_clip)
    out_dir = tmp_path / "frames"
    good = extract_frames(media, out_dir, fps=0.5)
    before = sorted(p.name for p in out_dir.iterdir())
    assert before

    broken_media_info = MediaInfo(
        path=short_clip,
        duration_s=12.0,
        video=VideoStreamInfo(index=99, codec="h264", width=640, height=360, fps=25.0),
    )
    with pytest.raises(MediaProcessingError):
        extract_frames(broken_media_info, out_dir, fps=0.5, hwaccel="none")

    assert sorted(p.name for p in out_dir.iterdir()) == before
    assert all(f.path.exists() for f in good.frames)
    assert not _leftovers(tmp_path)


def test_frames_replace_existing_file_leaves_no_temp(
    short_clip: Path, tmp_path: Path
) -> None:
    """Целевой путь занят обычным файлом -> он заменяется каталогом без мусора.

    `shutil.rmtree(..., ignore_errors=True)` на файле тихо не делает ничего,
    поэтому переименованный старый артефакт мог остаться навсегда.
    """
    media = probe(short_clip)
    target = tmp_path / "frames"
    target.write_text("это был файл, а не каталог", encoding="utf-8")

    artifact = extract_frames(media, target, fps=0.5, hwaccel="none")

    assert target.is_dir()
    assert artifact.frames
    assert not _leftovers(tmp_path), sorted(p.name for p in _leftovers(tmp_path))


# ==========================================================================
# Аппаратное ускорение — параметр, а не хардкод
# ==========================================================================


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        (("cuda", "videotoolbox", "vulkan"), "cuda"),  # nvdec целевой платформы
        (("videotoolbox", "vulkan"), "videotoolbox"),  # macOS без CUDA
        (("vulkan",), "none"),  # знакомых ускорителей нет
        ((), "none"),  # ffmpeg вообще без hwaccel
    ],
)
def test_resolve_hwaccel_auto_follows_preference_order(
    monkeypatch: pytest.MonkeyPatch, reported: tuple[str, ...], expected: str
) -> None:
    """``auto`` выбирает cuda -> videotoolbox -> CPU, а не «всегда none».

    Ожидаемое значение задано в тесте, а не вычислено той же функцией, что
    и в реализации: тихая потеря аппаратного декодирования тест ломает.
    """
    monkeypatch.setattr(ffmpeg_util, "available_hwaccels", lambda: reported)
    assert ffmpeg_util.resolve_hwaccel("auto") == expected


def test_resolve_hwaccel_none_is_cpu() -> None:
    assert resolve_hwaccel("none") == "none"


def test_explicit_hwaccel_returned_as_is_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Явно запрошенное доступное ускорение не подменяется предпочтением auto."""
    monkeypatch.setattr(ffmpeg_util, "available_hwaccels", lambda: ("cuda", "videotoolbox"))
    assert ffmpeg_util.resolve_hwaccel("videotoolbox") == "videotoolbox"


def test_explicit_unavailable_hwaccel_fails_loudly() -> None:
    """Явно запрошенное недоступное ускорение — ошибка, а не тихий CPU-фолбэк."""
    with pytest.raises(HwaccelUnavailableError):
        resolve_hwaccel("совершенно-несуществующий-hwaccel")


def test_frames_with_cpu_decoding(short_clip: Path, tmp_path: Path) -> None:
    media = probe(short_clip)
    artifact = extract_frames(media, tmp_path / "cpu", fps=1.0, hwaccel="none")
    _assert_frames_consistent(artifact, media.duration_s, 1.0)


@pytest.mark.gpu
def test_frames_with_nvdec(short_clip: Path, tmp_path: Path) -> None:
    """Проверяется только на целевой платформе WSL2 + RTX 4060."""
    media = probe(short_clip)
    artifact = extract_frames(media, tmp_path / "nvdec", fps=1.0, hwaccel="cuda")
    _assert_frames_consistent(artifact, media.duration_s, 1.0)
