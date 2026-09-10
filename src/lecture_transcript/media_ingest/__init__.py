"""Стадия media-ingest: запись лекции -> нормализованное аудио и кадры видео.

Публичный API:

* `probe` — состав дорожек и длительность входного файла;
* `extract_audio` — WAV PCM s16le, 16 кГц, моно;
* `extract_frames` — кадры видеотрека с заданной частотой и таймкодами;
* `run_audio_stage` / `run_frames_stage` — две независимые точки входа для
  оркестратора (`cli.StageContext`). Они подключают приём медиа к кэшу
  (design D8) двумя раздельными ячейками: смена аудиодорожки не
  инвалидирует кадры, смена частоты выборки не трогает WAV.

Типы данных берутся из `lecture_transcript.contracts` и реэкспортируются
здесь для удобства вызывающего кода.
"""

from __future__ import annotations

from ..contracts import (
    AudioArtifact,
    AudioTrackInfo,
    Frame,
    FramesArtifact,
    MediaInfo,
    MissingAudioTrackError,
    MissingVideoTrackError,
    PipelineError,
    UnreadableMediaError,
    VideoStreamInfo,
)
from ._ffmpeg import (
    DEFAULT_HWACCEL,
    HwaccelUnavailableError,
    MediaProcessingError,
    available_hwaccels,
    resolve_hwaccel,
)
from .audio import TARGET_SAMPLE_RATE, extract_audio, select_audio_track
from .frames import ImageFormat, extract_frames
from .probe import probe
from .stage import (
    AUDIO_NAME,
    FRAMES_DIR_NAME,
    audio_artifact,
    frames_artifact,
    run_audio_stage,
    run_frames_stage,
    run_stage,
)

__all__ = [
    # операции
    "probe",
    "extract_audio",
    "extract_frames",
    "select_audio_track",
    # точки входа оркестратора (две независимые стадии)
    "run_audio_stage",
    "run_frames_stage",
    "run_stage",
    "audio_artifact",
    "frames_artifact",
    "AUDIO_NAME",
    "FRAMES_DIR_NAME",
    # типы контракта
    "MediaInfo",
    "AudioTrackInfo",
    "VideoStreamInfo",
    "AudioArtifact",
    "Frame",
    "FramesArtifact",
    "ImageFormat",
    # ошибки
    "PipelineError",
    "UnreadableMediaError",
    "MissingAudioTrackError",
    "MissingVideoTrackError",
    "MediaProcessingError",
    "HwaccelUnavailableError",
    # аппаратное ускорение
    "DEFAULT_HWACCEL",
    "available_hwaccels",
    "resolve_hwaccel",
    "TARGET_SAMPLE_RATE",
]
