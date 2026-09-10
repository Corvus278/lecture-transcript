"""Стадия speech-transcription: нормализованное аудио -> слова с таймкодами.

Публичный вход — :func:`transcribe_speech`. Остальное нужно тем, кто
собирает стадию по частям: реестр бэкендов (CLI показывает
:func:`available_backend_names`), VAD и стадия пунктуации.
"""

from __future__ import annotations

from .audio_io import (
    UnreadableAudioError,
    clear_audio_cache,
    read_wav_mono,
    write_wav_mono,
)
from .base import IntervalAsrBackend
from .gigaam_backend import GIGAAM_BACKEND_NAME, GIGAAM_MAX_CHUNK_S, GigaAmBackend
from .offline import OFFLINE_ENV, enable_offline_mode, is_offline, offline_mode
from .pipeline import (
    NO_PUNCTUATION_WARNING,
    SpeechTranscriptionConfig,
    glossary_for_backend,
    punctuate_words,
    transcribe_speech,
)
from .punctuation import (
    DEFAULT_PUNCTUATOR_NAME,
    PunctuationResult,
    Punctuator,
    SbertPuncCaseRu,
    align_punctuated,
    check_punctuator,
    get_punctuator,
    list_punctuator_names,
    register_punctuator,
    restore_punctuation,
)
from .stage import run_asr_stage, run_punctuation_stage, run_vad_stage
from .registry import (
    DEFAULT_BACKEND_NAME,
    available_backend_names,
    check_backend,
    ensure_available,
    get_backend,
    list_backend_names,
    register,
    reset_registry,
    unregister,
)
from .timeline import (
    is_monotonic,
    slice_samples,
    split_interval,
    to_absolute,
    validate_words,
)
from .vad import VadConfig, VadResult, check_vad_backend, detect_speech, run_vad
from .whisper_backend import WHISPER_BACKEND_NAME, WhisperBackend

__all__ = [
    # точки входа оркестратора (cli._stage_callable находит их по имени)
    "run_vad_stage",
    "run_asr_stage",
    "run_punctuation_stage",
    # стадия целиком
    "SpeechTranscriptionConfig",
    "transcribe_speech",
    "NO_PUNCTUATION_WARNING",
    "glossary_for_backend",
    "punctuate_words",
    # реестр ASR-бэкендов
    "DEFAULT_BACKEND_NAME",
    "available_backend_names",
    "check_backend",
    "ensure_available",
    "get_backend",
    "list_backend_names",
    "register",
    "reset_registry",
    "unregister",
    # бэкенды
    "IntervalAsrBackend",
    "GigaAmBackend",
    "GIGAAM_BACKEND_NAME",
    "GIGAAM_MAX_CHUNK_S",
    "WhisperBackend",
    "WHISPER_BACKEND_NAME",
    # VAD
    "VadConfig",
    "VadResult",
    "check_vad_backend",
    "detect_speech",
    "run_vad",
    # пунктуация
    "DEFAULT_PUNCTUATOR_NAME",
    "Punctuator",
    "PunctuationResult",
    "SbertPuncCaseRu",
    "align_punctuated",
    "check_punctuator",
    "get_punctuator",
    "list_punctuator_names",
    "register_punctuator",
    "restore_punctuation",
    # утилиты
    "OFFLINE_ENV",
    "UnreadableAudioError",
    "clear_audio_cache",
    "enable_offline_mode",
    "is_monotonic",
    "is_offline",
    "offline_mode",
    "read_wav_mono",
    "slice_samples",
    "split_interval",
    "to_absolute",
    "validate_words",
    "write_wav_mono",
]
