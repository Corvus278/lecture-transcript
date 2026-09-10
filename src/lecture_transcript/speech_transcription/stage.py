"""Точки входа оркестратора: ``run_vad_stage``, ``run_asr_stage``, ``run_punctuation_stage``.

Оркестратор (``cli._stage_callable``) сначала ищет эти функции в пакете и
только при их отсутствии берёт свой адаптер. Адаптер зовёт
``detect_speech`` / ``AsrBackend.transcribe`` / ``restore_punctuation``
напрямую, мимо :func:`transcribe_speech` — и на этом пути терялись громкий
отказ пунктуации, предупреждение о неподдерживаемом глоссарии и освобождение
кэша сэмплов после ASR. Здесь стадии собраны из тех же шагов, что и
``transcribe_speech``.

Контракт: ``run_<stage>_stage(ctx)``, где ``ctx`` — ``cli.StageContext``
(используются ``config``, ``results``, ``stage_dirs``, ``backend``); возврат —
JSON-нагрузка **той же формы**, что у адаптеров ``_adapt_vad`` /
``_adapt_asr`` / ``_adapt_punctuation``: её читает стадия ``merge``.

Пакет стадии не импортирует ни ``cli``, ни ``media_ingest`` (стадии общаются
только через ``contracts``), поэтому разбор нагрузки ``audio`` и
(де)сериализация ``Transcription`` повторены здесь в минимальном объёме.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contracts import AudioArtifact, Glossary, SpeechInterval, Transcription, Word
from .audio_io import UnreadableAudioError, clear_audio_cache
from .pipeline import glossary_for_backend, punctuate_words
from .registry import ensure_available
from .vad import VadConfig, run_vad

logger = logging.getLogger(__name__)

__all__ = ["run_vad_stage", "run_asr_stage", "run_punctuation_stage"]


def run_vad_stage(ctx: Any) -> dict[str, Any]:
    """Стадия ``vad``: речевые интервалы с абсолютными таймкодами.

    Сэмплы записи остаются в кэше ``audio_io`` намеренно: следующая стадия
    ``asr`` читает тот же WAV и отпускает кэш сама.
    """
    cfg = ctx.config.speech_transcription
    audio = _audio_artifact(ctx)
    result = run_vad(
        audio,
        VadConfig(
            min_silence_s=cfg.min_silence_s,
            min_speech_s=cfg.min_speech_s,
            speech_pad_s=cfg.speech_pad_s,
            silero_threshold=cfg.vad_threshold,
        ),
        cfg.vad_backend,
    )
    logger.info(
        "VAD (%s): %d речевых интервалов, %.1f с речи из %.1f с записи",
        result.backend,
        len(result.intervals),
        result.speech_duration_s,
        audio.duration_s,
    )
    return {"intervals": [[iv.start_s, iv.end_s] for iv in result.intervals]}


def run_asr_stage(ctx: Any) -> dict[str, Any]:
    """Стадия ``asr``: слова с таймкодами; кэш сэмплов отпускается в любом случае."""
    cfg = ctx.config.speech_transcription
    audio = _audio_artifact(ctx)
    intervals = [
        SpeechInterval(float(start), float(end))
        for start, end in ctx.results["vad"]["intervals"]
    ]
    glossary = None
    terms = (ctx.results.get("glossary") or {}).get("terms")
    if cfg.use_glossary and terms:
        glossary = Glossary(terms=tuple(terms))

    backend = ctx.backend or ensure_available(cfg.asr_backend)
    effective, glossary_warnings = glossary_for_backend(backend, glossary)
    try:
        transcription = backend.transcribe(audio, intervals, effective)
    finally:
        # ASR — последний потребитель сэмплов: 351 МБ на 91-минутной записи
        # не должны висеть под моделью пунктуации и на стадии merge.
        clear_audio_cache()
    if glossary_warnings:
        transcription = dataclasses.replace(
            transcription,
            warnings=tuple(dict.fromkeys((*glossary_warnings, *transcription.warnings))),
        )
    return transcription_json(transcription)


def run_punctuation_stage(ctx: Any) -> dict[str, Any]:
    """Стадия ``punctuation``: пунктуация и регистр, таймкоды не трогаются (D6)."""
    # Если asr взят из кэша, а vad считался в этом процессе, сэмплы ещё в кэше.
    clear_audio_cache()
    cfg = ctx.config.speech_transcription
    transcription = transcription_obj(ctx.results["asr"])
    mode = cfg.punctuation
    needed = mode == "always" or (mode == "auto" and not transcription.has_punctuation)
    if not needed:
        logger.info(
            "стадия punctuation: не требуется (режим %s, пунктуация от ASR: %s)",
            mode,
            transcription.has_punctuation,
        )
        if transcription.has_punctuation:
            return transcription_json(transcription)
        # режим never при тексте без пунктуации — это отказ, и он громкий
        _words, _applied, stage_warnings = punctuate_words(
            transcription.words, enabled=False
        )
        return transcription_json(_with_warnings(transcription, stage_warnings))

    words, applied, stage_warnings = punctuate_words(
        transcription.words, punctuator=ctx.backend
    )
    return transcription_json(
        dataclasses.replace(
            _with_warnings(transcription, stage_warnings),
            words=tuple(words),
            has_punctuation=bool(applied or transcription.has_punctuation),
        )
    )


# --------------------------------------------------------------------------
# Разбор и сборка нагрузок (форма — как у адаптеров cli)
# --------------------------------------------------------------------------


def _audio_artifact(ctx: Any) -> AudioArtifact:
    """Развернуть нагрузку стадии ``audio``; путь — относительно её каталога."""
    data = ctx.results["audio"]["audio"]
    path = Path(ctx.stage_dirs["audio"]) / data["path"]
    if not path.is_file():
        raise UnreadableAudioError(f"аудиоартефакт стадии audio не найден: {path}")
    return AudioArtifact(
        path=path,
        sample_rate=int(data["sample_rate"]),
        duration_s=float(data["duration_s"]),
        source_track_index=int(data["source_track_index"]),
    )


def _with_warnings(transcription: Transcription, extra: list[str]) -> Transcription:
    return dataclasses.replace(
        transcription,
        warnings=tuple(dict.fromkeys((*transcription.warnings, *extra))),
    )


def transcription_json(transcription: Transcription) -> dict[str, Any]:
    """``Transcription`` -> JSON-нагрузка кэша (форма ``cli._transcription_json``)."""
    return {
        "backend": transcription.backend,
        "has_punctuation": transcription.has_punctuation,
        "used_glossary": transcription.used_glossary,
        "warnings": list(transcription.warnings),
        "words": [
            [w.text, w.start_s, w.end_s, w.confidence] for w in transcription.words
        ],
    }


def transcription_obj(data: Mapping[str, Any]) -> Transcription:
    """JSON-нагрузка -> ``Transcription`` (обратная к :func:`transcription_json`)."""
    return Transcription(
        backend=data["backend"],
        has_punctuation=bool(data["has_punctuation"]),
        used_glossary=bool(data["used_glossary"]),
        warnings=tuple(data.get("warnings", ())),
        words=tuple(
            Word(
                text=w[0],
                start_s=float(w[1]),
                end_s=float(w[2]),
                confidence=None if w[3] is None else float(w[3]),
            )
            for w in data["words"]
        ),
    )
