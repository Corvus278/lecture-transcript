"""Стадия распознавания речи целиком: аудио -> ``Transcription``.

    аудио -> VAD -> ASR(глоссарий, если поддерживается) -> пунктуация -> слова

Порядок и условия соответствуют design:

- D2: глоссарий из слайдов подаётся в ASR (hotwords / initial_prompt);
- D6: стадия пунктуации включается только для бэкендов с
  ``provides_punctuation is False``; таймкоды после неё не пересчитываются;
- D7: модели выгружаются между стадиями (``unload``), чтобы ASR и пунктуатор
  не жили в памяти одновременно.

Недоступный ASR-бэкенд отвергается в самом начале, до чтения аудио.
Бэкенд без поддержки глоссария не ломает прогон: распознавание идёт без
глоссария, а в ``Transcription.warnings`` попадает предупреждение о снижении
точности на терминологии.

Шаги «глоссарий под бэкенд» и «пунктуация» вынесены в
:func:`glossary_for_backend` и :func:`punctuate_words`: ими же пользуются
точки входа оркестратора (``stage.py``), чтобы поведение — в том числе
громкий отказ пунктуации — не зависело от того, каким путём запущена стадия.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..contracts import (
    AsrBackend,
    AudioArtifact,
    Availability,
    Glossary,
    Transcription,
    Word,
)
from .audio_io import clear_audio_cache
from .offline import enable_offline_mode
from .punctuation import (
    DEFAULT_PUNCTUATOR_NAME,
    NO_PUNCTUATION_WARNING,
    Punctuator,
    check_punctuator,
    get_punctuator,
    restore_punctuation,
)
from .registry import DEFAULT_BACKEND_NAME, ensure_available
from .timeline import validate_words
from .vad import VadBackendName, VadConfig, run_vad

logger = logging.getLogger(__name__)

__all__ = [
    "SpeechTranscriptionConfig",
    "transcribe_speech",
    "glossary_for_backend",
    "punctuate_words",
    "NO_PUNCTUATION_WARNING",
]


@dataclass(frozen=True)
class SpeechTranscriptionConfig:
    """Настройки стадии распознавания речи."""

    asr_backend: str = DEFAULT_BACKEND_NAME
    vad_backend: VadBackendName = "auto"
    vad: VadConfig = field(default_factory=VadConfig)
    restore_punctuation: bool = True
    punctuator: str = DEFAULT_PUNCTUATOR_NAME
    offline: bool = True


def transcribe_speech(
    audio: AudioArtifact,
    glossary: Glossary | None = None,
    config: SpeechTranscriptionConfig | None = None,
    *,
    backend: AsrBackend | None = None,
    punctuator: Punctuator | None = None,
) -> Transcription:
    """Прогнать аудио через VAD, ASR и (если нужно) пунктуацию.

    ``backend`` / ``punctuator`` позволяют подставить готовый объект в обход
    реестра — этим пользуются тесты и вызывающий код, собравший бэкенд сам.
    """
    cfg = config or SpeechTranscriptionConfig()
    warnings: list[str] = []

    if cfg.offline:
        enable_offline_mode()

    # 1. Бэкенд проверяется до любой работы: недоступный отвергается сразу.
    asr = backend if backend is not None else ensure_available(cfg.asr_backend)

    # 2. Детекция речи: длительная тишина в распознавание не попадает.
    vad_result = run_vad(audio, cfg.vad, cfg.vad_backend)
    warnings.extend(vad_result.warnings)
    logger.info(
        "VAD (%s): %d речевых интервалов, %.1f с речи из %.1f с записи",
        vad_result.backend,
        len(vad_result.intervals),
        vad_result.speech_duration_s,
        audio.duration_s,
    )

    # 3. Глоссарий: бэкенд без поддержки работает без него, но с предупреждением.
    effective_glossary, glossary_warnings = glossary_for_backend(asr, glossary)
    warnings.extend(glossary_warnings)

    # 4. Распознавание. Сэмплы записи отпускаются в любом случае, даже при сбое.
    try:
        result = asr.transcribe(audio, vad_result.intervals, effective_glossary)
    finally:
        clear_audio_cache()
    warnings.extend(result.warnings)
    words = list(result.words)
    asr.unload()  # D7: освобождаем память до загрузки модели пунктуации

    # 5. Пунктуация — только для бэкендов без своей (D6).
    has_punctuation = bool(result.has_punctuation)
    if not asr.provides_punctuation:
        words, has_punctuation, stage_warnings = punctuate_words(
            words,
            enabled=cfg.restore_punctuation,
            punctuator_name=cfg.punctuator,
            punctuator=punctuator,
        )
        warnings.extend(stage_warnings)

    # 6. Проверка временно́й шкалы перед выдачей наружу.
    warnings.extend(validate_words(words, audio.duration_s))

    return Transcription(
        words=tuple(words),
        backend=asr.name,
        has_punctuation=has_punctuation,
        used_glossary=bool(result.used_glossary),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def glossary_for_backend(
    asr: AsrBackend, glossary: Glossary | None
) -> tuple[Glossary | None, list[str]]:
    """Сценарий 5.5: бэкенд без поддержки глоссария работает без него, с предупреждением."""
    if glossary and not asr.supports_glossary:
        message = (
            f"ASR-бэкенд {asr.name!r} не поддерживает глоссарий — "
            f"распознавание выполнено без него ({len(glossary.terms)} терминов "
            "не подано), точность на терминологии и именах собственных ниже"
        )
        logger.warning(message)
        return None, [message]
    return glossary, []


def punctuate_words(
    words: Sequence[Word],
    *,
    enabled: bool = True,
    punctuator_name: str = DEFAULT_PUNCTUATOR_NAME,
    punctuator: Punctuator | None = None,
) -> tuple[list[Word], bool, list[str]]:
    """Восстановить пунктуацию и регистр, сохранив таймкоды слов (D6).

    Возвращает (слова, пунктуация применена, предупреждения). Никогда не
    бросает из-за пунктуатора: к этому моменту ASR уже отработал, и терять
    его результат нельзя. Любой отказ — полный или частичный — громкий:
    маркер в предупреждениях и запись в лог уровня ERROR.
    """
    words = list(words)
    if not words:
        return words, False, []
    if not enabled:
        return words, False, _refusal("восстановление пунктуации отключено конфигом")

    if punctuator is None:
        label = punctuator_name
        availability = check_punctuator(punctuator_name)
    else:
        label = getattr(punctuator, "name", punctuator_name)
        try:
            availability = punctuator.check_availability()
        except Exception as exc:  # noqa: BLE001 — сломанная проверка = недоступность
            availability = Availability(False, f"проверка доступности упала: {exc!r}")
    if not availability.available:
        return words, False, _refusal(
            f"пунктуатор {label!r} недоступен ({availability.reason}) — "
            "текст остаётся без пунктуации и в едином регистре"
        )
    if punctuator is None:
        punctuator = get_punctuator(punctuator_name)

    try:
        stage = restore_punctuation(words, punctuator)
    finally:
        try:
            punctuator.unload()  # D7
        except Exception as exc:  # noqa: BLE001 — выгрузка не роняет прогон
            logger.warning("не удалось выгрузить пунктуатор %r: %s", label, exc)
    return list(stage.words), stage.applied, list(stage.warnings)


def _refusal(reason: str) -> list[str]:
    """Причина отказа пунктуации + единый громкий маркер."""
    logger.warning("%s", reason)
    logger.error("%s", NO_PUNCTUATION_WARNING)
    return [reason, NO_PUNCTUATION_WARNING]
