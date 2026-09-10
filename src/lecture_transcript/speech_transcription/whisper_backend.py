"""ASR-бэкенд Whisper large-v3 (faster-whisper) — смешанная речь (D6).

Выбирается, когда в лекции много англоязычных терминов: в отличие от
GigaAM, Whisper распознаёт латиницу и не транслитерирует её. Пунктуацию
выдаёт сам, поэтому ``provides_punctuation = True`` и отдельная стадия
пунктуации для него не включается.

Глоссарий подаётся как ``initial_prompt`` — штатный для Whisper способ
подсказать написание терминов и имён собственных.

``faster_whisper`` импортируется лениво: без него модуль импортируется, а
:meth:`check_availability` объясняет, чего не хватает.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..contracts import Availability, Glossary, Word
from .base import IntervalAsrBackend

__all__ = ["WhisperBackend", "WHISPER_BACKEND_NAME", "build_initial_prompt"]

WHISPER_BACKEND_NAME = "whisper-large-v3"

_INSTALL_HINT = (
    "поставьте `pip install faster-whisper` (веса large-v3 ~4.7 ГБ качаются "
    "при первой загрузке) либо выберите другой ASR-бэкенд"
)

#: Сколько символов глоссария имеет смысл класть в initial_prompt.
#: Контекст Whisper ограничен 224 токенами — длинный список вытеснит сам себя.
MAX_PROMPT_CHARS = 800


class WhisperBackend(IntervalAsrBackend):
    """faster-whisper large-v3: смешанная речь, своя пунктуация, глоссарий в промпте."""

    name = WHISPER_BACKEND_NAME
    provides_punctuation = True
    supports_glossary = True

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = "ru",
        local_files_only: bool | None = None,
    ) -> None:
        super().__init__()
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._local_files_only = local_files_only
        self._model: Any | None = None

    # -- доступность ------------------------------------------------------

    def check_availability(self) -> Availability:
        if importlib.util.find_spec("faster_whisper") is None:
            return Availability(False, f"не установлен пакет 'faster_whisper': {_INSTALL_HINT}")
        return Availability(True)

    # -- жизненный цикл модели -------------------------------------------

    def _prepare(self, sample_rate: int) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # noqa: PLC0415 — ленивый импорт

        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            local_files_only=self._offline(),
        )

    def unload(self) -> None:
        """Освободить память под следующую стадию (D7)."""
        self._model = None

    def _offline(self) -> bool:
        """Скачивать веса запрещено, если явно указано или включён офлайн-режим."""
        if self._local_files_only is not None:
            return self._local_files_only
        return os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false", "False")

    # -- распознавание ----------------------------------------------------

    def _transcribe_chunk(
        self,
        samples: np.ndarray,
        sample_rate: int,
        glossary: Glossary | None,
    ) -> Sequence[Word]:
        """Распознать кусок с по-словными таймкодами; времена́ — от начала куска."""
        assert self._model is not None  # noqa: S101 — гарантировано _prepare
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        prompt = build_initial_prompt(glossary)
        if glossary and prompt is None:
            # Флаг used_glossary обязан отражать факт подачи, а не декларацию.
            self._glossary_rejected(
                "глоссарий не удалось положить в initial_prompt (пустые термины) — "
                "распознавание идёт без него, точность на терминологии ниже"
            )
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            initial_prompt=prompt,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return list(collect_words(segments))


def build_initial_prompt(glossary: Glossary | None) -> str | None:
    """Собрать initial_prompt из глоссария, не превышая полезной длины контекста.

    Термины подаются как есть, включая латиницу: именно написание из глоссария
    Whisper и должен воспроизвести вместо транслитерации.
    """
    if not glossary:
        return None
    parts: list[str] = []
    length = 0
    for term in glossary.terms:
        term = term.strip()
        if not term:
            continue
        addition = len(term) + 2
        if length + addition > MAX_PROMPT_CHARS:
            break
        parts.append(term)
        length += addition
    if not parts:
        return None
    return "Термины лекции: " + ", ".join(parts) + "."


def collect_words(segments: Any) -> list[Word]:
    """Разложить сегменты faster-whisper в плоский список слов.

    Если у сегмента нет по-словных таймкодов (модель их не вернула), сегмент
    отдаётся одним словом-фразой с временами сегмента — формат результата от
    этого не меняется.
    """
    words: list[Word] = []
    for segment in segments:
        segment_words = getattr(segment, "words", None)
        if segment_words:
            for word in segment_words:
                text = str(word.word).strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start_s=float(word.start),
                        end_s=float(word.end),
                        confidence=_probability(word),
                    )
                )
            continue
        text = str(getattr(segment, "text", "")).strip()
        if text:
            words.append(
                Word(text=text, start_s=float(segment.start), end_s=float(segment.end))
            )
    return words


def _probability(word: Any) -> float | None:
    value = getattr(word, "probability", None)
    return float(value) if value is not None else None
