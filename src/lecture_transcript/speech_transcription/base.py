"""Базовый ASR-бэкенд, распознающий речь по интервалам VAD.

Здесь живёт логика, одинаковая для всех бэкендов и потому не подлежащая
дублированию в GigaAM/Whisper:

- проверка доступности до начала прогона;
- нарезка аудио по речевым интервалам (длительная тишина в модель не идёт);
- пересчёт локальных таймкодов куска в абсолютные от начала записи;
- единый вид результата ``Transcription`` независимо от бэкенда.

Подкласс реализует ровно один метод — :meth:`IntervalAsrBackend._transcribe_chunk`,
который возвращает слова с временами **от начала куска**.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ..contracts import (
    AudioArtifact,
    Availability,
    BackendUnavailableError,
    Glossary,
    SpeechInterval,
    Transcription,
    Word,
)
from .audio_io import clear_audio_cache, read_wav_mono
from .timeline import slice_samples, split_interval, to_absolute

logger = logging.getLogger(__name__)

__all__ = ["IntervalAsrBackend"]


class IntervalAsrBackend:
    """Каркас ASR-бэкенда, реализующий контракт ``AsrBackend``."""

    name: str = ""
    provides_punctuation: bool = False
    supports_glossary: bool = False

    #: Предел длины куска, подаваемого в модель, секунды. ``None`` — предела нет.
    #: Интервал VAD длиннее предела режется на подкуски (см. ``split_interval``).
    max_chunk_s: float | None = None

    def __init__(self) -> None:
        self._run_warnings: list[str] = []
        self._glossary_used: bool = False
        self._glossary_refused: bool = False

    # -- контракт ---------------------------------------------------------

    def check_availability(self) -> Availability:  # pragma: no cover — переопределяется
        raise NotImplementedError

    def transcribe(
        self,
        audio: AudioArtifact,
        intervals: Sequence[SpeechInterval] | None = None,
        glossary: Glossary | None = None,
    ) -> Transcription:
        """Распознать речь на заданных интервалах, вернуть слова в абсолютном времени.

        ``intervals is None`` — интервалы не заданы (VAD не запускался),
        распознаётся вся запись целиком. Пустая последовательность — VAD
        отработал и речи не нашёл: в модель не идёт ничего, результат пуст,
        выдаётся предупреждение.
        """
        availability = self.check_availability()
        if not availability.available:
            raise BackendUnavailableError(
                f"ASR-бэкенд {self.name!r} недоступен: {availability.reason}"
            )

        self._run_warnings = []
        # Флаг глоссария ставится в момент фактической подачи, а не заранее:
        # нет кусков — нет и подачи.
        self._glossary_used = False
        self._glossary_refused = False
        passed = glossary if (glossary and self.supports_glossary) else None

        words: list[Word] = []
        try:
            samples, sample_rate = read_wav_mono(audio.path)
            if audio.sample_rate and sample_rate != audio.sample_rate:
                self._warn(
                    f"частота дискретизации WAV ({sample_rate} Гц) не совпадает с "
                    f"заявленной в артефакте ({audio.sample_rate} Гц) — "
                    "таймкоды могут быть смещены"
                )
            effective = self._effective_intervals(intervals, audio, samples, sample_rate)
            if effective:
                self._prepare(sample_rate)
                try:
                    for interval in effective:
                        for piece in split_interval(
                            samples, sample_rate, interval, self.max_chunk_s
                        ):
                            chunk = slice_samples(samples, sample_rate, piece)
                            if chunk.size == 0:
                                continue
                            local = self._transcribe_chunk(chunk, sample_rate, passed)
                            if passed is not None and not self._glossary_refused:
                                self._glossary_used = True
                            words.extend(
                                to_absolute(local, piece.start_s, clamp_to=piece)
                            )
                finally:
                    self._finalize()
        finally:
            # ASR — последний потребитель сэмплов записи: отпускаем кэш при
            # любом пути вызова (transcribe_speech, точка входа, адаптер cli).
            clear_audio_cache()

        words.sort(key=lambda word: (word.start_s, word.end_s))
        return Transcription(
            words=tuple(words),
            backend=self.name,
            has_punctuation=self.provides_punctuation,
            used_glossary=self._glossary_used,
            warnings=tuple(dict.fromkeys(self._run_warnings)),
        )

    def unload(self) -> None:
        """Выгрузить модель и освободить память (D7). По умолчанию нечего выгружать."""

    # -- точки расширения -------------------------------------------------

    def _transcribe_chunk(
        self,
        samples: np.ndarray,
        sample_rate: int,
        glossary: Glossary | None,
    ) -> Sequence[Word]:
        """Распознать один кусок. Времена́ слов — от начала куска."""
        raise NotImplementedError

    def _prepare(self, sample_rate: int) -> None:
        """Хук перед прогоном: сюда подкласс кладёт загрузку модели."""

    def _finalize(self) -> None:
        """Хук после прогона (модель здесь не выгружается — это делает ``unload``)."""

    # -- служебное --------------------------------------------------------

    def _warn(self, message: str) -> None:
        """Добавить предупреждение в результат и в лог."""
        self._run_warnings.append(message)
        logger.warning("%s: %s", self.name, message)

    def _glossary_rejected(self, message: str) -> None:
        """Отметить, что глоссарий фактически НЕ подан в модель.

        Флаг ``used_glossary`` обязан отражать реальность: бэкенд может
        объявлять поддержку глоссария, а установленная версия библиотеки —
        не иметь соответствующего параметра. Тогда флаг снимается, а причина
        уходит в предупреждения.
        """
        if not self._glossary_refused:
            self._glossary_refused = True
            self._glossary_used = False
            self._warn(message)

    def _effective_intervals(
        self,
        intervals: Sequence[SpeechInterval] | None,
        audio: AudioArtifact,
        samples: np.ndarray,
        sample_rate: int,
    ) -> list[SpeechInterval]:
        """Развести два разных случая, которые раньше сливались в «пустой список».

        ``None`` — интервалы не заданы, распознаём всю запись.
        Пустая последовательность — VAD речи не нашёл: в модель не идёт
        ничего (иначе тишина уезжает в ASR и Whisper на ней галлюцинирует),
        зато выдаётся предупреждение.
        """
        if intervals is None:
            duration = audio.duration_s or (
                samples.shape[0] / sample_rate if sample_rate else 0.0
            )
            return [SpeechInterval(0.0, duration)]
        ordered = sorted(intervals, key=lambda iv: iv.start_s)
        if not ordered:
            self._warn(
                "речь не обнаружена: VAD не вернул ни одного речевого интервала — "
                "распознавание не выполнялось, транскрипт пуст; "
                "проверьте аудиодорожку и параметры VAD"
            )
        return ordered
