"""Пересчёт таймкодов слов в абсолютные и проверки временно́й шкалы.

ASR-бэкенды распознают куски аудио, вырезанные по речевым интервалам, и
выдают времена от начала куска. Абсолютность таймкодов (требование
«Слова с временны́ми метками») обеспечивается ровно здесь — в одном месте,
общем для всех бэкендов.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..contracts import SpeechInterval, Word

__all__ = [
    "slice_samples",
    "split_interval",
    "to_absolute",
    "is_monotonic",
    "validate_words",
]


def slice_samples(
    samples: np.ndarray, sample_rate: int, interval: SpeechInterval
) -> np.ndarray:
    """Вырезать участок аудио по интервалу речи (границы клампятся в массив)."""
    start = max(0, int(round(interval.start_s * sample_rate)))
    end = min(samples.shape[0], int(round(interval.end_s * sample_rate)))
    if end <= start:
        return samples[:0]
    return samples[start:end]


def split_interval(
    samples: np.ndarray,
    sample_rate: int,
    interval: SpeechInterval,
    max_chunk_s: float | None,
    *,
    search_s: float | None = None,
    frame_s: float = 0.02,
) -> list[SpeechInterval]:
    """Нарезать речевой интервал на куски не длиннее ``max_chunk_s``.

    Интервал VAD ограничен только паузами: при ``min_silence_s = 1.0`` на
    реальной лекции он спокойно тянется на минуты. Модели ASR при этом имеют
    жёсткий предел длины входа (GigaAM 0.1.0 — 25 с, дальше ``ValueError``),
    поэтому длинный интервал режется здесь, а не в бэкенде.

    Точка разреза ищется по минимуму энергии во второй половине куска
    (или в последних ``search_s`` секундах, если задано): так разрез
    попадает в паузу между словами, если она там вообще есть. Среди почти
    одинаково тихих кадров берётся самый поздний — куски не мельчают. Таймкоды кусков — абсолютные, от начала
    записи, поэтому пересчёт времён в :func:`to_absolute` не меняется.

    ``max_chunk_s is None`` (или ``<= 0``) означает «резать не нужно».
    """
    if max_chunk_s is None or max_chunk_s <= 0:
        return [interval]
    if interval.end_s - interval.start_s <= max_chunk_s:
        return [interval]

    pieces: list[SpeechInterval] = []
    cursor = interval.start_s
    while interval.end_s - cursor > max_chunk_s:
        target = cursor + max_chunk_s
        # Окно поиска — вторая половина куска: кусок не короче половины
        # предела, а пауза ищется на 10 с, а не на последних 0.75 с.
        span = max_chunk_s / 2.0 if search_s is None else min(search_s, max_chunk_s / 2.0)
        window_start = target - span
        cut = _quietest_point(samples, sample_rate, window_start, target, frame_s)
        if not cursor < cut <= target:  # защита от зацикливания
            cut = target
        pieces.append(SpeechInterval(cursor, cut))
        cursor = cut
    pieces.append(SpeechInterval(cursor, interval.end_s))
    return pieces


def _quietest_point(
    samples: np.ndarray,
    sample_rate: int,
    window_start_s: float,
    window_end_s: float,
    frame_s: float,
) -> float:
    """Момент минимальной энергии в окне (середина самого тихого кадра)."""
    start = max(0, int(round(window_start_s * sample_rate)))
    end = min(samples.shape[0], int(round(window_end_s * sample_rate)))
    frame_len = max(1, int(round(frame_s * sample_rate)))
    frame_count = (end - start) // frame_len
    if frame_count < 2:  # окна нет (кусок за концом файла) — режем по границе
        return window_end_s
    block = samples[start : start + frame_count * frame_len].reshape(frame_count, frame_len)
    # einsum считает сумму квадратов сразу в float64, не материализуя квадраты.
    energy = np.einsum("ij,ij->i", block, block, dtype=np.float64)
    # Кадры в пределах 3 дБ от самого тихого — равноценные паузы; берём последний.
    candidates = np.flatnonzero(energy <= float(energy.min()) * 2.0 + 1e-12)
    index = int(candidates[-1])
    cut = window_start_s + (index + 0.5) * frame_s
    return min(max(cut, window_start_s), window_end_s)


def to_absolute(
    words: Iterable[Word],
    offset_s: float,
    *,
    clamp_to: SpeechInterval | None = None,
) -> list[Word]:
    """Сдвинуть локальные таймкоды слов на начало куска.

    `clamp_to` ограничивает результат границами интервала: локальное время
    за пределами куска — ошибка бэкенда, но она не должна ломать монотонность.
    """
    shifted: list[Word] = []
    for word in words:
        start = word.start_s + offset_s
        end = word.end_s + offset_s
        if clamp_to is not None:
            start = min(max(start, clamp_to.start_s), clamp_to.end_s)
            end = min(max(end, start), clamp_to.end_s)
        if end < start:
            end = start
        shifted.append(
            Word(
                text=word.text,
                start_s=float(start),
                end_s=float(end),
                confidence=word.confidence,
            )
        )
    return shifted


def is_monotonic(words: Sequence[Word]) -> bool:
    """Времена́ начала слов не убывают и ни одно слово не кончается раньше начала.

    Вырожденные слова (``end_s == start_s``) монотонность не нарушают —
    их ловит :func:`validate_words` отдельным замечанием.
    """
    previous = float("-inf")
    for word in words:
        if word.start_s < previous or word.end_s < word.start_s:
            return False
        previous = word.start_s
    return True


def validate_words(words: Sequence[Word], duration_s: float, *, tolerance_s: float = 0.05) -> list[str]:
    """Проверить последовательность слов; вернуть список замечаний (пустой — всё хорошо)."""
    problems: list[str] = []
    if not is_monotonic(words):
        problems.append("времена́ слов не монотонны")
    for word in words:
        if word.start_s < -tolerance_s:
            problems.append(f"слово {word.text!r} начинается до начала записи ({word.start_s:.3f} с)")
            break
    degenerate = [word for word in words if word.end_s <= word.start_s]
    if degenerate:
        problems.append(
            f"{len(degenerate)} слов нулевой длительности "
            f"(первое — {degenerate[0].text!r} на {degenerate[0].start_s:.3f} с): "
            "таймкоды схлопнуты в точку"
        )
    for word in words:
        if word.end_s > duration_s + tolerance_s:
            problems.append(
                f"слово {word.text!r} выходит за длительность записи "
                f"({word.end_s:.3f} с > {duration_s:.3f} с)"
            )
            break
    return problems
