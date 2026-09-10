"""Слияние речи и слайдов по единой оси времени (задачи 6.1, 6.2).

Стадия `merge` получает слова ASR (`Transcription.words`) и логические слайды
(`Slide` + опциональный `SlideOcr`) и раскладывает слова по секциям
(`TranscriptSection`).

Порядок стадий сборки (важен для инварианта полноты):

    merge  ->  consolidate  ->  cleanup  ->  paragraphs  ->  render

Полнота (6.1) проверяется **до** чистки: на выходе `merge_sections` объединение
слов всех секций в порядке следования равно исходному кортежу `words` —
ни одно слово не потеряно и не продублировано. Это инвариант, он проверяется
в коде (`_assert_completeness`, явный `raise`) и тестом. Чистка филлеров (6.3)
выполняется уже поверх собранных секций и намеренно удаляет слова, поэтому
после неё инвариант полноты не действует.

Правило привязки слова к слайду
-------------------------------
Слово относится к тому слайду, с интервалом которого у него наибольшее
пересечение по времени. При равном пересечении побеждает слайд с меньшим
`index`. Слово, не пересекающееся ни с одним слайдом, получает метку `None` —
такая речь уходит в отдельные секции без слайда и не теряется.

Правило допуска на границах секций (6.2)
----------------------------------------
Привязка по пересечению режет фразу ровно в момент переключения слайда.
Поэтому после первичной разметки каждая граница между соседними секциями
сдвигается к ближайшей **границе фразы**:

* граница фразы — это промежуток между соседними словами длиной
  >= `phrase_gap_s`, либо конец предложения (предыдущее слово оканчивается
  на `.`, `?`, `!`, `…`);
* граница сдвигается не более чем на `tolerance_s` секунд от исходной точки
  разреза (`section_boundary_tolerance_s` в конфиге, дефолт 2.0 с,
  ПЛЕЙСХОЛДЕР — подбирается на эталонной записи, задача 7.6);
* из подходящих кандидатов берётся ближайший по времени к исходной точке,
  при равенстве — более ранний (детерминированность);
* сдвиг не может опустошить ни одну из двух соседних секций: у каждой
  остаётся хотя бы одно слово. Именно это ограничение не даёт допуску
  «съесть» короткий слайд целиком даже при большом `tolerance_s`;
* если границы фразы в пределах допуска нет, разрез остаётся на месте:
  допуск ограничивает ущерб, но не отменяет переключение слайда.

Слайд, во время которого никто не говорил, всё равно порождает секцию с
пустым списком слов: изображение слайда — тоже содержание. Секции выходят
в хронологическом порядке (по началу интервала секции), поэтому такой немой
слайд стоит между соседями по времени, даже если допуск 6.2 увёл слова
соседней секции раньше его начала.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..contracts import PipelineError, Slide, SlideOcr, TranscriptSection, Word

__all__ = [
    "SENTENCE_END_CHARS",
    "BoundaryPolicy",
    "merge_sections",
]

# Символы конца предложения: после стадии пунктуации (design D6) именно они
# отмечают надёжную смысловую границу.
SENTENCE_END_CHARS: tuple[str, ...] = (".", "?", "!", "…")


@dataclass(frozen=True)
class BoundaryPolicy:
    """Параметры допуска на границах секций (6.2).

    tolerance_s: максимальный сдвиг границы секции от точки переключения
        слайда, с. Дефолт совпадает с `TranscriptAssemblyConfig
        .section_boundary_tolerance_s`.
    phrase_gap_s: пауза между словами, начиная с которой промежуток считается
        границей фразы, с. 0.45 с — типичная межфразовая пауза лектора:
        меньше — дыхание внутри фразы, больше — уже смысловой разрыв.
    """

    tolerance_s: float = 2.0
    phrase_gap_s: float = 0.45


def merge_sections(
    words: Sequence[Word],
    slides: Sequence[Slide],
    slide_ocrs: Sequence[SlideOcr] = (),
    policy: BoundaryPolicy | None = None,
) -> tuple[TranscriptSection, ...]:
    """Разложить слова по секциям слайдов и секциям речи вне слайдов.

    Возвращает секции в порядке следования по записи. Гарантия полноты:
    конкатенация `section.words` по всем секциям равна `tuple(words)`.
    """
    policy = policy or BoundaryPolicy()
    words = tuple(words)
    ordered_slides = tuple(sorted(slides, key=lambda s: (s.start_s, s.index)))
    ocr_by_index = {ocr.slide_index: ocr for ocr in slide_ocrs}

    labels = _assign_labels(words, ordered_slides)
    labels = _shift_boundaries(words, labels, policy)
    sections = _build_sections(words, labels, ordered_slides, ocr_by_index)
    _assert_completeness(words, sections)
    return sections


# --------------------------------------------------------------------------
# Первичная разметка
# --------------------------------------------------------------------------


def _assign_labels(
    words: Sequence[Word], slides: Sequence[Slide]
) -> list[int | None]:
    """Метка слова — позиция слайда в `slides` либо None (речь вне слайдов).

    Слово нулевой длительности (`end_s <= start_s`; ASR пропускает такие слова
    дальше с предупреждением) привязывается по точке `start_s` — см.
    `_point_label`. По пересечению у него всегда 0, и без этого правила оно
    уходило бы в «Речь вне слайдов», разрывая секцию слайда надвое.
    """
    if not slides:
        return [None] * len(words)

    starts = [s.start_s for s in slides]
    # Наибольший конец среди slides[0..pos]. Если он не дотягивает до слова,
    # левее пересечений нет — обход можно прервать. Прерывать по концу одного
    # текущего слайда нельзя: левее может стоять слайд с большим `end_s`
    # (вложенный интервал), и правило tie-break по `index` нарушилось бы.
    prefix_end: list[float] = []
    running_end = float("-inf")
    for slide in slides:
        running_end = max(running_end, slide.end_s)
        prefix_end.append(running_end)

    labels: list[int | None] = []
    for word in words:
        if word.end_s <= word.start_s:
            labels.append(_point_label(word.start_s, slides, starts, prefix_end))
            continue
        best_pos: int | None = None
        best_overlap = 0.0
        # Кандидаты — слайды, начавшиеся не позже конца слова; идём влево,
        # пока есть шанс на пересечение.
        upper = bisect.bisect_right(starts, word.end_s)
        for pos in range(upper - 1, -1, -1):
            if prefix_end[pos] <= word.start_s:
                break
            slide = slides[pos]
            overlap = min(word.end_s, slide.end_s) - max(word.start_s, slide.start_s)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pos = pos
            elif overlap == best_overlap and overlap > 0.0 and best_pos is not None:
                # При равном пересечении побеждает меньший slide.index.
                if slide.index < slides[best_pos].index:
                    best_pos = pos
        labels.append(best_pos if best_overlap > 0.0 else None)
    return labels


def _point_label(
    t: float,
    slides: Sequence[Slide],
    starts: Sequence[float],
    prefix_end: Sequence[float],
) -> int | None:
    """Метка точки `t`: слайд, интервалу которого она принадлежит.

    Пересечение нулевой длины — это вложенность, а не его отсутствие.
    Предпочтение — полуоткрытому интервалу `start_s <= t < end_s` (точка на
    моменте переключения относится к следующему слайду); точка ровно на
    `end_s` относится к слайду, только если никакому полуоткрытому интервалу
    она не принадлежит (конец последнего слайда). При равенстве — меньший
    `index`.
    """
    best_pos: int | None = None
    best_key: tuple[int, int] | None = None
    upper = bisect.bisect_right(starts, t)
    for pos in range(upper - 1, -1, -1):
        if prefix_end[pos] < t:
            break
        slide = slides[pos]
        if slide.end_s < t:
            continue
        key = (0 if t < slide.end_s else 1, slide.index)
        if best_key is None or key < best_key:
            best_key = key
            best_pos = pos
    return best_pos


# --------------------------------------------------------------------------
# Допуск на границах (6.2)
# --------------------------------------------------------------------------


def _runs(labels: Sequence[int | None]) -> list[tuple[int, int, int | None]]:
    """Непрерывные участки одинаковой метки: (start, end, label), end исключён."""
    runs: list[tuple[int, int, int | None]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def _is_phrase_boundary(
    words: Sequence[Word], cut: int, policy: BoundaryPolicy
) -> bool:
    """Можно ли резать речь перед словом `cut`, не разрывая фразу."""
    if cut <= 0 or cut >= len(words):
        return False
    gap = words[cut].start_s - words[cut - 1].end_s
    if gap >= policy.phrase_gap_s:
        return True
    return words[cut - 1].text.rstrip().endswith(SENTENCE_END_CHARS)


def _shift_boundaries(
    words: Sequence[Word],
    labels: list[int | None],
    policy: BoundaryPolicy,
) -> list[int | None]:
    """Сдвинуть границы секций к ближайшим границам фраз в пределах допуска."""
    runs = _runs(labels)
    if len(runs) < 2:
        return labels

    original_cuts = [run[0] for run in runs[1:]]
    new_cuts: list[int] = []
    for idx, cut in enumerate(original_cuts):
        # Слева нельзя зайти дальше уже сдвинутой предыдущей границы,
        # справа — дальше следующей исходной. Обе соседние секции обязаны
        # сохранить хотя бы по одному слову.
        low = (new_cuts[-1] if new_cuts else 0) + 1
        high = (
            original_cuts[idx + 1] if idx + 1 < len(original_cuts) else len(words)
        ) - 1
        new_cuts.append(_best_cut(words, cut, low, high, policy))

    shifted: list[int | None] = list(labels)
    bounds = [0, *new_cuts, len(words)]
    for run_pos, run in enumerate(runs):
        for i in range(bounds[run_pos], bounds[run_pos + 1]):
            shifted[i] = run[2]
    return shifted


def _best_cut(
    words: Sequence[Word],
    cut: int,
    low: int,
    high: int,
    policy: BoundaryPolicy,
) -> int:
    """Ближайшая к `cut` граница фразы в окне допуска; сам `cut`, если её нет."""
    if low > high or policy.tolerance_s <= 0.0:
        return cut
    anchor = words[cut].start_s if cut < len(words) else words[-1].end_s
    best = cut
    best_key: tuple[float, int] | None = None
    if _is_phrase_boundary(words, cut, policy) and low <= cut <= high:
        best_key = (0.0, cut)
    for candidate in range(low, high + 1):
        if candidate == cut or not _is_phrase_boundary(words, candidate, policy):
            continue
        distance = abs(words[candidate].start_s - anchor)
        if distance > policy.tolerance_s:
            continue
        key = (distance, candidate)
        if best_key is None or key < best_key:
            best_key = key
            best = candidate
    return best


# --------------------------------------------------------------------------
# Сборка секций
# --------------------------------------------------------------------------


def _build_sections(
    words: Sequence[Word],
    labels: Sequence[int | None],
    slides: Sequence[Slide],
    ocr_by_index: dict[int, SlideOcr],
) -> tuple[TranscriptSection, ...]:
    runs = [run for run in _runs(labels) if run[1] > run[0]]

    # Секции идут по хронологии, а не по индексу первого слова: после сдвига
    # границы (6.2) run слайда N может начинаться со слова, которое по времени
    # раньше немого слайда N-1, и тот уезжал в конец транскрипта
    # («Слайд 1 -> Слайд 3 -> Слайд 2»).
    #
    # Время run'а монотонизируется (max с предыдущим), поэтому ключи run'ов
    # неубывающие и стабильная сортировка не может переставить их между собой.
    # Это защищает инвариант полноты 6.1: конкатенация слов секций остаётся
    # равной входному кортежу при любом, даже невалидном, наборе слайдов.
    # Сортировка расставляет только немые слайды относительно run'ов.
    #
    # (ключ сортировки, слайд, слова)
    entries: list[tuple[tuple[float, int, int], Slide | None, tuple[Word, ...]]] = []
    seen_slides: set[int] = set()
    running_start = float("-inf")
    for start, end, label in runs:
        slide = slides[label] if label is not None else None
        if slide is not None:
            seen_slides.add(label)  # type: ignore[arg-type]
        own_start = slide.start_s if slide is not None else words[start].start_s
        running_start = max(running_start, own_start)
        entries.append(((running_start, 1, start), slide, tuple(words[start:end])))

    for pos, slide in enumerate(slides):
        if pos in seen_slides:
            continue
        entries.append(((slide.start_s, 0, slide.index), slide, ()))

    entries.sort(key=lambda item: item[0])

    sections: list[TranscriptSection] = []
    for _, slide, section_words in entries:
        if slide is not None:
            start_s, end_s = slide.start_s, slide.end_s
        else:
            start_s = section_words[0].start_s
            end_s = section_words[-1].end_s
        sections.append(
            TranscriptSection(
                start_s=start_s,
                end_s=end_s,
                slide=slide,
                slide_ocr=ocr_by_index.get(slide.index) if slide else None,
                words=section_words,
            )
        )
    return tuple(sections)


def _assert_completeness(
    words: Sequence[Word], sections: Iterable[TranscriptSection]
) -> None:
    """Инвариант 6.1: слова секций в порядке следования == исходные слова.

    Проверка намеренно сделана явным `raise`, а не `assert`: `assert`
    выключается флагом `-O`/`PYTHONOPTIMIZE`, и в оптимизированном прогоне
    потеря слова прошла бы молча.
    """
    merged = tuple(word for section in sections for word in section.words)
    if merged != tuple(words):
        raise PipelineError(
            "нарушен инвариант полноты merge: "
            f"{len(merged)} слов в секциях против {len(words)} на входе"
        )
