"""Укрупнение секций после слияния: транскрипт как исходник конспекта.

Порядок стадий сборки:

    merge  ->  consolidate  ->  cleanup  ->  paragraphs  ->  render

Сухой прогон на эталонной записи показал, что секции, которые выдаёт `merge`,
рвут поток чтения. Здесь два правила, применяемые строго по порядку:

1. **Короткая «речь вне слайдов» присоединяется к соседнему слайду**
   (`absorb_short_offslide_sections`, структурно — меняет секции).
   На платформе записи каждая смена слайда сопровождается провалом
   демонстрации на 3–13 с (DECISIONS 5.12). Речь в этом провале по смыслу —
   хвост предыдущего слайда или начало следующего, а не речь без
   демонстрации. Секция вне слайдов короче `short_offslide_section_s`
   присоединяется к **предыдущей** секции слайда, если её нет — к
   **следующей**. Длинная секция вне слайдов (например, вступление до первого
   слайда) остаётся отдельной: это настоящая речь без демонстрации, которой
   требует спека 6.1. Сужение формулировки спеки до этого случая —
   сознательное отклонение, записанное в DECISIONS. Полнота 6.1 сохраняется:
   слова целиком переходят в соседнюю секцию, порядок не меняется, это
   проверяется явно (`_assert_completeness`).
2. **Соседние секции слайдов с одинаковым содержимым показываются одной
   секцией** (`group_same_content_slides`, применяет рендер). Это
   группировка для вывода, а не слияние `TranscriptSection`: у секции в
   контракте один `slide`, а у группы может быть несколько различающихся PNG.
   Гибридный OCR выбрасывает рукописное (design D5), поэтому одинаковый
   `SlideOcr.markdown` бывает у состояний слайда с разным рукописным решением
   — показать только последний PNG значило бы потерять решение. В
   `Transcript.sections` каждый слайд остаётся со своим PNG и интервалом.
   «Одинаковое содержимое» — совпадение `content_signature`:
   `SlideOcr.markdown` без ссылки на изображение, без пометок ненадёжности,
   с нормализованными пробелами. Пустой или отсутствующий OCR одинаковым
   содержимым не считается: два нераспознанных слайда могут быть разными.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..contracts import SlideOcr, TranscriptSection
from .merge import _assert_completeness

__all__ = [
    "OCR_LOW_CONFIDENCE_MARK",
    "ConsolidationPolicy",
    "absorb_short_offslide_sections",
    "consolidate_sections",
    "content_signature",
    "group_same_content_slides",
    "ocr_body",
]

# Пометка неуверенного фрагмента в формате slide-ocr. Продублирована здесь,
# а не импортирована: код стадий не импортирует другие стадии.
OCR_LOW_CONFIDENCE_MARK = "_(?)_"

_IMAGE_LINK = re.compile(r"!\[[^\]]*\]\([^)]*\)")


@dataclass(frozen=True)
class ConsolidationPolicy:
    """Параметры укрупнения секций.

    short_offslide_section_s: секция вне слайдов короче этого порога, с,
        присоединяется к соседней секции слайда. Дефолт 15 с: провал
        демонстрации при смене слайда на платформе записи длится 3–13 с
        (DECISIONS 5.12), в сухом прогоне 44 из 45 секций вне слайдов короче
        15 с, а самая короткая настоящая речь без демонстрации (вступление)
        — 26 с. Порог выше самого длинного провала с запасом и ниже самой
        короткой настоящей секции. 0 выключает правило. В общем конфиге —
        `TranscriptAssemblyConfig.short_offslide_section_s`.
    """

    short_offslide_section_s: float = 15.0


def consolidate_sections(
    sections: Sequence[TranscriptSection], policy: ConsolidationPolicy | None = None
) -> tuple[TranscriptSection, ...]:
    """Правило 1 со структурной проверкой полноты слов."""
    policy = policy or ConsolidationPolicy()
    words = tuple(word for section in sections for word in section.words)
    result = absorb_short_offslide_sections(sections, policy.short_offslide_section_s)
    _assert_completeness(words, result)
    return result


def absorb_short_offslide_sections(
    sections: Sequence[TranscriptSection], max_s: float
) -> tuple[TranscriptSection, ...]:
    """Правило 1: короткую речь вне слайдов — в предыдущий слайд, иначе в следующий."""
    result: list[TranscriptSection] = []
    # Короткие секции, у которых нет предыдущей секции слайда: ждут следующую.
    pending: list[TranscriptSection] = []
    for section in sections:
        if section.slide is None and _is_short(section, max_s):
            if result and result[-1].slide is not None:
                result[-1] = _join(result[-1], section, keep=result[-1])
            else:
                pending.append(section)
            continue
        if pending and section.slide is not None:
            for short in reversed(pending):
                section = _join(short, section, keep=section)
            pending = []
        elif pending:
            result.extend(pending)
            pending = []
        result.append(section)
    result.extend(pending)
    return tuple(result)


def group_same_content_slides(
    sections: Sequence[TranscriptSection],
) -> tuple[tuple[TranscriptSection, ...], ...]:
    """Правило 2: подряд идущие секции слайдов с одинаковым содержимым — в группу.

    Возвращает группы в исходном порядке; секция вне слайдов или слайд с
    другим/пустым содержимым начинает новую группу. Конкатенация групп равна
    входу — секции не теряются и не переставляются.
    """
    groups: list[list[TranscriptSection]] = []
    for section in sections:
        if groups and section.slide is not None and groups[-1][-1].slide is not None:
            previous = content_signature(groups[-1][-1].slide_ocr)
            if previous is not None and previous == content_signature(section.slide_ocr):
                groups[-1].append(section)
                continue
        groups.append([section])
    return tuple(tuple(group) for group in groups)


def ocr_body(markdown: str) -> str:
    """Содержимое `SlideOcr.markdown` без ссылок на изображение и без
    пометки ненадёжности, которую slide-ocr мог встроить в начало цитатой.

    Ведущие строки `> ...` отбрасываются: текст слайда, начинающийся с `>`,
    slide-ocr экранирует, поэтому неэкранированная цитата в начале — это
    служебная пометка, а не содержимое. Пометку ненадёжности во врезке ставит
    рендер, один раз.
    """
    lines = _IMAGE_LINK.sub("", markdown).strip().splitlines()
    start = 0
    while start < len(lines) and (
        not lines[start].strip() or lines[start].lstrip().startswith(">")
    ):
        start += 1
    return "\n".join(line.rstrip() for line in lines[start:]).strip()


def content_signature(ocr: SlideOcr | None) -> str | None:
    """Сигнатура содержимого слайда для сравнения соседей; None — сравнивать нечего."""
    if ocr is None:
        return None
    text = ocr_body(ocr.markdown).replace(OCR_LOW_CONFIDENCE_MARK, " ")
    text = " ".join(text.split())
    return text or None


def _is_short(section: TranscriptSection, max_s: float) -> bool:
    return max_s > 0.0 and section.end_s - section.start_s < max_s


def _join(
    first: TranscriptSection, second: TranscriptSection, *, keep: TranscriptSection
) -> TranscriptSection:
    """Две соседние секции в одну: слова подряд, интервал — объединение,
    слайд и OCR — от `keep`."""
    return TranscriptSection(
        start_s=min(first.start_s, second.start_s),
        end_s=max(first.end_s, second.end_s),
        slide=keep.slide,
        slide_ocr=keep.slide_ocr,
        words=first.words + second.words,
        paragraphs=(),
    )
