"""Разбиение речи секции на абзацы по смысловым границам (задача 6.4).

Требование спеки: «не одним блоком и не по каждой паузе». Поэтому пауза сама
по себе абзац не начинает — она лишь **разрешает** разрыв, а решение принимает
накопленный объём абзаца.

Алгоритм
--------
Идём по словам и копим абзац. Разрыв ставится перед словом `i`, если:

* абзац уже «созрел» — его длительность >= `min_paragraph_s` или в нём
  >= `min_paragraph_words` слов — и перед словом `i` есть смысловая граница:
  конец предложения у предыдущего слова (точка/вопрос/восклицание после
  стадии пунктуации, design D6) либо пауза >= `pause_s`;
* либо абзац перерос `max_paragraph_s` / `max_paragraph_words` — тогда рвём
  на первой же паузе >= `soft_pause_s`, а если и её нет, то принудительно.

Дефолты (подбираемые, задача 7.6)
---------------------------------
* `pause_s = 2.5` — из конфига (`paragraph_pause_s`): пауза лектора на смене
  мысли, дыхательные паузы внутри фразы короче.
* `min_paragraph_s = 45.0` — при темпе 120–150 слов/мин это ~100–110 слов,
  то есть плотный, но обозримый абзац конспекта. На десятиминутном монологе
  даёт порядка десятка абзацев, а не сотню по числу пауз.
* `max_paragraph_s = 180.0` — три минуты сплошного текста уже нечитаемы,
  дальше рвём даже без хорошей границы.
* `min_paragraph_words = 90`, `max_paragraph_words = 320` — страховка на
  случай нетипичного темпа речи: 90 слов — это те же ~45 с при 120 слов/мин,
  но порог срабатывает и когда речь идёт без пауз быстрее обычного.
* `soft_pause_s = 0.8` — «хоть какая-то» пауза для принудительного разрыва.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..contracts import TranscriptSection, Word

__all__ = [
    "ParagraphPolicy",
    "split_words",
    "paragraph_text",
    "capitalize_sentences",
    "split_paragraphs",
    "fill_paragraphs",
]

_SENTENCE_END = (".", "?", "!", "…")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:…»)])")
_SPACE_AFTER_OPEN = re.compile(r"([«(])\s+")
_OPENING = "«\"'(["
_CLOSING = "»\"')]"
# Сокращения с точкой, после которых предложение не кончается («т. е.»).
_ABBREVIATIONS = frozenset(
    {"т", "е", "д", "п", "т.е", "т.д", "т.п", "др", "пр", "ср", "см", "рис",
     "стр", "напр", "г", "гг", "в", "вв", "им"}
)


@dataclass(frozen=True)
class ParagraphPolicy:
    """Параметры разбиения на абзацы. Обоснование дефолтов — в docstring модуля."""

    pause_s: float = 2.5
    soft_pause_s: float = 0.8
    min_paragraph_s: float = 45.0
    max_paragraph_s: float = 180.0
    min_paragraph_words: int = 90
    max_paragraph_words: int = 320


def split_words(
    words: Sequence[Word], policy: ParagraphPolicy | None = None
) -> tuple[tuple[Word, ...], ...]:
    """Разбить слова на группы-абзацы. Чистая функция: одинаковый вход —
    одинаковый выход, поэтому рендер может пересчитать группы, чтобы взять
    таймкод начала абзаца."""
    policy = policy or ParagraphPolicy()
    if not words:
        return ()

    groups: list[tuple[Word, ...]] = []
    start = 0
    for i in range(1, len(words)):
        current = words[start:i]
        duration = current[-1].end_s - current[0].start_s
        gap = words[i].start_s - words[i - 1].end_s
        sentence_end = words[i - 1].text.rstrip().endswith(_SENTENCE_END)

        mature = (
            duration >= policy.min_paragraph_s
            or len(current) >= policy.min_paragraph_words
        )
        overgrown = (
            duration >= policy.max_paragraph_s
            or len(current) >= policy.max_paragraph_words
        )

        should_break = False
        if mature and (sentence_end or gap >= policy.pause_s):
            should_break = True
        elif overgrown and gap >= policy.soft_pause_s:
            should_break = True
        elif duration >= policy.max_paragraph_s * 1.5:
            should_break = True

        if should_break:
            groups.append(tuple(current))
            start = i

    groups.append(tuple(words[start:]))
    return tuple(groups)


def paragraph_text(group: Sequence[Word]) -> str:
    """Склеить слова абзаца в текст. Объекты `Word` не меняются; в строке
    поднимается регистр начала предложений (`capitalize_sentences`)."""
    text = " ".join(word.text.strip() for word in group if word.text.strip())
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return capitalize_sentences(text.strip())


def capitalize_sentences(text: str) -> str:
    """Заглавная буква в начале абзаца и после `.?!…` — только в тексте.

    Чистка удаляет филлер вместе с заглавной («Ну вот получаем» -> «получаем»),
    а пересоздавать `Word` нельзя (решение 2.4.2). Поэтому регистр
    поднимается здесь, в строке абзаца: объекты `Word` и таймкоды не
    меняются. Трогается только строчная кириллица: слово с латиницы или
    цифры («x равно», «2 штуки») остаётся как есть. После сокращений из
    `_ABBREVIATIONS` («т. е.») предложение не считается законченным.
    """
    parts = re.split(r"(\s+)", text)
    sentence_start = True
    for pos, token in enumerate(parts):
        if not token or token.isspace():
            continue
        if sentence_start:
            lead = len(token) - len(token.lstrip(_OPENING))
            if lead < len(token) and "а" <= token[lead] <= "я" or (
                lead < len(token) and token[lead] == "ё"
            ):
                token = token[:lead] + token[lead].upper() + token[lead + 1 :]
                parts[pos] = token
        tail = token.rstrip(_CLOSING)
        core = tail.rstrip(".?!…").lower()
        sentence_start = tail.endswith(_SENTENCE_END) and not (
            tail.endswith(".") and core in _ABBREVIATIONS
        )
    return "".join(parts)


def split_paragraphs(
    words: Sequence[Word], policy: ParagraphPolicy | None = None
) -> tuple[str, ...]:
    """Готовые тексты абзацев секции."""
    return tuple(
        paragraph_text(group) for group in split_words(words, policy) if group
    )


def fill_paragraphs(
    sections: Sequence[TranscriptSection], policy: ParagraphPolicy | None = None
) -> tuple[TranscriptSection, ...]:
    """Проставить `paragraphs` во всех секциях."""
    result: list[TranscriptSection] = []
    for section in sections:
        result.append(
            TranscriptSection(
                start_s=section.start_s,
                end_s=section.end_s,
                slide=section.slide,
                slide_ocr=section.slide_ocr,
                words=section.words,
                paragraphs=split_paragraphs(section.words, policy),
            )
        )
    return tuple(result)
