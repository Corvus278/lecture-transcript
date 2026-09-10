"""Сборка глоссария терминов лекции по всем слайдам (задача 4.7).

Решения, принятые здесь:

1. **Источник** — только текстовые фрагменты (`kind="text"`). Формулы в
   глоссарий не идут: ASR они не помогают.

2. **Отсечение оформления** — без стоп-листов под конкретную платформу
   (design запрещает хардкод под шаблон). Надпись считается элементом
   оформления, если выполнены три условия:

   * она встречается почти на всех слайдах записи
     (>= `decoration_slide_ratio`, по умолчанию 0.8, при минимум
     `min_slides_for_decoration` слайдах);
   * она держится на одном и том же месте: не меньше
     `position_stable_ratio` её боксов лежат в пределах
     `position_tolerance_px` от медианы центров. Именно доля, а не «все
     боксы»: медиана берётся ради устойчивости к выбросам, и один
     съехавший бокс (титульный слайд, сбитый макет) не должен отменять
     отсечение целиком. Если боксов нет, критерий позиции пропускается;
   * она не состоит целиком из слов, которые встречаются на слайдах и вне
     её. Это отличает служебную плашку платформы от заголовка темы,
     повторённого в шапке каждого слайда: слова темы — это словарь самой
     лекции, и терять их (самый ценный для ASR термин) нельзя, а слова
     логотипа больше нигде не встречаются.

3. **Дедупликация** — по ключу «casefold + ё→е + лёгкое отсечение русского
   окончания». Полноценная лемматизация потребовала бы pymorphy/pymystem,
   которых нет в зависимостях; суффиксное отсечение склеивает «уравнение /
   уравнения / уравнений» и достаточно для подсказок ASR. Канонической
   формой берётся самая частая словоформа (при равенстве — самая короткая,
   затем по алфавиту), чтобы результат был детерминированным.

4. **Что считается термином** — слово длиной >= `min_term_length`, не из
   служебного списка, без цифр; плюс двусловные сочетания с заглавной
   второй частью («теорема Виета») как имена собственные. Биграмма не
   берётся, если обе её части набраны капсом (заголовок, а не имя
   собственное) или если вторая часть — служебное слово.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence

from ..contracts import Glossary, SlideOcr

#: Служебные слова, которые не являются терминами лекции.
STOP_WORDS = frozenset(
    """
    если тогда значит также поэтому потому который которая которое которые
    когда чтобы этот эта это эти того этом этого таким образом более менее
    между через после перед около каждый каждая любой любое такой такая такие
    была были было быть есть будет будут может можно нужно надо очень уже
    ещё еще только всех всего весь вся все или либо даже итак пусть здесь
    там где как что чем чтоб при над под без для про свои свой своя
    the and for with this that from you are not but all any
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-']*")
_ENDINGS = tuple(
    sorted(
        {
            "иями", "ями", "ами", "ого", "его", "ему", "ыми", "ими", "ией",
            "ов", "ев", "ая", "ое", "ые", "ий", "ый", "ой", "ей", "ам", "ям",
            "ах", "ях", "ом", "ем", "ью", "ия", "ии", "ие", "ию",
            "а", "о", "е", "и", "ы", "у", "ю", "я", "ь", "й",
        },
        key=len,
        reverse=True,
    )
)
#: Хвосты, дострагиваемые вторым проходом: склеивают «уравнение/уравнений».
_TAIL_CHARS = ("и", "ь", "й")
#: Гласные (после casefold и ё→е) — нужны для беглой гласной, см. normalize_key.
_VOWELS = frozenset("аеиоуыэюя")


@dataclass(frozen=True)
class GlossaryConfig:
    """Параметры сборки глоссария. TODO(7.6): подстроить на эталонной записи."""

    min_term_length: int = 4
    #: Доля слайдов, с которой повторяющаяся надпись считается оформлением.
    decoration_slide_ratio: float = 0.8
    #: Меньше этого числа слайдов — эвристика оформления не применяется.
    min_slides_for_decoration: int = 3
    #: Допустимый разброс центров бокса надписи, пиксели.
    position_tolerance_px: int = 32
    #: Доля боксов надписи, которая обязана уложиться в этот разброс.
    position_stable_ratio: float = 0.8
    #: Брать ли фрагменты, помеченные как неуверенные.
    include_low_confidence: bool = False
    max_terms: int = 500
    #: Ручной аварийный выход: надписи, которые не должны попасть в глоссарий.
    #:
    #: Отсечение оформления делается геометрией и словарём лекции (решение 2),
    #: и стоп-лист его НЕ заменяет: по умолчанию он пуст и в коде никогда не
    #: заполняется под конкретную платформу или шаблон — это ровно тот
    #: хардкод, который запрещает design. Список существует только затем,
    #: чтобы человек мог вмешаться, когда вероятностная эвристика ошиблась на
    #: чужом оформлении (design, раздел Risks), и заполняется из конфигурации
    #: прогона. Сравнение идёт по нормализованному ключу, поэтому достаточно
    #: одной словоформы; фраза из нескольких слов вырезает и себя, и свои
    #: слова по отдельности.
    stopwords: tuple[str, ...] = ()


DEFAULT_GLOSSARY = GlossaryConfig()


def normalize_key(word: str) -> str:
    """Ключ дедупликации: регистр, ё→е и лёгкое отсечение окончания."""
    key = unicodedata.normalize("NFKC", word).casefold().replace("ё", "е")
    key = key.strip("-'")
    for ending in _ENDINGS:
        if key.endswith(ending) and len(key) - len(ending) >= 4:
            key = key[: -len(ending)]
            break
    if key.endswith(_TAIL_CHARS) and len(key) - 1 >= 4:
        key = key[:-1]
    # Беглая гласная перед последней согласной: «корень» -> «корен» -> «корн»,
    # иначе «корень» и «корни» -> «корн» дают в глоссарии две записи одного
    # слова. Правило применяется ко всем словоформам одинаково, поэтому
    # группировка остаётся согласованной.
    if (
        len(key) - 1 >= 4
        and key[-1] not in _VOWELS
        and key[-2] in "ео"
        and key[-3] not in _VOWELS
    ):
        key = key[:-2] + key[-1]
    return key


def normalize_line(text: str) -> str:
    """Ключ строки для поиска повторяющихся надписей оформления."""
    compact = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]+", " ", compact).split())


def _center(bbox) -> tuple[float, float]:
    return (bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0)


def find_decoration_lines(
    slides: Sequence[SlideOcr],
    config: GlossaryConfig = DEFAULT_GLOSSARY,
) -> set[str]:
    """Нормализованные строки, признанные элементами оформления.

    Критерий: надпись есть почти на всех слайдах, стоит на одном месте и не
    состоит целиком из слов лекции (см. решение 2 в докстринге модуля).
    """
    total = len(slides)
    if total < config.min_slides_for_decoration:
        return set()

    slides_with_line: dict[str, set[int]] = defaultdict(set)
    centers: dict[str, list[tuple[float, float]]] = defaultdict(list)
    surface: dict[str, str] = {}
    for position, slide in enumerate(slides):
        for fragment in slide.fragments:
            key = normalize_line(fragment.text)
            if not key:
                continue
            slides_with_line[key].add(position)
            surface.setdefault(key, fragment.text)
            if fragment.bbox is not None:
                centers[key].append(_center(fragment.bbox))

    candidates: set[str] = set()
    for key, positions in slides_with_line.items():
        if len(positions) / total < config.decoration_slide_ratio:
            continue
        if not _position_is_stable(centers.get(key, []), config):
            continue
        candidates.add(key)
    if not candidates:
        return set()

    vocabulary = _lecture_vocabulary(slides, candidates, config)
    return {
        key
        for key in candidates
        if not _made_of_lecture_words(surface[key], vocabulary, config)
    }


def _position_is_stable(
    points: Sequence[tuple[float, float]],
    config: GlossaryConfig,
) -> bool:
    """Держится ли надпись на одном месте.

    Считается доля боксов, попавших в допуск вокруг медианы центров: медиана
    и берётся ради устойчивости к выбросам, поэтому сверять её жёстким «все
    до одного» бессмысленно — один съехавший бокс отключил бы отсечение.
    Без боксов критерий позиции не применяется.
    """
    if not points:
        return True
    mx = median(x for x, _ in points)
    my = median(y for _, y in points)
    inside = sum(
        1
        for x, y in points
        if abs(x - mx) <= config.position_tolerance_px
        and abs(y - my) <= config.position_tolerance_px
    )
    return inside / len(points) >= config.position_stable_ratio


def _lecture_vocabulary(
    slides: Sequence[SlideOcr],
    candidates: set[str],
    config: GlossaryConfig,
) -> set[str]:
    """Ключи терминов, встречающихся на слайдах вне строк-кандидатов."""
    vocabulary: set[str] = set()
    for slide in slides:
        for fragment in slide.fragments:
            if fragment.kind != "text":
                continue
            if normalize_line(fragment.text) in candidates:
                continue
            for term in extract_terms(fragment.text, config):
                if " " in term:
                    continue
                vocabulary.add(normalize_key(term))
    return vocabulary


def _made_of_lecture_words(
    text: str,
    vocabulary: set[str],
    config: GlossaryConfig,
) -> bool:
    """Состоит ли строка целиком из слов, звучащих на слайдах и вне её.

    Заголовок темы («Квадратные уравнения» в шапке каждого слайда) — это
    словарь самой лекции, и он обязан попасть в глоссарий. Плашка платформы
    приносит собственные слова («Онлайн-школа»), которых в содержании
    слайдов нет, поэтому целиком «своей» она не окажется.
    """
    words = [term for term in extract_terms(text, config) if " " not in term]
    if not words:
        return False
    return all(normalize_key(word) in vocabulary for word in words)


def extract_terms(text: str, config: GlossaryConfig = DEFAULT_GLOSSARY) -> list[str]:
    """Кандидаты в термины из одной строки слайда."""
    tokens = [match.group(0) for match in _TOKEN_RE.finditer(text)]
    terms: list[str] = []
    for index, token in enumerate(tokens):
        if len(token) < config.min_term_length:
            continue
        if token.casefold().replace("ё", "е") in STOP_WORDS:
            continue
        terms.append(token)
        # Имя собственное после нарицательного: «теорема Виета».
        if index + 1 < len(tokens):
            nxt = tokens[index + 1]
            if (
                nxt[:1].isupper()
                and len(nxt) >= config.min_term_length
                and not (token.isupper() and nxt.isupper())  # заголовок капсом
                and nxt.casefold().replace("ё", "е") not in STOP_WORDS
            ):
                terms.append(f"{token} {nxt}")
    return terms


def stopword_keys(config: GlossaryConfig = DEFAULT_GLOSSARY) -> frozenset[str]:
    """Нормализованные ключи ручного стоп-листа (см. `GlossaryConfig.stopwords`)."""
    keys: set[str] = set()
    for entry in config.stopwords:
        parts = [normalize_key(part) for part in entry.split() if part]
        parts = [part for part in parts if part]
        if not parts:
            continue
        keys.add(" ".join(parts))
        keys.update(parts)
    return frozenset(keys)


def build_glossary(
    slides: Iterable[SlideOcr],
    config: GlossaryConfig = DEFAULT_GLOSSARY,
) -> Glossary:
    """Собрать глоссарий терминов по всем слайдам записи."""
    slides = list(slides)
    if not slides:
        return Glossary()

    decoration = find_decoration_lines(slides, config)
    stopped = stopword_keys(config)
    surface_counts: dict[str, Counter[str]] = defaultdict(Counter)
    order: dict[str, int] = {}

    for slide in slides:
        for fragment in slide.fragments:
            if fragment.kind != "text":
                continue
            if fragment.low_confidence and not config.include_low_confidence:
                continue
            if normalize_line(fragment.text) in decoration:
                continue
            for term in extract_terms(fragment.text, config):
                key = " ".join(normalize_key(part) for part in term.split())
                if not key or key in stopped:
                    continue
                surface_counts[key][term] += 1
                order.setdefault(key, len(order))

    terms: list[tuple[int, int, str]] = []
    for key, counter in surface_counts.items():
        total = sum(counter.values())
        canonical = min(counter.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0]
        terms.append((-total, order[key], canonical))
    terms.sort()
    return Glossary(tuple(term for _, _, term in terms[: config.max_terms]))
