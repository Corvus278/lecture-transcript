"""Роутинг строк слайда в формульный распознаватель (задача 4.3, design D5).

D5: «PaddleOCR отдаёт боксы строк с confidence; строки с математическими
символами или низкой уверенностью уходят в pix2tex, остальное берётся как
текст».

Уточнение к этой формулировке (см. `looks_like_handwriting`): низкая
уверенность — признак ДВУХ разных ситуаций, и D5 различает их явно.

* печатная формула, которую текстовый OCR прочитал плохо, — категория [2]
  из D5, её и задумано отправлять в pix2tex;
* рукописная пометка маркером — категория [3], которая «не берётся локально
  ни OCR, ни pix2tex, ни VLM» и решается «не моделью, а ссылкой» на PNG.

Различаем их по наличию математических признаков в самой строке: pix2tex
всегда что-то возвращает, поэтому отправить туда рукописную строку — значит
подменить её галлюцинацией, чего D5 прямо не хочет («изображение слайда под
рукой полезнее, чем галлюцинация модели на месте рукописной записи»).

Пороги вынесены в `RoutingConfig` и подбираются эмпирически на эталонной
записи (задача 7.6) — здесь стоят разумные стартовые значения.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..contracts import FragmentKind

#: Символы, однозначно указывающие на формулу.
STRONG_MATH_CHARS = frozenset(
    "√∛∜∫∮∑∏±∓≤≥≠≡≈∞∈∉∋⊂⊃⊆⊇∪∩∅∀∃∄∂∇⊥∠∡°′″×÷⋅→⇒⇔↔↦∧∨¬⌊⌋⌈⌉∆"
)
#: Надстрочные/подстрочные индексы — тоже сильный признак формулы.
SCRIPT_CHARS = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ⁺⁻⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₍₎")
#: Греческие буквы, используемые как математические обозначения.
GREEK_CHARS = frozenset("αβγδεζηθικλμνξοπρςστυφχψωΓΔΘΛΞΠΣΦΨΩ")
#: Слабые признаки: сами по себе формулу не доказывают.
WEAK_MATH_CHARS = frozenset("=+*/^_<>|\\−–—")

#: Латинские «слова», которые на самом деле математические функции.
MATH_WORDS = frozenset(
    {
        "sin", "cos", "tg", "ctg", "tan", "cot", "sec", "csc", "log", "lg",
        "ln", "lim", "max", "min", "sup", "inf", "exp", "sqrt", "det", "mod",
        "arcsin", "arccos", "arctg", "arcctg", "sh", "ch", "th",
    }
)

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")


@dataclass(frozen=True)
class RoutingConfig:
    """Пороги роутинга. TODO(7.6): подобрать эмпирически на эталонной записи."""

    #: Ниже этой уверенности строка считается плохо прочитанной текстовым OCR.
    low_confidence_threshold: float = 0.60
    #: Отправлять ли неуверенные строки в формульный распознаватель (D5).
    route_low_confidence: bool = True
    #: Доля «словесных» символов, выше которой строка со слабыми признаками
    #: (`=`, `+`, `^`, ...) считается обычным текстом.
    weak_max_word_ratio: float = 0.5
    #: То же для строк с сильными математическими символами: строка,
    #: почти целиком состоящая из слов, формулой не считается.
    strong_max_word_ratio: float = 0.7
    #: Ниже этой уверенности строка, в которой нет вообще никаких
    #: математических признаков, считается нераспознаваемой рукописной
    #: пометкой (D5, категория [3]) и в текст слайда не попадает.
    handwriting_threshold: float = 0.35


DEFAULT_ROUTING = RoutingConfig()


def has_math_symbols(text: str) -> bool:
    """Есть ли в строке сильные математические символы."""
    chars = set(text)
    return bool(
        chars & STRONG_MATH_CHARS or chars & SCRIPT_CHARS or chars & GREEK_CHARS
    )


def has_weak_math_symbols(text: str) -> bool:
    """Есть ли в строке слабые математические признаки (`=`, `+`, `^`, ...)."""
    return bool(set(text) & WEAK_MATH_CHARS)


def has_any_math_signal(text: str) -> bool:
    """Есть ли в строке хоть какой-то математический признак.

    Сильные символы (`√`, `∫`, `α`, индексы) либо слабые операторы (`=`, `+`,
    `^`) при наличии букв/цифр. Ровно этот предикат отделяет плохо прочитанную
    печатную формулу от рукописной пометки.
    """
    if has_math_symbols(text):
        return True
    return has_weak_math_symbols(text) and any(ch.isalnum() for ch in text)


def word_ratio(text: str) -> float:
    """Доля символов, входящих в естественно-языковые слова (>=3 букв).

    Математические функции (`sin`, `log`, `lim`) словами не считаются.
    """
    compact = "".join(text.split())
    if not compact:
        return 0.0
    word_chars = sum(
        len(word)
        for word in _WORD_RE.findall(text)
        if word.lower() not in MATH_WORDS
    )
    return word_chars / len(compact)


def looks_like_formula(text: str, config: RoutingConfig = DEFAULT_ROUTING) -> bool:
    """Похожа ли строка на формулу по одному только её тексту."""
    if not text.strip():
        return False
    ratio = word_ratio(text)
    if has_math_symbols(text):
        return ratio <= config.strong_max_word_ratio
    if has_weak_math_symbols(text) and any(ch.isalnum() for ch in text):
        return ratio <= config.weak_max_word_ratio
    return False


def route_line(
    text: str,
    confidence: float = 1.0,
    config: RoutingConfig = DEFAULT_ROUTING,
) -> FragmentKind:
    """Куда отправить строку: в формульный распознаватель или взять текстом.

    Возвращает "formula" для строк с математическими символами и (по D5) для
    строк с низкой уверенностью — но во втором случае только если в строке
    есть хоть какие-то математические признаки. Строка без них при низкой
    уверенности — не плохо прочитанная формула, а рукописное (D5, [3]), и
    отправлять её в pix2tex значит получить галлюцинацию вместо текста.
    """
    if not text.strip():
        return "text"
    if config.route_low_confidence and confidence < config.low_confidence_threshold:
        # Низкая уверенность снимает порог доли слов, но не заменяет собой
        # само наличие математики в строке.
        return "formula" if has_any_math_signal(text) else "text"
    return "formula" if looks_like_formula(text, config) else "text"


def looks_like_handwriting(
    text: str,
    confidence: float,
    config: RoutingConfig = DEFAULT_ROUTING,
) -> bool:
    """Строка похожа на рукописную пометку и распознаванию не подлежит.

    Признак — очень низкая уверенность текстового OCR при полном отсутствии
    математических символов: печатный текст на белом фоне PaddleOCR читает с
    ~0.99, печатная формула хотя бы даёт операторы. D5 требует не тащить такое
    в текст: доступ к рукописному даёт ссылка на PNG слайда.
    """
    if not text.strip():
        return False
    return confidence < config.handwriting_threshold and not has_any_math_signal(text)


def explain_route(
    text: str,
    confidence: float = 1.0,
    config: RoutingConfig = DEFAULT_ROUTING,
) -> str:
    """Человекочитаемая причина решения роутера — для диагностики и логов."""
    kind = route_line(text, confidence, config)
    if kind == "text":
        if looks_like_handwriting(text, confidence, config):
            return "текст: рукописная пометка — не распознаётся (D5), см. PNG слайда"
        if config.route_low_confidence and confidence < config.low_confidence_threshold:
            return (
                f"текст: низкая уверенность {confidence:.2f}, но математических "
                "символов нет — в формульный распознаватель не идёт"
            )
        return "текст: нет математических символов либо строка состоит из слов"
    if config.route_low_confidence and confidence < config.low_confidence_threshold:
        return f"формула: низкая уверенность {confidence:.2f} < {config.low_confidence_threshold:.2f}"
    if has_math_symbols(text):
        return "формула: сильные математические символы"
    return "формула: операторы при малой доле слов"
