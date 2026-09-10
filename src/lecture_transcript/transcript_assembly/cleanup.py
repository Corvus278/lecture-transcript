"""Консервативная чистка филлеров и оборванных самоповторов (задача 6.3).

Стадия работает поверх уже собранных секций (см. порядок стадий в
`merge`: merge -> cleanup -> paragraphs -> render), то есть **после** проверки
инварианта полноты 6.1.

Два жёстких ограничения:

1. Чистка только **фильтрует** список слов и никогда не создаёт новые `Word`
   и не правит их поля. Поэтому привязка таймкодов не ломается: у оставшихся
   слов `start_s`/`end_s`/`text` в точности те же, что пришли из ASR.
2. Правила консервативные: удаляется только заведомый мусор, всё сомнительное
   остаётся (прямое требование design). Цена ложного пропуска — лишнее «э»
   в тексте, цена ложного удаления — потерянный смысл.

Что удаляется
-------------
* одиночные филлеры-заполнители из `FILLER_WORDS` («э», «ээ», «мм», …);
* фразовые филлеры из `FILLER_PHRASES` («ну вот») — у них нет содержательных
  омонимов, поэтому вхождение удаляется любое;
* обороты из `AMBIGUOUS_FILLER_PHRASES` («как бы», «это самое», …) —
  **только когда это вставка**: оборот не открывает предложение и обособлен
  с обеих сторон запятой/тире И паузой (см. `_is_insertion`). В тексте без
  пунктуации такие обороты не удаляются никогда — пауза омонимию не разрешает;
* непосредственный повтор n-граммы (n = 3, 2, 1): серия схлопывается до
  одной копии, удаляются ранние — последнюю лектор договорил
  («то есть то есть» -> «то есть», «мы мы мы рассмотрим» -> «мы рассмотрим»);
  если же заглавную букву несёт ранняя копия и копии совпадают посимвольно
  («Он он знает», «Мы мы мы»), удаляются поздние — регистр начала
  предложения дороже позиции;
* оборванное начало слова с дефисом, продолженное следующим словом
  («давайте рас- рассмотрим» -> «давайте рассмотрим»).

Что НЕ удаляется
----------------
* слово, несущее конец предложения («вот.») — иначе потеряется граница
  предложения, по которой работает разбиение на абзацы;
* удвоения из `ALLOWED_DOUBLES` («очень очень», «чуть чуть») — это норма языка;
* повтор n-граммы, в которой есть число, знак формулы или однобуквенное
  обозначение (`_is_math_token`): «два плюс два плюс два», «икс плюс икс
  плюс икс», «ноль ноль один», «Пусть n n стремится», «точка А А равна» —
  это арифметика и диктовка, а не оговорка. Профиль проекта — лекция по
  математике, такой повтор здесь норма. Строчные служебные «в, и, с, к, у, я»
  обозначениями не считаются — «в в этом случае» чистится;
* повтор, где заглавную несёт ранняя копия, а пунктуация копий различается
  («Это, это важно», «Мы мы, конечно»): любое удаление потеряло бы заглавную
  или запятую, а пересоздавать `Word` нельзя;
* однобуквенные токены в роли филлера: «м» — это метр и имя точки/переменной,
  поэтому в `FILLER_WORDS` его нет (там остались только звуки-заполнители
  длиной от одного символа, у которых нет омонима-обозначения: «э», «ээ», …);
* усечённый префикс без дефиса («раз разность») — слишком легко спутать с
  осмысленным словом;
* вся секция целиком: если правила съели все слова, возвращается исходный
  список.

Общий принцип (прямое требование design): при любом сомнении — НЕ удалять.
Цена ложного пропуска — лишнее «э» в тексте, цена ложного удаления —
потерянный смысл.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..contracts import TranscriptSection, Word

__all__ = [
    "FILLER_WORDS",
    "FILLER_PHRASES",
    "AMBIGUOUS_FILLER_PHRASES",
    "ALLOWED_DOUBLES",
    "NUMERAL_WORDS",
    "MATH_TOKENS",
    "CleanupPolicy",
    "cleanup_words",
    "cleanup_sections",
]

# Одиночные филлеры. Только звуки-заполнители и слова, которые в русской
# лекторской речи не несут смысла ни в одном контексте.
# «м» здесь намеренно нет: в лекции это «метр» и распространённое имя
# точки/переменной («возьмём м равное трём»).
FILLER_WORDS: frozenset[str] = frozenset(
    {
        "э",
        "ээ",
        "эээ",
        "ээээ",
        "эм",
        "эмм",
        "мм",
        "ммм",
        "мда",
        "ааа",
        "аа",
        "ээм",
        "нуу",
        "нуэ",
    }
)

# Фразовые филлеры без содержательных омонимов: удаляются при точном
# совпадении подряд, контекст не проверяется.
FILLER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("ну", "вот"),
    ("вот", "ну"),
)

# Обороты, омонимичные содержательным конструкциям: «Как бы вы доказали это»,
# «Так сказать нельзя», «Скажем так же и для суммы», «Это самое важное
# свойство». Удаляются только как обособленная вставка (`_is_insertion`).
AMBIGUOUS_FILLER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("как", "бы"),
    ("так", "сказать"),
    ("скажем", "так"),
    ("это", "самое"),
)

# Удвоения, нормальные для языка: не считаются самоповтором.
ALLOWED_DOUBLES: frozenset[str] = frozenset(
    {
        "очень",
        "чуть",
        "еле",
        "давно",
        "далеко",
        "долго",
        "много",
        "совсем",
        "тихо",
        "быстро",
        "едва",
    }
)

# Числительные и счётные слова (в том числе в косвенных падежах). Повтор
# такого токена — почти всегда содержание: арифметика или диктовка числа.
NUMERAL_WORDS: frozenset[str] = frozenset(
    {
        "ноль", "нуля", "нулю", "нулём", "нуль",
        "один", "одна", "одно", "одного", "одной", "одному", "одним",
        "два", "две", "двух", "двум", "двумя", "дважды",
        "три", "трёх", "трех", "трём", "трем", "тремя", "трижды",
        "четыре", "четырёх", "четырех", "четырём", "четырем",
        "пять", "пяти", "пятью",
        "шесть", "шести", "семь", "семи", "восемь", "восьми",
        "девять", "девяти", "десять", "десяти",
        "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
        "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать",
        "девятнадцать", "двадцать", "тридцать", "сорок", "пятьдесят",
        "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
        "сто", "ста", "двести", "триста", "четыреста", "пятьсот",
        "шестьсот", "семьсот", "восемьсот", "девятьсот",
        "тысяча", "тысячи", "тысяч", "миллион", "миллиона", "миллионов",
        "раз", "раза",
    }
)

# Знаки и слова математической записи, включая названия букв. n-грамма с
# таким токеном не может считаться оговоркой: «икс плюс икс плюс икс» — формула.
MATH_TOKENS: frozenset[str] = frozenset(
    {
        "плюс", "минус", "равно", "равен", "равна", "равны", "равняется",
        "умножить", "умноженное", "разделить", "делить", "деленное",
        "дробь", "модуль", "степень", "степени", "квадрат", "квадрате",
        "куб", "кубе", "корень", "корня", "факториал", "процент", "процентов",
        "сумма", "интеграл",
        "икс", "игрек", "зет", "эн", "ка", "эль", "эс", "тэ",
        "бэ", "вэ", "гэ", "дэ", "жэ", "пэ", "цэ", "эр", "эф", "аш",
        "альфа", "бета", "гамма", "дельта", "эпсилон", "дзета", "эта",
        "тета", "тэта", "йота", "каппа", "лямбда", "мю", "ню", "кси",
        "пи", "ро", "сигма", "тау", "фи", "хи", "пси", "омега",
    }
)

# Однобуквенные русские служебные слова (предлоги, союзы, местоимение), которые
# НЕ считаются обозначениями при дедупе: как имена точек эти буквы вслух
# называют иначе («вэ», «эс»/«цэ», «ка»). «а» и «о» сюда намеренно не входят —
# они произносятся так же, как точки A и O.
_SERVICE_LETTERS: frozenset[str] = frozenset({"в", "и", "с", "к", "у", "я"})

_SENTENCE_END = (".", "?", "!", "…")
# Знаки, которыми обособляется вставка внутри предложения.
_CLAUSE_END = (",", ";", ":", "—", "–", "-")
_STRIP_CHARS = " \t«»\"'`(),.!?;:—–…"


@dataclass(frozen=True)
class CleanupPolicy:
    """Переключатели чистки.

    enabled: соответствует `TranscriptAssemblyConfig.filler_cleanup`.
    max_repeat_ngram: максимальная длина повторяемой n-граммы (3 — «то есть,
        то есть» и подобные обороты; длиннее — уже риск съесть намеренный
        повтор определения).
    filler_gap_s: пауза вокруг омонимичного оборота, с. Дополнительное
        условие к запятым/тире с обеих сторон, самостоятельным признаком
        вставки не является. 0.3 с — заметная заминка: внутри фразы слова
        идут плотнее, поэтому при меньшем промежутке оборот остаётся даже
        между запятыми.
    """

    enabled: bool = True
    max_repeat_ngram: int = 3
    filler_gap_s: float = 0.3


def cleanup_sections(
    sections: Sequence[TranscriptSection], policy: CleanupPolicy | None = None
) -> tuple[TranscriptSection, ...]:
    """Применить чистку к словам каждой секции, сохранив остальные поля."""
    policy = policy or CleanupPolicy()
    result: list[TranscriptSection] = []
    for section in sections:
        cleaned = cleanup_words(section.words, policy)
        if cleaned == section.words:
            result.append(section)
            continue
        result.append(
            TranscriptSection(
                start_s=section.start_s,
                end_s=section.end_s,
                slide=section.slide,
                slide_ocr=section.slide_ocr,
                words=cleaned,
                paragraphs=section.paragraphs,
            )
        )
    return tuple(result)


def cleanup_words(
    words: Sequence[Word], policy: CleanupPolicy | None = None
) -> tuple[Word, ...]:
    """Вернуть слова без филлеров и оборванных самоповторов.

    Объекты `Word` не пересоздаются — результат состоит из тех же объектов
    в том же порядке, поэтому таймкоды остаются на месте.
    """
    policy = policy or CleanupPolicy()
    if not policy.enabled or not words:
        return tuple(words)

    survivors = list(range(len(words)))
    survivors = _drop_fillers(words, survivors, policy)
    for size in range(policy.max_repeat_ngram, 0, -1):
        survivors = _drop_repeats(words, survivors, size)
    survivors = _drop_truncated_starts(words, survivors)

    if not survivors:
        return tuple(words)
    return tuple(words[i] for i in survivors)


# --------------------------------------------------------------------------
# Внутреннее
# --------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return text.strip(_STRIP_CHARS).lower().replace("ё", "е")


def _protected(word: Word) -> bool:
    """Слово несёт конец предложения — не трогаем."""
    return word.text.rstrip().endswith(_SENTENCE_END)


def _starts_upper(word: Word) -> bool:
    """Слово начинается с заглавной буквы (после стадии пунктуации — начало
    предложения либо имя собственное)."""
    stripped = word.text.lstrip(_STRIP_CHARS)
    return bool(stripped) and stripped[0].isupper()


def _is_math_token(early: Word, late: Word) -> bool:
    """Позиция повтора (слово ранней копии и парное ему слово поздней) несёт
    число, знак формулы или обозначение — повтор содержательный, не оговорка.

    Однобуквенный токен считается обозначением («n», «x», «А», «а», «о», «м»):
    повтор такого токена («Пусть n n стремится», «точка А А равна», «центр
    о о один») слишком похож на диктовку, чтобы его резать. Исключение —
    строчные русские служебные слова из `_SERVICE_LETTERS`: их повтор —
    обычная запинка («в в этом случае»), а буквы B, C, K вслух называют
    «вэ», «цэ», «ка», а не «в», «с», «к». Если обе копии написаны заглавной
    («точка В В»), это всё же имя точки, и защита остаётся.
    """
    norm = _normalize(early.text)
    if not norm:
        return False
    if len(norm) == 1:
        if norm in _SERVICE_LETTERS:
            return _starts_upper(early) and _starts_upper(late)
        return True
    if any(ch.isdigit() for ch in norm):
        return True
    return norm in NUMERAL_WORDS or norm in MATH_TOKENS


def _set_off_left(
    words: Sequence[Word], survivors: list[int], pos: int, policy: CleanupPolicy
) -> bool:
    """Слева от `pos` вставка обособлена: у предыдущего слова запятая/тире
    И перед оборотом пауза `filler_gap_s`.

    Начало фрагмента обособлением не считается, одна пауза — тоже: без
    пунктуации вставку от начала фразы не отличить.
    """
    if pos == 0:
        return False
    previous = words[survivors[pos - 1]]
    if not previous.text.rstrip().endswith(_CLAUSE_END):
        return False
    gap = words[survivors[pos]].start_s - previous.end_s
    return gap >= policy.filler_gap_s


def _set_off_right(
    words: Sequence[Word],
    survivors: list[int],
    last: int,
    policy: CleanupPolicy,
) -> bool:
    """Справа от `last` вставка обособлена: у последнего слова оборота
    запятая/тире И после него пауза `filler_gap_s` (на конце фрагмента пауза
    не проверяется — сравнивать не с чем)."""
    current = words[survivors[last]]
    if not current.text.rstrip().endswith(_CLAUSE_END):
        return False
    if last + 1 >= len(survivors):
        return True
    gap = words[survivors[last + 1]].start_s - current.end_s
    return gap >= policy.filler_gap_s


def _is_insertion(
    words: Sequence[Word],
    survivors: list[int],
    pos: int,
    size: int,
    policy: CleanupPolicy,
) -> bool:
    """Омонимичный оборот в позиции `pos` — вставка, а не часть предложения.

    Требования (все сразу, при невыполнении любого оборот остаётся):

    * оборот не открывает предложение — иначе «Как бы вы доказали это»
      превратится в утверждение, а «Это самое важное свойство» потеряет
      подлежащее. Признак начала предложения — заглавная буква оборота или
      точка у предыдущего слова;
    * оборот обособлен **пунктуацией** с обеих сторон (запятая/тире у
      предыдущего слова и у последнего слова оборота) **и** паузой
      `filler_gap_s` с обеих сторон. Пауза — лишь дополнительное условие:
      сама по себе она омонимию не разрешает, перед «как бы вы доказали»
      лектор делает её так же, как перед вставным «как бы». Поэтому в тексте
      без пунктуации (стадия пунктуации отказала или бэкенд её не дал)
      омонимичный оборот не удаляется никогда, а «Итак, как бы вы…» с
      запятой только слева — тоже.
    """
    if _starts_upper(words[survivors[pos]]):
        return False
    if pos > 0 and _protected(words[survivors[pos - 1]]):
        return False
    return _set_off_left(words, survivors, pos, policy) and _set_off_right(
        words, survivors, pos + size - 1, policy
    )


def _drop_fillers(
    words: Sequence[Word], survivors: list[int], policy: CleanupPolicy
) -> list[int]:
    norms = [_normalize(words[i].text) for i in survivors]
    drop = [False] * len(survivors)

    for pos, norm in enumerate(norms):
        if norm in FILLER_WORDS and not _protected(words[survivors[pos]]):
            drop[pos] = True

    for phrase in (*FILLER_PHRASES, *AMBIGUOUS_FILLER_PHRASES):
        ambiguous = phrase in AMBIGUOUS_FILLER_PHRASES
        size = len(phrase)
        pos = 0
        while pos + size <= len(norms):
            window = tuple(norms[pos : pos + size])
            removable = (
                window == phrase
                and not any(
                    _protected(words[survivors[pos + k]]) for k in range(size)
                )
                and (
                    not ambiguous
                    or _is_insertion(words, survivors, pos, size, policy)
                )
            )
            if removable:
                for k in range(size):
                    drop[pos + k] = True
                pos += size
            else:
                pos += 1

    return [idx for pos, idx in enumerate(survivors) if not drop[pos]]


def _drop_repeats(
    words: Sequence[Word], survivors: list[int], size: int
) -> list[int]:
    """Схлопнуть серию непосредственно повторённой n-граммы до одной копии.

    Скан идёт по уже прореженному списку и после удаления остаётся на месте,
    поэтому серия любой длины («Мы мы мы») доводится до одной копии.

    Какую копию удалять:

    * по умолчанию раннюю — лектор договорил последнюю; её пунктуация
      («мы, конечно») принадлежит предложению, а запятая после ранней копии
      («это, это важно») — лишь разделитель запинки;
    * если заглавную несёт только ранняя копия («Он он знает»), удаляется
      поздняя — но лишь когда копии совпадают посимвольно с точностью до
      регистра. Иначе любое удаление портит текст: пропадёт заглавная или
      пропадёт/повиснет запятая («Это, это важно», «Мы мы, конечно»), а
      перенести регистр на выжившее слово нельзя — `Word` не пересоздаётся
      (решение 2.4.2). Такой повтор остаётся: лишнее слово безвредно.
    """
    kept = list(survivors)
    pos = 0
    while pos + 2 * size <= len(kept):
        early = [words[i] for i in kept[pos : pos + size]]
        late = [words[i] for i in kept[pos + size : pos + 2 * size]]
        first = [_normalize(w.text) for w in early]
        second = [_normalize(w.text) for w in late]
        repeated = first == second and all(first)
        if repeated and any(_is_math_token(e, l) for e, l in zip(early, late)):
            # Число, знак формулы или однобуквенное обозначение: повтор
            # содержательный («два плюс два плюс два»), не оговорка.
            repeated = False
        if repeated and size == 1 and first[0] in ALLOWED_DOUBLES:
            # Тройной повтор допустимого удвоения — всё равно самоповтор.
            repeated = (
                pos + 3 <= len(kept)
                and _normalize(words[kept[pos + 2]].text) == first[0]
            )
        target: int | None = None
        if repeated:
            if not (_starts_upper(early[0]) and not _starts_upper(late[0])):
                target = pos
            elif _same_up_to_case(early, late):
                target = pos + size
        if target is not None and not any(
            _protected(words[i]) for i in kept[target : target + size]
        ):
            del kept[target : target + size]
            continue
        pos += 1
    return kept


def _same_up_to_case(early: Sequence[Word], late: Sequence[Word]) -> bool:
    """Копии совпадают посимвольно, включая пунктуацию, без учёта регистра."""
    return all(
        e.text.strip().lower().replace("ё", "е")
        == l.text.strip().lower().replace("ё", "е")
        for e, l in zip(early, late)
    )


def _drop_truncated_starts(words: Sequence[Word], survivors: list[int]) -> list[int]:
    """«давайте рас- рассмотрим» -> «давайте рассмотрим»."""
    drop = [False] * len(survivors)
    for pos in range(len(survivors) - 1):
        raw = words[survivors[pos]].text.strip(" \t«»\"'`")
        if not raw.endswith("-"):
            continue
        stem = _normalize(raw).rstrip("-")
        nxt = _normalize(words[survivors[pos + 1]].text)
        if stem and nxt.startswith(stem) and len(nxt) > len(stem):
            if not _protected(words[survivors[pos]]):
                drop[pos] = True
    return [idx for pos, idx in enumerate(survivors) if not drop[pos]]
