"""Тесты стадии `merge` (задачи 6.1–6.6).

Все данные синтетические: списки `Word` и `Slide`/`SlideOcr` по контракту.
Эталонная запись здесь не обрабатывается — сценарии 6.2 и 6.4 воспроизведены
эквивалентными синтетическими случаями (фраза через момент переключения
слайда; десятиминутный поток слов с паузами разной длины).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lecture_transcript.config import TranscriptAssemblyConfig
from lecture_transcript.contracts import (
    OcrFragment,
    Rect,
    Slide,
    SlideOcr,
    Transcription,
    Word,
)
from lecture_transcript.transcript_assembly import (
    BoundaryPolicy,
    CleanupPolicy,
    ConsolidationPolicy,
    RenderPolicy,
    consolidate_sections,
    assemble,
    cleanup_words,
    merge_sections,
    render_markdown,
    split_words,
    write_transcript,
)

REGION = Rect(x=0, y=0, width=1280, height=720)


# --------------------------------------------------------------------------
# Хелперы построения синтетики
# --------------------------------------------------------------------------


def mk_words(spec: list[tuple[str, float, float]]) -> tuple[Word, ...]:
    return tuple(Word(text=t, start_s=s, end_s=e) for t, s, e in spec)


def mk_slide(index: int, start_s: float, end_s: float, root: Path | None = None) -> Slide:
    root = root or Path("out")
    return Slide(
        index=index,
        start_s=start_s,
        end_s=end_s,
        region=REGION,
        representative_timestamp_s=(start_s + end_s) / 2,
        image_path=root / "slides" / f"{index:03d}.png",
    )


def mk_ocr(slide: Slide, markdown: str, *, unreliable: bool = False) -> SlideOcr:
    return SlideOcr(
        slide_index=slide.index,
        image_path=slide.image_path,
        fragments=(OcrFragment(text=markdown, kind="text", confidence=0.99),),
        markdown=markdown,
        unreliable=unreliable,
        backend="paddle+pix2tex",
    )


def even_words(count: int, start_s: float = 0.0, step: float = 0.5) -> tuple[Word, ...]:
    return mk_words(
        [(f"слово{i}", start_s + i * step, start_s + i * step + step * 0.7)
         for i in range(count)]
    )


# --------------------------------------------------------------------------
# 6.1 Полнота слияния
# --------------------------------------------------------------------------


def test_merge_preserves_every_word_in_order() -> None:
    words = even_words(120, step=0.5)  # 0..60 c
    slides = [mk_slide(1, 0.0, 20.0), mk_slide(2, 25.0, 45.0), mk_slide(3, 45.0, 60.0)]

    sections = merge_sections(words, slides)

    merged = tuple(w for s in sections for w in s.words)
    assert merged == words
    assert len(merged) == len(set(id(w) for w in merged))


def test_speech_outside_slides_goes_to_own_section() -> None:
    words = mk_words(
        [
            ("Начнём", 1.0, 1.5),
            ("занятие.", 1.6, 2.2),
            ("Пока", 12.0, 12.4),  # демонстрации нет
            ("демонстрации", 12.5, 13.2),
            ("нет.", 13.3, 13.8),
            ("Вот", 21.0, 21.4),
            ("слайд.", 21.5, 22.0),
        ]
    )
    slides = [mk_slide(1, 0.0, 10.0), mk_slide(2, 20.0, 30.0)]

    sections = merge_sections(words, slides)

    kinds = [(s.slide.index if s.slide else None) for s in sections]
    assert kinds == [1, None, 2]
    assert tuple(w.text for w in sections[1].words) == ("Пока", "демонстрации", "нет.")
    assert tuple(w for s in sections for w in s.words) == words


def test_slide_without_speech_still_produces_section() -> None:
    words = mk_words([("Тишина", 0.5, 1.0), ("была", 1.1, 1.6)])
    slides = [mk_slide(1, 0.0, 5.0), mk_slide(2, 5.0, 10.0)]

    sections = merge_sections(words, slides)

    assert [s.slide.index for s in sections if s.slide] == [1, 2]
    assert sections[1].words == ()


def test_sections_are_ordered_chronologically() -> None:
    """Немой слайд не уезжает в конец из-за сдвига границы соседней секции.

    Допуск 6.2 переносит слово «слайд» (1.8–2.6) в секцию слайда 3, и по
    индексу первого слова эта секция оказывается раньше немого слайда 2.
    Порядок в транскрипте — хронологический, поэтому секции идут 1, 2, 3.
    """
    words = mk_words(
        [
            ("Первый", 0.5, 1.3),
            ("слайд", 1.8, 2.6),
            ("дальше", 2.7, 3.5),
            ("третий.", 3.6, 4.4),
        ]
    )
    slides = [mk_slide(1, 0.0, 2.0), mk_slide(2, 2.5, 3.0), mk_slide(3, 3.0, 5.0)]

    sections = merge_sections(words, slides, policy=BoundaryPolicy(tolerance_s=2.0))

    assert [s.slide.index for s in sections if s.slide] == [1, 2, 3]
    starts = [s.start_s for s in sections]
    assert starts == sorted(starts), "интервалы в заголовках идут назад"
    assert tuple(w for s in sections for w in s.words) == words


def test_zero_duration_word_inside_slide_stays_in_its_section() -> None:
    """Слово нулевой длительности (ASR пропускает его дальше с предупреждением)
    внутри интервала слайда относится к этому слайду: пересечение нулевой длины
    — это вложенность, а не отсутствие пересечения. Иначе слайд выводится
    дважды, а между копиями встаёт «Речь вне слайдов»."""
    slide1 = mk_slide(1, 0.0, 30.0)
    slide2 = mk_slide(2, 30.0, 60.0)
    words = mk_words(
        [
            ("Рассмотрим", 1.0, 1.5),
            ("предел", 1.6, 2.2),
            ("функции.", 2.3, 2.3),
            ("Он", 5.0, 5.3),
            ("равен", 5.4, 5.8),
            ("нулю.", 5.9, 6.4),
        ]
    )

    sections = merge_sections(words, [slide1, slide2], [mk_ocr(slide1, "$x^2$")])

    assert [(s.slide.index if s.slide else None) for s in sections] == [1, 2]
    assert sections[0].words == words
    md = render_markdown(sections)
    assert md.count("## Слайд 1 ") == 1
    assert "Речь вне слайдов" not in md


def test_zero_duration_word_is_attached_by_point_in_interval() -> None:
    """Точка на моменте переключения относится к следующему слайду
    (интервал полуоткрытый), точка на конце последнего слайда — к нему."""
    words = mk_words(
        [
            ("раз", 29.0, 29.5),
            ("точка", 30.0, 30.0),
            ("два", 30.5, 31.0),
            ("конец", 60.0, 60.0),
        ]
    )
    slides = [mk_slide(1, 0.0, 30.0), mk_slide(2, 30.0, 60.0)]

    sections = merge_sections(words, slides, policy=BoundaryPolicy(tolerance_s=0.0))

    assert section_of(sections, "точка").slide.index == 2
    assert section_of(sections, "конец").slide.index == 2


def test_equal_overlap_tie_break_not_cut_short_by_nested_slide() -> None:
    """Правило привязки: при равном пересечении побеждает меньший `index`.
    Короткий слайд, закончившийся до слова, не должен обрывать обход кандидатов
    (P2 №15 итерации 1; достижимо только на вложенных интервалах)."""
    slides = [mk_slide(1, 0.0, 20.0), mk_slide(3, 10.0, 11.0), mk_slide(2, 11.5, 20.0)]

    sections = merge_sections(
        mk_words([("слово", 15.0, 16.0)]), slides, policy=BoundaryPolicy(tolerance_s=0.0)
    )

    assert section_of(sections, "слово").slide.index == 1


def test_word_assigned_to_slide_with_larger_overlap() -> None:
    words = mk_words([("граничное", 9.8, 10.6)])  # 0.6 c во втором слайде против 0.2 в первом
    slides = [mk_slide(1, 0.0, 10.0), mk_slide(2, 10.0, 20.0)]

    sections = merge_sections(words, slides, policy=BoundaryPolicy(tolerance_s=0.0))

    with_words = [s for s in sections if s.words]
    assert len(with_words) == 1
    assert with_words[0].slide is not None and with_words[0].slide.index == 2


# --------------------------------------------------------------------------
# 6.2 Допуск на границах секций
# --------------------------------------------------------------------------

# Эквивалент момента переключения слайда на эталонной записи: фраза
# «следующий пример.» начинается до переключения (10.0 с) и кончается после.
BOUNDARY_WORDS = mk_words(
    [
        ("Итак,", 6.0, 6.3),
        ("давайте", 6.4, 6.8),
        ("разберём", 6.9, 7.4),
        ("следующий", 9.0, 9.5),
        ("пример.", 9.9, 10.6),
        ("Здесь", 11.6, 12.0),
        ("мы", 12.1, 12.4),
        ("видим", 12.5, 13.0),
    ]
)
BOUNDARY_SLIDES = [mk_slide(1, 0.0, 10.0), mk_slide(2, 10.0, 20.0)]


def section_of(sections, text: str):
    for section in sections:
        if any(w.text == text for w in section.words):
            return section
    raise AssertionError(f"слово {text!r} потеряно")


def test_phrase_across_slide_switch_is_not_split() -> None:
    without_tolerance = merge_sections(
        BOUNDARY_WORDS, BOUNDARY_SLIDES, policy=BoundaryPolicy(tolerance_s=0.0)
    )
    # Без допуска фраза действительно разрывается посередине.
    assert section_of(without_tolerance, "следующий") is not section_of(
        without_tolerance, "пример."
    )

    sections = merge_sections(
        BOUNDARY_WORDS, BOUNDARY_SLIDES, policy=BoundaryPolicy(tolerance_s=2.0)
    )
    assert section_of(sections, "следующий") is section_of(sections, "пример.")
    # Граница уехала только до ближайшей границы фразы, начало осталось на месте.
    assert section_of(sections, "разберём").slide.index == 1
    assert section_of(sections, "Здесь").slide.index == 2
    assert tuple(w for s in sections for w in s.words) == BOUNDARY_WORDS


def test_tolerance_does_not_swallow_short_slide() -> None:
    """Клэмп `low`/`high` в `_shift_boundaries` не даёт границе уехать за
    соседнюю секцию (решение 2.4.1).

    Данные подобраны так, что защита реально исполняется:

    * речь идёт сплошным потоком с промежутками 0.1 с и без точек, поэтому
      исходные точки разреза (индексы 3 и 5) границами фразы НЕ являются —
      сдвиг обязан искать кандидата;
    * единственная граница фразы в записи — пауза 1.9 с перед словом «новая»
      (индекс 8), то есть уже за интервалом короткого слайда 2;
    * `tolerance_s` заведомо перекрывает обе соседние секции.

    Без клэмпа обе границы уедут на индекс 8, и слайд 2 останется без слов.
    """
    words = mk_words(
        [
            ("Возьмём", 9.0, 9.3),
            ("первый", 9.4, 9.7),
            ("пример", 9.8, 10.1),
            ("короткая", 10.2, 10.5),
            ("вставка", 10.6, 10.9),
            ("продолжаем", 11.0, 11.3),
            ("мысль", 11.4, 11.7),
            ("дальше", 11.8, 12.1),
            ("новая", 14.0, 14.3),
            ("тема", 14.4, 14.7),
        ]
    )
    slides = [mk_slide(1, 0.0, 10.0), mk_slide(2, 10.0, 11.0), mk_slide(3, 11.0, 30.0)]

    sections = merge_sections(words, slides, policy=BoundaryPolicy(tolerance_s=1000.0))

    by_index = {s.slide.index: s for s in sections if s.slide}
    assert all(by_index[i].words for i in (1, 2, 3)), "короткий слайд опустошён"
    assert section_of(sections, "короткая").slide.index == 2
    # Сдвиг действительно произошёл: слово из интервала слайда 3 ушло к слайду 2
    # (граница поехала вправо до паузы перед «новая»), но не дальше клэмпа.
    assert section_of(sections, "дальше").slide.index == 2
    assert section_of(sections, "новая").slide.index == 3
    assert tuple(w for s in sections for w in s.words) == words


def test_tolerance_does_not_shift_boundary_left_past_previous_cut() -> None:
    """Зеркало предыдущего теста: левая половина клэмпа (`low`).

    Единственная граница фразы — после «Итак.» (индекс 1), то есть ДО первого
    разреза (индекс 4). Первая граница законно уезжает влево до неё. Вторая
    граница (индекс 5 -> 6) при снятом `low` тоже нашла бы этого кандидата
    (он в пределах `tolerance_s`) и уехала бы за уже поставленную первую —
    короткий слайд 2 остался бы без слов.
    """
    words = mk_words(
        [
            ("Итак.", 1.0, 1.3),
            ("возьмём", 9.0, 9.3),
            ("первый", 9.4, 9.7),
            ("пример", 9.8, 10.1),
            ("короткая", 10.2, 10.5),
            ("вставка", 10.6, 10.9),
            ("продолжаем", 11.0, 11.3),
            ("мысль", 11.4, 11.7),
            ("дальше", 11.8, 12.1),
        ]
    )
    slides = [mk_slide(1, 0.0, 10.0), mk_slide(2, 10.0, 11.0), mk_slide(3, 11.0, 30.0)]

    sections = merge_sections(words, slides, policy=BoundaryPolicy(tolerance_s=1000.0))

    by_index = {s.slide.index: s for s in sections if s.slide}
    assert all(by_index[i].words for i in (1, 2, 3)), "короткий слайд опустошён"
    assert section_of(sections, "короткая").slide.index == 2
    # Сдвиг исполнялся: первая граница ушла влево до «Итак.».
    assert section_of(sections, "возьмём").slide.index == 2
    assert section_of(sections, "продолжаем").slide.index == 3
    assert tuple(w for s in sections for w in s.words) == words


# --------------------------------------------------------------------------
# 6.3 Чистка филлеров и самоповторов
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, expected",
    [
        (["э", "мы", "рассмотрим"], ["мы", "рассмотрим"]),
        (["ээ", "ну", "вот", "смотрите"], ["смотрите"]),
        (["то", "есть", "то", "есть", "важно"], ["то", "есть", "важно"]),
        (["мы", "мы", "мы", "рассмотрим"], ["мы", "рассмотрим"]),
        (["давайте", "рас-", "рассмотрим"], ["давайте", "рассмотрим"]),
    ],
)
def test_cleanup_removes_obvious_noise(source: list[str], expected: list[str]) -> None:
    words = mk_words([(t, i * 1.0, i * 1.0 + 0.5) for i, t in enumerate(source)])
    assert [w.text for w in cleanup_words(words)] == expected


@pytest.mark.parametrize(
    "source",
    [
        ["очень", "очень", "важный", "результат"],
        ["то", "есть", "предел", "равен", "нулю"],
        ["как", "раз", "этот", "случай"],
        # Без пунктуации омонимичный оборот не удаляется (см. тест ниже).
        ["как", "бы", "предел", "функции"],
        ["мы", "рассмотрим", "предел", "функции"],
        ["ну", "и", "получаем", "ответ"],
        ["чуть", "чуть", "правее"],
    ],
)
def test_cleanup_keeps_meaningful_speech(source: list[str]) -> None:
    words = mk_words([(t, i * 1.0, i * 1.0 + 0.5) for i, t in enumerate(source)])
    assert [w.text for w in cleanup_words(words)] == source


@pytest.mark.parametrize(
    "source",
    [
        # Находка 3: омонимичные обороты в начале предложения — часть фразы.
        ["Как", "бы", "вы", "доказали", "это"],
        ["Так", "сказать", "нельзя"],
        ["Это", "самое", "важное", "свойство"],
        ["Скажем", "так", "же", "и", "для", "суммы"],
        # Находка 4: арифметика и диктовка формул — не самоповтор.
        ["два", "плюс", "два", "плюс", "два"],
        ["икс", "плюс", "икс", "плюс", "икс", "равно"],
        ["ноль", "ноль", "один"],
        ["раз", "два", "раз", "два", "три"],
        ["Пусть", "n", "n", "стремится"],
        ["точка", "А", "А", "равна"],
        # Находка 6: однобуквенный токен — единица измерения или обозначение.
        ["длина", "равна", "пяти", "м", "и", "ширина"],
        ["возьмём", "м", "равное", "трём"],
    ],
)
def test_cleanup_keeps_mathematical_speech(source: list[str]) -> None:
    """Лекция по математике: числительные, обозначения и омонимы филлеров
    остаются в тексте — «сомнительное не чистится» (design, раздел Risks)."""
    words = mk_words([(t, i * 1.0, i * 1.0 + 0.5) for i, t in enumerate(source)])
    assert [w.text for w in cleanup_words(words)] == source


def test_cleanup_keeps_ambiguous_phrase_inside_dense_sentence() -> None:
    """«предел как бы равен» без обособления — оборот остаётся: без паузы и
    запятой вставку от части предложения не отличить."""
    words = mk_words(
        [
            ("предел", 0.0, 0.6),
            ("как", 0.65, 0.85),
            ("бы", 0.9, 1.05),
            ("равен", 1.1, 1.6),
        ]
    )
    assert [w.text for w in cleanup_words(words)] == [
        "предел",
        "как",
        "бы",
        "равен",
    ]


def test_cleanup_removes_ambiguous_phrase_when_it_is_an_insertion() -> None:
    """Тот же оборот, обособленный запятыми и паузами, — это вставка."""
    words = mk_words(
        [
            ("предел,", 0.0, 0.6),
            ("как", 1.0, 1.2),
            ("бы,", 1.3, 1.5),
            ("равен", 2.0, 2.5),
            ("нулю.", 2.6, 3.1),
        ]
    )
    assert [w.text for w in cleanup_words(words)] == ["предел,", "равен", "нулю."]


@pytest.mark.parametrize("gap_s", [0.05, 0.35, 1.0, 3.0])
@pytest.mark.parametrize(
    "source",
    [
        # Текст без пунктуации (стадия пунктуации отказала): пауза омонимию
        # не разрешает — перед «как бы вы…» лектор делает её так же, как
        # перед вставным «как бы».
        ["как", "бы", "вы", "доказали", "это"],
        ["Как", "бы", "вы", "доказали", "это"],
        ["это", "самое", "важное", "свойство"],
        ["так", "сказать", "нельзя"],
        ["и", "скажем", "так", "же", "для", "суммы"],
        ["предел", "как", "бы", "равен", "нулю"],
        # Пунктуация есть, но только с одной стороны — это не вставка.
        ["Итак,", "как", "бы", "вы", "доказали", "это?"],
        ["Скажите,", "как", "бы", "вы", "это", "доказали?"],
    ],
)
def test_cleanup_keeps_ambiguous_phrase_without_punctuation_on_both_sides(
    source: list[str], gap_s: float
) -> None:
    words = mk_words(
        [(t, i * (0.4 + gap_s), i * (0.4 + gap_s) + 0.4) for i, t in enumerate(source)]
    )
    assert [w.text for w in cleanup_words(words)] == source


def test_cleanup_removes_unambiguous_fillers_without_punctuation() -> None:
    """Контроль: однозначные филлеры и самоповторы в тексте без пунктуации
    по-прежнему удаляются — правило для омонимов их не затрагивает."""
    source = ["ээ", "ну", "вот", "смотрите", "мы", "мы", "рассмотрим", "э", "предел"]
    words = mk_words([(t, i * 0.75, i * 0.75 + 0.4) for i, t in enumerate(source)])
    assert [w.text for w in cleanup_words(words)] == [
        "смотрите",
        "мы",
        "рассмотрим",
        "предел",
    ]


def test_cleanup_repeat_keeps_capital_of_sentence_start() -> None:
    """Находка 5: из пары «Он он» уходит поздняя копия, заглавная остаётся."""
    words = mk_words([("Он", 0.0, 0.3), ("он", 0.4, 0.7), ("знает", 0.8, 1.3)])
    assert [w.text for w in cleanup_words(words)] == ["Он", "знает"]


@pytest.mark.parametrize(
    "source, expected",
    [
        # Тройная запинка в начале предложения дочищается до одной копии.
        (["Мы", "мы", "мы", "рассмотрим"], ["Мы", "рассмотрим"]),
        (["То", "есть", "то", "есть", "то", "есть", "предел"], ["То", "есть", "предел"]),
        # Однобуквенные служебные слова — не обозначения: их повтор — запинка.
        (["в", "в", "этом", "случае"], ["в", "этом", "случае"]),
        (["и", "и", "тогда"], ["и", "тогда"]),
        (["я", "думаю", "я", "думаю", "что"], ["я", "думаю", "что"]),
        (["В", "в", "этом", "случае"], ["В", "этом", "случае"]),
        # Без конфликта регистра запятая-разделитель уходит с ранней копией.
        (["это,", "это", "важно."], ["это", "важно."]),
    ],
)
def test_cleanup_finishes_repeats_without_breaking_text(
    source: list[str], expected: list[str]
) -> None:
    words = mk_words([(t, i * 1.0, i * 1.0 + 0.5) for i, t in enumerate(source)])
    assert [w.text for w in cleanup_words(words)] == expected


@pytest.mark.parametrize(
    "source",
    [
        # Удалить позднюю копию — повиснет/пропадёт запятая, раннюю — пропадёт
        # заглавная; новый `Word` создавать нельзя (решение 2.4.2) -> не трогаем.
        ["Это,", "это", "важно."],
        ["Мы", "мы,", "конечно,", "рассмотрим"],
        # Обозначения точек и отрезков остаются под защитой после сужения.
        ["точка", "В", "В", "равна"],
        ["точка", "а", "а", "равна"],
        ["центр", "о", "о", "один"],
        ["отрезок", "бэ", "бэ", "один"],
    ],
)
def test_cleanup_keeps_repeat_when_removal_would_damage_text(source: list[str]) -> None:
    words = mk_words([(t, i * 1.0, i * 1.0 + 0.5) for i, t in enumerate(source)])
    assert [w.text for w in cleanup_words(words)] == source


def test_cleanup_keeps_timecodes_of_surviving_words() -> None:
    words = mk_words(
        [
            ("э", 0.0, 0.2),
            ("предел,", 0.3, 0.9),
            ("как", 1.4, 1.6),
            ("бы,", 1.65, 1.8),
            ("равен", 2.2, 2.7),
            ("нулю.", 2.8, 3.3),
        ]
    )
    cleaned = cleanup_words(words)

    assert [w.text for w in cleaned] == ["предел,", "равен", "нулю."]
    originals = {w.text: w for w in words}
    for word in cleaned:
        assert word is originals[word.text]  # объекты те же, таймкоды не сдвинуты


def test_cleanup_does_not_drop_word_carrying_sentence_end() -> None:
    words = mk_words([("Ну", 0.0, 0.3), ("вот.", 0.4, 0.9), ("Дальше", 1.0, 1.5)])
    assert [w.text for w in cleanup_words(words)] == ["Ну", "вот.", "Дальше"]


def test_cleanup_disabled_is_identity() -> None:
    words = mk_words([("э", 0.0, 0.2), ("ну", 0.3, 0.5), ("вот", 0.6, 0.9)])
    assert cleanup_words(words, CleanupPolicy(enabled=False)) == words


# --------------------------------------------------------------------------
# 6.4 Абзацы
# --------------------------------------------------------------------------


def build_monologue(total_s: float = 600.0) -> tuple[tuple[Word, ...], int]:
    """Десятиминутный монолог: слова по 0.5 с, каждые 12 слов — пауза 2.6 с."""
    words: list[tuple[str, float, float]] = []
    t = 0.0
    i = 0
    pauses = 0
    while t < total_s:
        in_sentence = i % 12
        text = f"слово{i}" + ("." if in_sentence == 11 else "")
        words.append((text, t, t + 0.35))
        t += 0.5
        if in_sentence == 11:
            t += 2.6  # смысловая пауза
            pauses += 1
        i += 1
    return mk_words(words), pauses


def test_long_monologue_split_into_reasonable_paragraphs() -> None:
    words, pauses = build_monologue()
    groups = split_words(words)

    # Ожидания абсолютные, а не вычисленные теми же константами, что и код:
    # 840 слов, 70 пауз, десятиминутный монолог. Абзац конспекта — порядка
    # минуты речи, значит абзацев единицы-десятки, а не сотня по числу пауз.
    assert len(groups) > 1, "текст не должен быть одним блоком"
    assert 4 <= len(groups) <= 20, f"абзацев {len(groups)} при {pauses} паузах"
    assert sum(len(g) for g in groups) == len(words)
    durations = [g[-1].end_s - g[0].start_s for g in groups]
    assert max(durations) <= 90.0, "абзац длиннее полутора минут нечитаем"
    assert min(durations) >= 20.0, "абзац короче 20 с — разрыв по каждой паузе"


def test_paragraph_split_is_not_triggered_by_every_pause() -> None:
    # Короткий фрагмент с двумя паузами — остаётся одним абзацем.
    words = mk_words(
        [("раз", 0.0, 0.4), ("два.", 0.5, 1.0), ("три", 4.0, 4.4), ("четыре.", 4.5, 5.0)]
    )
    assert len(split_words(words)) == 1


# --------------------------------------------------------------------------
# 6.5 Шаблон вывода
# --------------------------------------------------------------------------


def build_render_case(root: Path):
    slide1 = mk_slide(12, 2400.0, 2880.0, root=root)
    slide2 = mk_slide(13, 2880.0, 2900.0, root=root)
    ocr1 = mk_ocr(
        slide1,
        "**Пример 2.** $\\sqrt{x^2-x} = -5$\n"
        f"![слайд 12]({(root / 'slides' / '012.png').resolve()})",
    )
    ocr2 = mk_ocr(slide2, "", unreliable=True)
    words = list(
        mk_words([(f"речь{i}", 2400.0 + i * 2.0, 2400.0 + i * 2.0 + 1.5) for i in range(200)])
    )
    words += list(mk_words([("Короткая", 2882.0, 2882.6), ("секция.", 2882.7, 2883.4)]))
    transcription = Transcription(
        words=tuple(words), backend="gigaam-v2", has_punctuation=True
    )
    return transcription, [slide1, slide2], [ocr1, ocr2]


def test_render_has_heading_inset_image_and_formula(tmp_path: Path) -> None:
    out = tmp_path / "out"
    transcription, slides, ocrs = build_render_case(out)

    transcript = assemble(
        transcription,
        slides,
        ocrs,
        TranscriptAssemblyConfig(),
        source_path=Path("/media/lectures/wr_20260909_1150.mp4"),
        output_dir=out,
        duration_s=5482.0,
    )
    md = transcript.markdown

    # Заголовок секции: номер слайда и интервал.
    # Заголовок: номер, тема из первой строки OCR (без разметки и формулы),
    # интервал в едином формате H:MM:SS.
    assert re.search(r"^## Слайд 12 — Пример 2\. — 0:40:00–0:48:00$", md, re.M)
    assert re.search(r"^## Слайд 13 — 0:48:00–0:48:20$", md, re.M)
    # Врезка содержимого слайда и формула в $...$.
    assert re.search(r"^> \*\*Пример 2\.\*\* \$\\sqrt\{x\^2-x\} = -5\$$", md, re.M)
    # Обязательная ссылка на PNG, путь относительный.
    assert "![Слайд 12](slides/012.png)" in md
    assert "![Слайд 13](slides/013.png)" in md
    assert str(tmp_path) not in md and "/slides/012.png" not in md
    # Ссылка на картинку ровно одна на слайд — врезка OCR не задваивает её.
    assert md.count("slides/012.png") == 1
    # Шапка с метаданными: только имя файла, без абсолютного пути.
    assert "- **Исходный файл:** wr_20260909_1150.mp4" in md
    assert "- **Длительность:** 1:31:22" in md
    assert "- **ASR-бэкенд:** gigaam-v2" in md
    assert "- **OCR-бэкенд:** paddle+pix2tex" in md
    # Ненадёжный/пустой OCR помечен явно.
    assert "не распознан" in md


def test_render_intermediate_timecodes_only_in_long_sections(tmp_path: Path) -> None:
    out = tmp_path / "out"
    transcription, slides, ocrs = build_render_case(out)
    transcript = assemble(
        transcription,
        slides,
        ocrs,
        TranscriptAssemblyConfig(timecode_every_s=120.0),
        source_path=Path("lecture.mp4"),
        output_dir=out,
    )
    long_part, short_part = transcript.markdown.split("## Слайд 13")
    assert re.search(r"^\*\*\[\d+:\d\d:\d\d\]\*\*$", long_part, re.M)
    assert not re.search(r"^\*\*\[\d+:\d\d:\d\d\]\*\*$", short_part, re.M)


def test_first_paragraph_gets_timecode_when_speech_starts_late() -> None:
    """Находка 2: слайд показан 16 минут, речь пошла на 900-й секунде.

    Без таймкода перед первым абзацем время абзаца определяется только
    интервалом в заголовке — с ошибкой во всю длительность секции.
    """
    slide = mk_slide(1, 0.0, 1000.0)
    sections = merge_sections(
        mk_words([("Речь", 900.0, 900.5), ("началась.", 900.6, 901.2)]), [slide]
    )
    md = render_markdown(sections, policy=RenderPolicy(timecode_every_s=120.0))
    assert "**[0:15:00]**" in md


def test_first_paragraph_without_delay_has_no_timecode() -> None:
    """Обратная сторона: речь с начала слайда лишнего таймкода не получает."""
    slide = mk_slide(1, 0.0, 1000.0)
    sections = merge_sections(mk_words([("Речь", 1.0, 1.5)]), [slide])
    md = render_markdown(sections, policy=RenderPolicy(timecode_every_s=120.0))
    assert not re.search(r"^\*\*\[\d+:\d\d:\d\d\]\*\*$", md, re.M)


def test_render_speech_outside_slides_has_own_heading() -> None:
    words = mk_words([("Без", 100.0, 100.5), ("слайда.", 100.6, 101.2)])
    sections = merge_sections(words, [mk_slide(1, 0.0, 50.0)])
    md = render_markdown(sections, policy=RenderPolicy())
    assert re.search(r"^## Речь вне слайдов — 0:01:40–0:01:41$", md, re.M)


def test_render_image_path_stays_relative_for_nested_output(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "out"
    slide = mk_slide(3, 0.0, 10.0, root=(out / "assets").resolve())
    sections = merge_sections(mk_words([("текст", 1.0, 1.5)]), [slide])
    md = render_markdown(sections, output_dir=out)
    assert "![Слайд 3](assets/slides/003.png)" in md


# --------------------------------------------------------------------------
# 6.6 Детерминированность
# --------------------------------------------------------------------------


def test_two_runs_produce_byte_identical_file(tmp_path: Path) -> None:
    config = TranscriptAssemblyConfig()
    produced: list[bytes] = []
    for run in ("run1", "run2"):
        out = tmp_path / run
        transcription, slides, ocrs = build_render_case(out)
        transcript = assemble(
            transcription,
            slides,
            ocrs,
            config,
            source_path=Path("/media/wr_20260909_1150.mp4"),
            output_dir=out,
            duration_s=5482.0,
            extra_metadata={"z_extra": "1", "a_extra": "2"},
        )
        target = write_transcript(transcript, out, config)
        produced.append(target.read_bytes())

    assert produced[0] == produced[1]
    assert produced[0].endswith(b"\n")


def test_render_marks_unreliable_ocr(tmp_path: Path) -> None:
    out = tmp_path / "out"
    slide = mk_slide(7, 0.0, 30.0, root=out)
    ocr = mk_ocr(slide, "**Теорема.** $a^2 + b^2 = c^2$", unreliable=True)
    sections = merge_sections(mk_words([("речь", 1.0, 1.5)]), [slide], [ocr])
    md = render_markdown(sections, output_dir=out)

    assert "> _Распознавание ненадёжно, сверьтесь с изображением слайда._" in md
    assert "> **Теорема.** $a^2 + b^2 = c^2$" in md
    assert "![Слайд 7](slides/007.png)" in md


def test_render_marks_unreliable_ocr_without_recognized_text(tmp_path: Path) -> None:
    """Находка 10: пустой И ненадёжный OCR — два разных состояния, оба явно
    обозначены («OCR не отработал» против «OCR сам себе не доверяет»)."""
    out = tmp_path / "out"
    slide = mk_slide(9, 0.0, 30.0, root=out)
    ocr = mk_ocr(slide, "", unreliable=True)
    sections = merge_sections(mk_words([("речь", 1.0, 1.5)]), [slide], [ocr])
    md = render_markdown(sections, output_dir=out)

    assert "> _Распознавание ненадёжно, сверьтесь с изображением слайда._" in md
    assert "> _Текст на слайде не распознан — смотрите изображение._" in md


def test_completeness_violation_raises_and_survives_optimize_flag() -> None:
    """Находка 13: инвариант 6.1 держится явным `raise`, а не `assert`,
    который выключается флагом `-O`."""
    from lecture_transcript.contracts import PipelineError
    from lecture_transcript.transcript_assembly.merge import _assert_completeness

    words = mk_words([("раз", 0.0, 0.4), ("два", 0.5, 0.9)])
    sections = merge_sections(words, [mk_slide(1, 0.0, 5.0)])
    broken = tuple(
        type(s)(
            start_s=s.start_s,
            end_s=s.end_s,
            slide=s.slide,
            slide_ocr=s.slide_ocr,
            words=s.words[:-1],
            paragraphs=s.paragraphs,
        )
        for s in sections
    )
    with pytest.raises(PipelineError):
        _assert_completeness(words, broken)


def test_render_slide_without_ocr(tmp_path: Path) -> None:
    out = tmp_path / "out"
    slide = mk_slide(4, 0.0, 30.0, root=out)
    sections = merge_sections(mk_words([("речь", 1.0, 1.5)]), [slide])
    md = render_markdown(sections, output_dir=out)

    assert "> _Содержимое слайда не распознано — смотрите изображение._" in md
    assert "![Слайд 4](slides/004.png)" in md


DETERMINISM_SCRIPT = """
import sys
from pathlib import Path

sys.path.insert(0, {tests!r})
from test_transcript_assembly import build_render_case

from lecture_transcript.config import TranscriptAssemblyConfig
from lecture_transcript.transcript_assembly import assemble, write_transcript

out = Path(sys.argv[1])
transcription, slides, ocrs = build_render_case(out)
config = TranscriptAssemblyConfig()
transcript = assemble(
    transcription,
    slides,
    ocrs,
    config,
    source_path=Path("/media/wr_20260909_1150.mp4"),
    output_dir=out,
    duration_s=5482.0,
    extra_metadata={{"z_extra": "1", "a_extra": "2"}},
)
write_transcript(transcript, out, config)
"""


def test_determinism_across_processes_and_hash_seeds(tmp_path: Path) -> None:
    """Разный PYTHONHASHSEED не должен менять вывод: порядок множеств не влияет."""
    import os
    import subprocess
    import sys

    script = tmp_path / "run.py"
    script.write_text(
        DETERMINISM_SCRIPT.format(tests=str(Path(__file__).parent)), encoding="utf-8"
    )

    outputs: list[bytes] = []
    for seed in ("0", "12345"):
        out = tmp_path / f"seed{seed}"
        env = dict(os.environ, PYTHONHASHSEED=seed)
        subprocess.run(
            [sys.executable, str(script), str(out)], check=True, env=env
        )
        outputs.append((out / "transcript.md").read_bytes())

    assert outputs[0] == outputs[1]


# --------------------------------------------------------------------------
# MVP-формат: укрупнение секций, навигация, формат времени, регистр
# --------------------------------------------------------------------------


def kinds(sections) -> list[int | None]:
    return [(s.slide.index if s.slide else None) for s in sections]


def test_short_offslide_section_joins_previous_slide() -> None:
    """Провал демонстрации при смене слайда (4 с) — хвост предыдущего слайда."""
    words = mk_words(
        [
            ("Первый", 1.0, 1.5),
            ("слайд", 1.6, 2.2),
            ("хвост", 31.0, 31.5),
            ("фразы.", 34.5, 35.0),
            ("Второй", 41.0, 41.5),
            ("слайд.", 41.6, 42.2),
        ]
    )
    slides = [mk_slide(1, 0.0, 30.0), mk_slide(2, 40.0, 70.0)]
    merged = merge_sections(words, slides, policy=BoundaryPolicy(tolerance_s=0.0))
    assert kinds(merged) == [1, None, 2]

    sections = consolidate_sections(merged)

    assert kinds(sections) == [1, 2]
    assert [w.text for w in sections[0].words] == ["Первый", "слайд", "хвост", "фразы."]
    assert (sections[0].start_s, sections[0].end_s) == (0.0, 35.0)  # объединение
    assert tuple(w for s in sections for w in s.words) == words


def test_short_offslide_section_without_previous_slide_joins_next() -> None:
    words = mk_words(
        [("Начнём", 1.0, 1.5), ("занятие.", 3.5, 4.0), ("Слайд", 11.0, 11.5)]
    )
    merged = merge_sections(
        words, [mk_slide(1, 10.0, 40.0)], policy=BoundaryPolicy(tolerance_s=0.0)
    )
    assert kinds(merged) == [None, 1]

    sections = consolidate_sections(merged)

    assert kinds(sections) == [1]
    assert sections[0].words == words
    assert sections[0].start_s == 1.0


def test_offslide_section_not_shorter_than_threshold_stays_separate() -> None:
    """Порог строгий: секция ровно 15 с — уже настоящая речь без демонстрации."""
    words = mk_words(
        [
            ("Слайд.", 1.0, 1.5),
            ("Долгое", 31.0, 31.5),
            ("отступление.", 45.5, 46.0),  # 31.0–46.0 = 15.0 с
            ("Снова.", 51.0, 51.5),
        ]
    )
    slides = [mk_slide(1, 0.0, 30.0), mk_slide(2, 50.0, 80.0)]
    merged = merge_sections(words, slides, policy=BoundaryPolicy(tolerance_s=0.0))

    assert kinds(consolidate_sections(merged)) == [1, None, 2]
    wider = ConsolidationPolicy(short_offslide_section_s=15.5)
    assert kinds(consolidate_sections(merged, wider)) == [1, 2]
    assert kinds(consolidate_sections(merged, ConsolidationPolicy(0.0))) == [1, None, 2]


def same_content_case(tmp_path: Path, first_png: bytes, second_png: bytes):
    out = tmp_path / "out"
    first = mk_slide(19, 0.0, 30.0, root=out)
    second = mk_slide(20, 30.0, 60.0, root=out)
    for slide, payload in ((first, first_png), (second, second_png)):
        slide.image_path.parent.mkdir(parents=True, exist_ok=True)
        slide.image_path.write_bytes(payload)
    ocrs = [
        mk_ocr(first, "Линейное уравнение\n\n$x + 1 = 0$\n\n![слайд 19](a.png)"),
        # То же содержимое: другие пробелы, пометка неуверенности, пометка
        # ненадёжности slide-ocr цитатой и другая ссылка на картинку.
        mk_ocr(
            second,
            "> **Распознано неуверенно.** Сверьтесь.\n\nЛинейное  уравнение\n\n"
            "$x + 1 = 0$ _(?)_\n\n![слайд 20](b.png)",
            unreliable=True,
        ),
    ]
    words = mk_words([("начало", 1.0, 1.5), ("решения.", 31.0, 31.5)])
    sections = merge_sections(words, [first, second], ocrs)
    return out, sections


def test_same_content_slides_with_different_png_show_both_images(tmp_path: Path) -> None:
    """Одинаковый OCR (рукописное OCR выбрасывает) — одна секция, но оба PNG:
    иначе первое рукописное решение пропадёт из транскрипта."""
    out, sections = same_content_case(tmp_path, b"blank", b"answer -0.2")

    md = render_markdown(sections, output_dir=out)

    assert len(sections) == 2  # структура не теряет слайды
    headings = re.findall(r"^## .*$", md, re.M)
    assert headings == ["## Слайды 19–20 — Линейное уравнение — 0:00:00–0:01:00"]
    assert md.index("![Слайд 19](slides/019.png)") < md.index("![Слайд 20](slides/020.png)")
    assert len(re.findall(r"^> \$x \+ 1 = 0\$", md, re.M)) == 1  # врезка одна
    assert md.count("ненадёжно") == 1 and "> >" not in md
    assert "\nНачало решения.\n" in md  # речь группы идёт подряд, одним потоком


def test_same_content_slides_with_identical_png_show_it_once(tmp_path: Path) -> None:
    out, sections = same_content_case(tmp_path, b"same bytes", b"same bytes")

    md = render_markdown(sections, output_dir=out)

    assert re.findall(r"^## .*$", md, re.M) == [
        "## Слайды 19–20 — Линейное уравнение — 0:00:00–0:01:00"
    ]
    assert md.count("![Слайд") == 1
    assert "![Слайд 19](slides/019.png)" in md


@pytest.mark.parametrize(
    "first_md, second_md",
    [
        ("![слайд 19](a.png)", "![слайд 20](b.png)"),  # оба пустые — не одно и то же
        ("Линейное уравнение", "Квадратное уравнение"),
    ],
)
def test_slides_with_empty_or_different_ocr_are_not_grouped(
    tmp_path: Path, first_md: str, second_md: str
) -> None:
    first = mk_slide(19, 0.0, 30.0, root=tmp_path)
    second = mk_slide(20, 30.0, 60.0, root=tmp_path)
    ocrs = [mk_ocr(first, first_md), mk_ocr(second, second_md)]
    words = mk_words([("раз", 1.0, 1.5), ("два", 31.0, 31.5)])
    md = render_markdown(merge_sections(words, [first, second], ocrs), output_dir=tmp_path)
    assert len(re.findall(r"^## Слайд \d+ ", md, re.M)) == 2


@pytest.mark.parametrize(
    "markdown, expected",
    [
        ("**Пример 2.** $\\sqrt{x^2-x} = -5$\n\n![слайд](a.png)", "Пример 2."),
        ("$$x^2 + 1$$\n\nЛинейное уравнение", "Линейное уравнение"),
        ("> **Распознано неуверенно.** Сверьтесь.\n\nТема урока _(?)_", "Тема урока"),
        ("Цена 100\\$ за штуку", "Цена 100\\$ за штуку"),
        ("# Заголовок слайда", "Заголовок слайда"),
        ("![слайд](a.png)", ""),
        ("$x$", ""),
    ],
)
def test_slide_title_is_first_text_line_without_markup(markdown: str, expected: str) -> None:
    from lecture_transcript.transcript_assembly.render import slide_title

    slide = mk_slide(1, 0.0, 10.0)
    assert slide_title(mk_ocr(slide, markdown)) == expected


def test_slide_title_is_truncated_on_word_boundary() -> None:
    from lecture_transcript.transcript_assembly.render import TITLE_MAX_CHARS, slide_title

    long_line = " ".join(["уравнение"] * 20)
    title = slide_title(mk_ocr(mk_slide(1, 0.0, 10.0), long_line))
    assert title.endswith("…") and len(title) <= TITLE_MAX_CHARS + 1
    assert title[:-1].split(" ") == ["уравнение"] * len(title[:-1].split(" "))


def test_all_times_in_file_use_one_format(tmp_path: Path) -> None:
    """Заголовки, промежуточные таймкоды и шапка — только H:MM:SS."""
    out = tmp_path / "out"
    transcription, slides, ocrs = build_render_case(out)
    transcript = assemble(
        transcription, slides, ocrs, TranscriptAssemblyConfig(),
        source_path=Path("lecture.mp4"), output_dir=out, duration_s=5482.0,
    )
    md = transcript.markdown
    times = re.findall(r"(?<![\d:])\d+(?::\d\d)+(?![\d:])", md)
    assert times and all(re.fullmatch(r"\d+:\d\d:\d\d", t) for t in times), times


def test_capitalize_sentences_in_rendered_text_only() -> None:
    """Регистр поднимается в строке абзаца; `Word` и таймкоды не трогаются."""
    from lecture_transcript.transcript_assembly.paragraphs import capitalize_sentences

    assert capitalize_sentences("получаем ответ. дальше! ещё… «итак» всё") == (
        "Получаем ответ. Дальше! Ещё… «Итак» всё"
    )
    # Латиница и цифры не трогаются; после сокращения предложение не кончается.
    assert capitalize_sentences("x равно нулю. 2 корня. т. е. предел") == (
        "x равно нулю. 2 корня. Т. е. предел"
    )

    words = mk_words(
        [("Ну", 0.0, 0.3), ("вот", 0.4, 0.7), ("получаем", 0.8, 1.3), ("ответ.", 1.4, 1.9)]
    )
    transcription = Transcription(words=words, backend="asr", has_punctuation=True)
    transcript = assemble(
        transcription, [mk_slide(1, 0.0, 10.0)], (), TranscriptAssemblyConfig(),
        source_path=Path("lecture.mp4"),
    )
    section = transcript.sections[0]
    assert [w.text for w in section.words] == ["получаем", "ответ."]
    assert section.words[0] is words[2]
    assert section.paragraphs == ("Получаем ответ.",)
    assert "\nПолучаем ответ.\n" in transcript.markdown
