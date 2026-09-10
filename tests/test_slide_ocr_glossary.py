"""Глоссарий терминов лекции по всем слайдам (задача 4.7)."""

from __future__ import annotations

from pathlib import Path

from lecture_transcript.contracts import OcrFragment, Rect, SlideOcr
from lecture_transcript.slide_ocr.glossary import (
    DEFAULT_GLOSSARY,
    GlossaryConfig,
    build_glossary,
    extract_terms,
    find_decoration_lines,
    normalize_key,
)

#: Сквозная служебная надпись — логотип платформы в одном и том же месте.
FOOTER = "ЛогоПлатформа Онлайн-школа"
FOOTER_BBOX = Rect(x=40, y=560, width=300, height=30)

CONTENT = [
    ["Квадратные уравнения", "Дискриминант уравнения", "Пример 2."],
    ["Теорема Виета", "Сумма корней уравнения", "Пример 3."],
    ["Иррациональные неравенства", "Область определения"],
    ["Дискриминант и корни", "Уравнения с модулем"],
    ["Логарифмические уравнения", "Основание логарифма"],
]


def _slide(index: int, lines: list[str], footer: bool = True, footer_bbox: Rect = FOOTER_BBOX) -> SlideOcr:
    fragments = [
        OcrFragment(
            text=line,
            kind="text",
            confidence=0.95,
            bbox=Rect(x=40, y=60 + position * 60, width=600, height=40),
        )
        for position, line in enumerate(lines)
    ]
    if footer:
        fragments.append(
            OcrFragment(text=FOOTER, kind="text", confidence=0.93, bbox=footer_bbox)
        )
    return SlideOcr(
        slide_index=index,
        image_path=Path(f"slides/{index:03d}.png"),
        fragments=tuple(fragments),
        markdown="",
        backend="fake",
    )


def _slides(footer: bool = True) -> list[SlideOcr]:
    return [_slide(i + 1, lines, footer=footer) for i, lines in enumerate(CONTENT)]


def _jittered_footer_bbox(position: int) -> Rect:
    """Плашка на своём месте, но боксы дрожат на ±5 px, как у настоящего OCR."""
    dx, dy = (-5, 3, 0, 5, -3, 2, -2, 4, -4, 1)[position % 10], (3, -4, 5, 0, -2)[position % 5]
    return Rect(x=40 + dx, y=560 + dy, width=300, height=30)


def _slides_with_jitter(outlier_at: int | None = None) -> list[SlideOcr]:
    """10 слайдов с плашкой; на слайде `outlier_at` она заметно выше."""
    slides = []
    for position in range(10):
        bbox = _jittered_footer_bbox(position)
        if position == outlier_at:
            bbox = Rect(x=bbox.x, y=120, width=bbox.width, height=bbox.height)
        slides.append(_slide(position + 1, CONTENT[position % len(CONTENT)], footer_bbox=bbox))
    return slides


def test_джиттер_боксов_не_мешает_признать_надпись_оформлением():
    assert "логоплатформа онлайн школа" in find_decoration_lines(_slides_with_jitter())


def test_один_выбившийся_бокс_не_отменяет_отсечение_оформления():
    # Медиана берётся ради устойчивости к выбросам: титульный слайд, где
    # плашка стоит выше, не должен пускать логотип платформы в глоссарий.
    slides = _slides_with_jitter(outlier_at=0)
    assert "логоплатформа онлайн школа" in find_decoration_lines(slides)
    joined = " ".join(build_glossary(slides).terms).casefold()
    assert "логоплатформа" not in joined and "онлайн-школа" not in joined


def test_заголовок_темы_в_шапке_каждого_слайда_остаётся_термином():
    # Тема лекции, повторённая в шапке каждого слайда, удовлетворяет обоим
    # признакам оформления, но это самый ценный для ASR термин: её слова
    # звучат и в содержании слайдов, в отличие от слов плашки платформы.
    header = "Квадратные уравнения"
    body = [
        ["Решите уравнение и запишите ответ", "Дискриминант положителен"],
        ["Квадратные корни и уравнения", "Пример 2."],
        ["Уравнения с параметром", "Дискриминант равен нулю"],
        ["Полное квадратное уравнение", "Пример 3."],
        ["Неполные уравнения", "Квадратные неравенства"],
    ]
    slides = []
    for position, lines in enumerate(body):
        fragments = [
            OcrFragment(text=header, kind="text", confidence=0.96, bbox=Rect(40, 30, 500, 40)),
            OcrFragment(text=FOOTER, kind="text", confidence=0.93, bbox=FOOTER_BBOX),
        ] + [
            OcrFragment(
                text=line,
                kind="text",
                confidence=0.95,
                bbox=Rect(40, 200 + index * 60, 600, 40),
            )
            for index, line in enumerate(lines)
        ]
        slides.append(
            SlideOcr(
                slide_index=position + 1,
                image_path=Path(f"slides/{position:03d}.png"),
                fragments=tuple(fragments),
                markdown="",
                backend="fake",
            )
        )

    decoration = find_decoration_lines(slides)
    assert "квадратные уравнения" not in decoration
    # Плашка платформы при этом отсекается по-прежнему.
    assert "логоплатформа онлайн школа" in decoration

    terms = " ".join(build_glossary(slides).terms).casefold()
    assert "уравнени" in terms and "квадратн" in terms
    assert "логоплатформа" not in terms


def test_математические_термины_попадают_в_глоссарий():
    terms = {term.casefold() for term in build_glossary(_slides()).terms}
    for expected in ("дискриминант", "теорема виета", "логарифма", "неравенства"):
        assert any(expected in term for term in terms), expected


def test_сквозная_служебная_надпись_не_попадает_в_глоссарий():
    glossary = build_glossary(_slides())
    joined = " ".join(glossary.terms).casefold()
    assert "логоплатформа" not in joined
    assert "онлайн-школа" not in joined


def test_надпись_на_всех_слайдах_в_одном_месте_признана_оформлением():
    decoration = find_decoration_lines(_slides())
    assert "логоплатформа онлайн школа" in decoration


def test_надпись_гуляющая_по_слайдам_оформлением_не_считается():
    slides = [
        _slide(1, CONTENT[0], footer_bbox=Rect(40, 560, 300, 30)),
        _slide(2, CONTENT[1], footer_bbox=Rect(40, 120, 300, 30)),
        _slide(3, CONTENT[2], footer_bbox=Rect(500, 300, 300, 30)),
        _slide(4, CONTENT[3], footer_bbox=Rect(40, 60, 300, 30)),
        _slide(5, CONTENT[4], footer_bbox=Rect(600, 500, 300, 30)),
    ]
    assert find_decoration_lines(slides) == set()


def test_на_паре_слайдов_эвристика_оформления_не_срабатывает():
    # Слишком мало слайдов, чтобы отличить оформление от содержания.
    slides = _slides()[:2]
    assert find_decoration_lines(slides) == set()


def test_дедупликация_словоформ():
    slides = [
        _slide(1, ["Уравнение и его корни"], footer=False),
        _slide(2, ["Уравнения и корней"], footer=False),
        _slide(3, ["Уравнения"], footer=False),
    ]
    terms = [term.casefold() for term in build_glossary(slides).terms]
    assert sum(1 for term in terms if term.startswith("уравнени")) == 1
    assert len(terms) == len(set(terms))


def test_нормализация_ключа_склеивает_словоформы():
    assert normalize_key("Уравнение") == normalize_key("уравнения") == normalize_key("УРАВНЕНИЙ")


def test_словоформы_с_беглой_гласной_склеиваются():
    # «корень» / «корни» / «корня» — одно слово, в глоссарии одна запись.
    assert normalize_key("корень") == normalize_key("корни") == normalize_key("корня")
    slides = [
        _slide(1, ["Корень уравнения"], footer=False),
        _slide(2, ["Корни уравнения"], footer=False),
        _slide(3, ["Корня не существует"], footer=False),
    ]
    terms = [term.casefold() for term in build_glossary(slides).terms]
    assert sum(1 for term in terms if term.startswith("корн") or term.startswith("корен")) == 1


def test_биграммы_из_капса_и_служебных_слов_не_берутся():
    assert extract_terms("КВАДРАТНЫЕ УРАВНЕНИЯ И ИХ КОРНИ") == [
        "КВАДРАТНЫЕ",
        "УРАВНЕНИЯ",
        "КОРНИ",
    ]
    assert "уравнение Которое" not in extract_terms("уравнение Которое имеет корни")
    # Имя собственное после нарицательного по-прежнему берётся.
    assert "теорема Виета" in extract_terms("теорема Виета для уравнения")


def test_запись_без_слайдов_даёт_пустой_глоссарий():
    glossary = build_glossary([])
    assert glossary.terms == ()
    assert not glossary


def test_формулы_и_неуверенные_фрагменты_в_глоссарий_не_идут():
    slide = SlideOcr(
        slide_index=1,
        image_path=Path("slides/001.png"),
        fragments=(
            OcrFragment(text=r"$\sqrt{x^2-x} = -5$", kind="formula", confidence=0.9),
            OcrFragment(text="Неразборчивое слово", kind="text", confidence=0.2, low_confidence=True),
            OcrFragment(text="Дискриминант", kind="text", confidence=0.97),
        ),
        markdown="",
    )
    terms = build_glossary([slide]).terms
    assert terms == ("Дискриминант",)


def test_ручной_стоп_лист_вырезает_надпись_из_глоссария():
    # Аварийный выход для случая, когда геометрическая эвристика ошиблась на
    # чужом оформлении: два слайда — эвристика не работает вовсе.
    slides = [_slide(1, CONTENT[0]), _slide(2, CONTENT[1])]
    default_terms = build_glossary(slides).terms
    assert any("ЛогоПлатформа" in term for term in default_terms)

    config = GlossaryConfig(stopwords=("ЛогоПлатформа Онлайн-школа",))
    terms = build_glossary(slides, config).terms
    joined = " ".join(terms).casefold()
    assert "логоплатформа" not in joined and "онлайн-школа" not in joined
    # Остальные термины лекции на месте.
    assert any("искриминант" in term for term in terms)


def test_стоп_лист_сравнивает_по_словоформам():
    slides = [_slide(1, ["Уравнение и его корни"], footer=False)]
    config = GlossaryConfig(stopwords=("уравнения",))
    terms = " ".join(build_glossary(slides, config).terms).casefold()
    assert "уравнени" not in terms
    assert "корни" in terms


def test_по_умолчанию_стоп_лист_пуст_и_ничего_не_меняет():
    assert DEFAULT_GLOSSARY.stopwords == ()
    slides = _slides()
    assert build_glossary(slides).terms == build_glossary(slides, GlossaryConfig()).terms
    assert build_glossary(slides, GlossaryConfig(stopwords=())).terms == build_glossary(slides).terms


def test_настройка_порога_доли_слайдов():
    slides = _slides()
    # Требуем присутствия на всех слайдах — надпись всё ещё оформление.
    strict = GlossaryConfig(decoration_slide_ratio=1.0)
    assert find_decoration_lines(slides, strict)
    # Надпись только на трёх слайдах из пяти — при строгом пороге не режется.
    partial = slides[:3] + [_slide(4, CONTENT[3], footer=False), _slide(5, CONTENT[4], footer=False)]
    assert find_decoration_lines(partial, strict) == set()
