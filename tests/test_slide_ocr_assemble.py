"""Сборка слайда в Markdown и маркировка неуверенного (задачи 4.5, 4.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pytest

from lecture_transcript.contracts import Availability, OcrFragment, Rect, Slide
from lecture_transcript.slide_ocr.hybrid_backend import HybridBackend
from lecture_transcript.slide_ocr.assemble import (
    AssembleConfig,
    LOW_CONFIDENCE_MARK,
    assemble_slide_ocr,
    is_unreliable,
    recognize_slide,
    sort_fragments,
)


def _fragment(text: str, y: int = 0, confidence: float = 0.95, kind: str = "text") -> OcrFragment:
    return OcrFragment(
        text=text,
        kind=kind,
        confidence=confidence,
        bbox=Rect(x=10, y=y, width=200, height=40),
    )


# --------------------------------------------------------------------------
# 4.5 — порядок строк и обязательная ссылка на PNG
# --------------------------------------------------------------------------


def test_ссылка_на_png_есть_для_пустого_слайда():
    result = assemble_slide_ocr(12, Path("slides/012.png"), fragments=())
    assert "![слайд 12](slides/012.png)" in result.markdown
    assert result.fragments == ()
    assert result.unreliable is False  # пустой слайд ошибкой не считается


def test_ссылка_на_png_есть_для_непустого_слайда():
    result = assemble_slide_ocr(3, Path("slides/003.png"), [_fragment("Пример 2.")])
    assert result.markdown.rstrip().endswith("![слайд 3](slides/003.png)")
    assert "Пример 2." in result.markdown


def test_порядок_строк_сверху_вниз():
    fragments = [
        _fragment("нижняя", y=300),
        _fragment("верхняя", y=10),
        _fragment("средняя", y=150),
    ]
    ordered = [fragment.text for fragment in sort_fragments(fragments)]
    assert ordered == ["верхняя", "средняя", "нижняя"]

    markdown = assemble_slide_ocr(1, Path("s.png"), fragments).markdown
    assert markdown.index("верхняя") < markdown.index("средняя") < markdown.index("нижняя")


def _boxed(text: str, x: int, y: int, width: int = 200, height: int = 40) -> OcrFragment:
    return OcrFragment(
        text=text, kind="text", confidence=0.95, bbox=Rect(x=x, y=y, width=width, height=height)
    )


def test_две_строки_на_одной_линии_идут_слева_направо_несмотря_на_джиттер():
    # Боксы PaddleOCR дрожат по вертикали на 1-3 px; без группировки в
    # визуальные строки правая колонка обгоняла бы левую.
    fragments = [
        _boxed("ПРАВАЯ", x=500, y=101),
        _boxed("ЛЕВАЯ", x=20, y=103),
        _boxed("СЛЕДУЮЩАЯ", x=20, y=200),
    ]
    assert [f.text for f in sort_fragments(fragments)] == ["ЛЕВАЯ", "ПРАВАЯ", "СЛЕДУЮЩАЯ"]


def test_строка_с_текстом_и_формулой_идёт_в_порядке_слайда():
    # Ровно случай из design D5: «Пример 2.  √(x²−x) = −5» — два бокса на
    # одной визуальной строке, у которых y отличается на пару пикселей.
    formula = OcrFragment(
        text=r"$\sqrt{x^2-x} = -5$",
        kind="formula",
        confidence=0.85,
        bbox=Rect(x=260, y=58, width=300, height=40),
    )
    fragments = [formula, _boxed("Пример 2.", x=20, y=60)]
    markdown = assemble_slide_ocr(12, Path("slides/012.png"), fragments).markdown
    assert markdown.index("Пример 2.") < markdown.index(r"$\sqrt{x^2-x} = -5$")


def test_разные_визуальные_строки_не_склеиваются():
    # Соседние строки слайда: центры расходятся сильнее допуска, порядок
    # остаётся сверху вниз даже при обратном порядке по x.
    fragments = [
        _boxed("вторая", x=20, y=120),
        _boxed("первая", x=500, y=60),
    ]
    assert [f.text for f in sort_fragments(fragments)] == ["первая", "вторая"]


def test_фрагменты_без_боксов_сохраняют_порядок_бэкенда():
    fragments = [
        OcrFragment(text="раз", kind="text", confidence=0.9),
        OcrFragment(text="два", kind="text", confidence=0.9),
        OcrFragment(text="три", kind="text", confidence=0.9),
    ]
    assert [f.text for f in sort_fragments(fragments)] == ["раз", "два", "три"]


def test_формула_попадает_в_markdown_как_latex():
    fragments = [
        _fragment("Пример 2.", y=10),
        _fragment(r"$\sqrt{x^2-x} = -5$", y=80, kind="formula"),
    ]
    markdown = assemble_slide_ocr(12, Path("slides/012.png"), fragments).markdown
    assert markdown.index("Пример 2.") < markdown.index(r"$\sqrt{x^2-x} = -5$")
    assert "![слайд 12]" in markdown


def test_разметка_из_текста_слайда_экранируется():
    # Стадия сама вводит соглашение «$...$ — это LaTeX»: доллар в печатном
    # тексте слайда обязан остаться долларом, а не открыть формулу.
    fragments = [
        _boxed("Цена: 100$ за 5$ штуки", x=10, y=10),
        _boxed("# не заголовок", x=10, y=80),
        _boxed("Функция f_1 и f_2 при 5*3 = 15", x=10, y=150),
        _boxed("a | b | c", x=10, y=220),
    ]
    markdown = assemble_slide_ocr(1, Path("s.png"), fragments).markdown
    assert "100$ за" not in markdown and r"100\$ за" in markdown
    assert not any(line.startswith("# ") for line in markdown.splitlines())
    assert r"f\_1" in markdown and r"5\*3" in markdown
    assert r"a \| b \| c" in markdown


def test_формула_в_markdown_не_экранируется():
    formula = OcrFragment(text=r"$\sqrt{x^2-x} = -5$", kind="formula", confidence=0.85)
    markdown = assemble_slide_ocr(1, Path("s.png"), [formula]).markdown
    assert r"$\sqrt{x^2-x} = -5$" in markdown


def test_строки_слайда_не_склеиваются_в_один_абзац():
    # Одиночный перевод строки в CommonMark — soft break, структура строк
    # слайда при рендере схлопнулась бы в абзац.
    fragments = [_boxed("Строка один", x=10, y=10), _boxed("Строка два", x=10, y=80)]
    markdown = assemble_slide_ocr(1, Path("s.png"), fragments).markdown
    assert "Строка один\n\nСтрока два" in markdown


def test_ссылка_на_png_работает_при_пробелах_и_скобках_в_пути():
    markdown = assemble_slide_ocr(
        1, Path("/tmp/Лекция 3 (запись)/slides/012.png"), fragments=()
    ).markdown
    link = markdown.strip()
    assert link.startswith("![слайд 1](") and link.endswith(".png)")
    inside = link[len("![слайд 1](") : -1]
    assert " " not in inside and "(" not in inside and ")" not in inside
    assert "%20" in inside and "%28" in inside


# --------------------------------------------------------------------------
# 4.6 — маркировка неуверенных фрагментов и ненадёжного слайда
# --------------------------------------------------------------------------


def test_правило_ненадёжного_слайда():
    config = AssembleConfig()
    low = [_fragment("a", confidence=0.1), _fragment("b", confidence=0.2)]
    high = [_fragment("a", confidence=0.9), _fragment("b", confidence=0.95)]
    mixed_ok = [_fragment("a", confidence=0.9), _fragment("b", confidence=0.95), _fragment("c", confidence=0.1)]

    assert is_unreliable(assemble_slide_ocr(1, Path("s.png"), low, config=config).fragments) is True
    assert is_unreliable(assemble_slide_ocr(1, Path("s.png"), high, config=config).fragments) is False
    assert is_unreliable(assemble_slide_ocr(1, Path("s.png"), mixed_ok, config=config).fragments) is False
    assert is_unreliable((), config) is False


class _SharpnessBackend:
    """Фейковый бэкенд: уверенность падает вместе с резкостью изображения.

    Нужен, чтобы проверить ЛОГИКУ маркировки на реально размытом кропе,
    не поднимая настоящие веса моделей.
    """

    name = "sharpness-fake"

    def check_availability(self) -> Availability:
        return Availability(True)

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
        confidence = max(0.0, min(1.0, sharpness / 300.0))
        return (
            OcrFragment(
                text="Как называются эти числа",
                kind="text",
                confidence=confidence,
                bbox=Rect(0, 0, image.shape[1], image.shape[0]),
            ),
        )

    def unload(self) -> None:
        return None


def _crop_image(tmp_path: Path, blur: bool) -> Path:
    image = np.full((200, 800), 255, np.uint8)
    cv2.putText(image, "Kak nazyvayutsya eti chisla", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    if blur:
        image = cv2.GaussianBlur(image, (31, 31), 0)
    path = tmp_path / ("blurred.png" if blur else "sharp.png")
    cv2.imwrite(str(path), image)
    return path


def _slide(path: Path) -> Slide:
    return Slide(
        index=7,
        start_s=0.0,
        end_s=10.0,
        region=Rect(0, 0, 800, 200),
        representative_timestamp_s=5.0,
        image_path=path,
    )


def test_размытый_кроп_помечается_ненадёжным(tmp_path: Path):
    backend = _SharpnessBackend()

    sharp = recognize_slide(_slide(_crop_image(tmp_path, blur=False)), backend)
    blurred = recognize_slide(_slide(_crop_image(tmp_path, blur=True)), backend)

    assert sharp.unreliable is False
    assert sharp.fragments[0].low_confidence is False

    assert blurred.unreliable is True
    assert blurred.fragments[0].low_confidence is True
    assert LOW_CONFIDENCE_MARK in blurred.markdown
    # Ссылка на изображение сохраняется и у ненадёжного слайда.
    assert f"![слайд 7]({blurred.image_path.as_posix()})" in blurred.markdown


class _NeverCalledFormulaBackend:
    name = "formula-fake"

    def check_availability(self) -> Availability:
        return Availability(True)

    def latex_from_image(self, image) -> str:  # pragma: no cover — не должен вызываться
        raise AssertionError("в формульный распознаватель ничего не должно уйти")

    def unload(self) -> None:
        return None


def test_размытый_кроп_через_гибрид_помечается_ненадёжным(tmp_path: Path):
    # Через гибрид строки с уверенностью ниже порога рукописного отбрасываются;
    # слайд без единого фрагмента не должен выглядеть честно пустым.
    backend = HybridBackend(_SharpnessBackend(), _NeverCalledFormulaBackend())

    sharp = recognize_slide(_slide(_crop_image(tmp_path, blur=False)), backend)
    blurred = recognize_slide(_slide(_crop_image(tmp_path, blur=True)), backend)

    assert sharp.unreliable is False
    assert blurred.fragments == ()
    assert blurred.unreliable is True
    # Видимую пометку рисует рендер транскрипта по флагу, не стадия:
    # иначе на ненадёжном слайде она выходит дважды.
    assert "неуверенно" not in blurred.markdown.casefold()
    assert f"![слайд 7]({blurred.image_path.as_posix()})" in blurred.markdown


def test_бэкенд_выставил_флаг_вручную_и_он_не_снимается():
    fragment = OcrFragment(text="хм", kind="text", confidence=0.99, low_confidence=True)
    result = assemble_slide_ocr(1, Path("s.png"), [fragment])
    assert result.fragments[0].low_confidence is True
    assert result.unreliable is True
