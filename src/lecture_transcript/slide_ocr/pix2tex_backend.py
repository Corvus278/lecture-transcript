"""Бэкенд распознавания печатных формул в LaTeX — pix2tex (задача 4.4).

Импорт `pix2tex` ленивый. Модель не даёт собственной оценки уверенности,
поэтому формульным фрагментам присваивается `DEFAULT_FORMULA_CONFIDENCE`,
ограниченная сверху уверенностью исходной строки текстового OCR
(см. `hybrid_backend`): размытая строка не может стать «уверенной формулой».
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Sequence

from ..contracts import Availability, OcrFragment, Rect

#: Уверенность формульного фрагмента по умолчанию — pix2tex своей не даёт
#: (design D5: точность на печатных формулах ~85%).
DEFAULT_FORMULA_CONFIDENCE = 0.85

_INSTALL_HINT = (
    "установите pix2tex: pip install 'pix2tex[gui]' или pip install pix2tex "
    "(веса ~100 МБ скачиваются один раз при доступной сети)"
)


def local_weights_dirs() -> tuple[Path, ...]:
    """Каталоги, куда pix2tex кладёт свои чекпойнты (внутрь пакета)."""
    try:
        spec = importlib.util.find_spec("pix2tex")
    except Exception:  # pragma: no cover — битая установка
        return ()
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    return tuple(Path(location) / "model" / "checkpoints" for location in locations)


def has_local_weights() -> bool:
    """Есть ли локальные веса — проверка без сети (задача 4.9)."""
    for path in local_weights_dirs():
        try:
            if path.is_dir() and any(path.glob("*.pth")):
                return True
        except OSError:  # pragma: no cover — недоступный каталог
            continue
    return False


def latex_to_markdown(latex: str) -> str:
    """LaTeX -> Markdown-фрагмент `$...$` согласно контракту `OcrFragment`."""
    text = (latex or "").strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = re.sub(r"^\\\[|\\\]$", "", text).strip()
    text = re.sub(r"^\\\(|\\\)$", "", text).strip()
    text = " ".join(text.split())
    if not text:
        return ""
    return f"${text}$"


class Pix2TexBackend:
    """Печатные формулы -> LaTeX."""

    name = "pix2tex"

    def __init__(self) -> None:
        self._model: Any | None = None

    def check_availability(self) -> Availability:
        """Проверка до прогона: установлен ли pix2tex и лежат ли веса рядом."""
        try:
            if importlib.util.find_spec("pix2tex") is None:
                return Availability(False, f"библиотека 'pix2tex' не установлена — {_INSTALL_HINT}")
            if importlib.util.find_spec("torch") is None:
                return Availability(False, "библиотека 'torch' не установлена — нужна для pix2tex")
        except Exception as exc:  # битая установка: find_spec бросает наружу
            return Availability(False, f"установка pix2tex повреждена: {exc}")
        if not has_local_weights():
            return Availability(
                False,
                "веса pix2tex не найдены локально — скачайте их один раз "
                f"при доступной сети: {_INSTALL_HINT}",
            )
        return Availability(True)

    def _load(self) -> Any:
        if self._model is None:
            from .offline import enforce_offline

            enforce_offline()
            from pix2tex.cli import LatexOCR  # ленивый импорт

            self._model = LatexOCR()
        return self._model

    def latex_from_image(self, image: Any) -> str:
        """LaTeX по PIL-изображению (кроп строки или весь слайд)."""
        model = self._load()
        return str(model(image))

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        """Распознать изображение целиком как одну формулу."""
        from PIL import Image  # входит в зависимости проекта

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            latex = self.latex_from_image(image)
        markdown = latex_to_markdown(latex)
        if not markdown:
            return ()
        return (
            OcrFragment(
                text=markdown,
                kind="formula",
                confidence=DEFAULT_FORMULA_CONFIDENCE,
                bbox=Rect(0, 0, image.width, image.height),
            ),
        )

    def unload(self) -> None:
        """Выгрузить модель (design D7)."""
        self._model = None
