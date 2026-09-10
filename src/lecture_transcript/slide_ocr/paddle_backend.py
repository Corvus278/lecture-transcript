"""Бэкенд распознавания печатного текста слайда — PaddleOCR ru (задача 4.2).

Тяжёлые импорты (`paddleocr`, `paddle`) ленивые: модуль импортируется в
окружении без установленного PaddleOCR, а `check_availability()` в таком
окружении возвращает `Availability(False, ...)` с инструкцией по установке.

Разбор выдачи PaddleOCR вынесен в чистую функцию `parse_paddle_output()`,
чтобы её можно было тестировать без весов.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Sequence

from ..contracts import Availability, OcrFragment, Rect

#: Уверенность, ниже которой строка сразу помечается неуверенной.
DEFAULT_LOW_CONFIDENCE = 0.60

_INSTALL_HINT = (
    "установите PaddleOCR: pip install paddlepaddle paddleocr "
    "(на macOS arm64 — CPU-колесо paddlepaddle), затем скачайте веса "
    "русской модели один раз при доступной сети"
)


def local_weights_dirs() -> tuple[Path, ...]:
    """Каталоги, где лежат скачанные веса PaddleOCR.

    PaddleOCR 3.x качает модели через PaddleX в `<CACHE_DIR>/official_models`,
    где `CACHE_DIR = os.environ.get("PADDLE_PDX_CACHE_HOME", "~/.paddlex")`
    (сверено по колесу paddlex 3.2.0: `utils/cache.py`,
    `inference/utils/official_models.py`). Образ Docker задаёт
    `PADDLE_PDX_CACHE_HOME=/cache/paddlex`. Смотрим ровно туда, куда смотрит
    сам PaddleX: при заданной переменной `~/.paddlex` он не читает. Служебные
    `func_ret/`, `locks/`, которые PaddleX создаёт в корне кэша, весами не
    считаются. `~/.paddleocr` — каталог весов PaddleOCR 2.x.
    """
    home = Path.home()
    pdx_cache = os.environ.get("PADDLE_PDX_CACHE_HOME")
    pdx_root = Path(pdx_cache) if pdx_cache else home / ".paddlex"
    return (pdx_root / "official_models", home / ".paddleocr")


def is_paddleocr_3(engine_cls: Any) -> bool:
    """PaddleOCR 3.x отличается от 2.x наличием `predict` у класса."""
    return callable(getattr(engine_cls, "predict", None))


def paddleocr_init_kwargs(engine_cls: Any, lang: str) -> dict[str, Any]:
    """Аргументы конструктора `PaddleOCR` под установленную версию.

    3.x (сверено по колесу paddleocr 3.2.0, `_pipelines/ocr.py` и
    `_common_args.py`): неизвестный аргумент `show_log` уходит в
    `parse_common_args` и роняет конструктор `ValueError: Unknown argument`;
    `use_angle_cls` там лишь устаревшее имя `use_textline_orientation`.
    2.x принимает прежний набор.
    """
    if is_paddleocr_3(engine_cls):
        return {"lang": lang, "use_textline_orientation": True}
    return {"lang": lang, "use_angle_cls": True, "show_log": False}


def has_local_weights() -> bool:
    """Есть ли локальные веса — проверка без сети (задача 4.9)."""
    for path in local_weights_dirs():
        try:
            if path.is_dir() and any(path.iterdir()):
                return True
        except OSError:  # pragma: no cover — недоступный каталог
            continue
    return False


def _quad_to_rect(quad: Sequence[Sequence[float]]) -> Rect | None:
    """Четырёхугольник PaddleOCR -> охватывающий прямоугольник."""
    try:
        xs = [float(point[0]) for point in quad]
        ys = [float(point[1]) for point in quad]
    except (TypeError, IndexError, ValueError):
        return None
    if not xs or not ys:
        return None
    x0, y0 = int(round(min(xs))), int(round(min(ys)))
    x1, y1 = int(round(max(xs))), int(round(max(ys)))
    return Rect(x=x0, y=y0, width=max(0, x1 - x0), height=max(0, y1 - y0))


def parse_paddle_output(raw: Any) -> list[OcrFragment]:
    """Привести выдачу PaddleOCR к фрагментам контракта.

    Поддерживаются оба формата:
    * классический — `[[[quad, (text, score)], ...]]` (PaddleOCR 2.x);
    * словарный — `{"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}`
      (PaddleOCR 3.x / PaddleX), в том числе завёрнутый в список.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        return _parse_dict_output(raw)
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], dict):
            fragments: list[OcrFragment] = []
            for page in raw:
                fragments.extend(_parse_dict_output(page))
            return fragments
        return _parse_classic_output(raw)
    return []


def _field(payload: dict, *names: str) -> Any:
    """Значение первого присутствующего ключа выдачи.

    Присутствие проверяется через `is None`, а не через истинность: PaddleX
    отдаёт `rec_polys`/`rec_scores` как `numpy.ndarray`, и `bool(ndarray)` с
    более чем одним элементом бросает `ValueError`.
    """
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return []


def _parse_dict_output(page: dict) -> list[OcrFragment]:
    nested = page.get("res", None)
    payload = nested if isinstance(nested, dict) else page
    texts = _field(payload, "rec_texts")
    scores = _field(payload, "rec_scores")
    polys = _field(payload, "rec_polys", "dt_polys")
    fragments: list[OcrFragment] = []
    for index, text in enumerate(texts):
        if not str(text).strip():
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        bbox = _quad_to_rect(polys[index]) if index < len(polys) else None
        fragments.append(
            OcrFragment(
                text=str(text).strip(),
                kind="text",
                confidence=score,
                bbox=bbox,
                low_confidence=score < DEFAULT_LOW_CONFIDENCE,
            )
        )
    return fragments


def _parse_classic_output(raw: Sequence[Any]) -> list[OcrFragment]:
    pages = raw
    # PaddleOCR 2.x оборачивает страницы в список: [[line, line, ...]].
    if pages and isinstance(pages[0], (list, tuple)) and pages[0] and isinstance(pages[0][0], (list, tuple)):
        first = pages[0][0]
        if len(first) == 2 and isinstance(first[0], (list, tuple)) and first[0] and isinstance(first[0][0], (list, tuple)):
            pass  # уже страницы
        else:
            pages = [pages]
    fragments: list[OcrFragment] = []
    for page in pages:
        if not page:
            continue
        for line in page:
            if not line or len(line) < 2:
                continue
            quad, payload = line[0], line[1]
            if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                text, score = str(payload[0]), float(payload[1])
            else:
                text, score = str(payload), 0.0
            if not text.strip():
                continue
            fragments.append(
                OcrFragment(
                    text=text.strip(),
                    kind="text",
                    confidence=score,
                    bbox=_quad_to_rect(quad),
                    low_confidence=score < DEFAULT_LOW_CONFIDENCE,
                )
            )
    return fragments


class PaddleTextBackend:
    """Печатный текст слайда: боксы строк + confidence."""

    name = "paddle"

    def __init__(self, lang: str = "ru", use_gpu: bool = False) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self._engine: Any | None = None

    def check_availability(self) -> Availability:
        """Проверка до прогона: установлен ли PaddleOCR и скачаны ли веса.

        Веса проверяются отдельно от библиотеки: офлайн-режим (задача 4.9)
        не даст скачать их на ходу, и без этой проверки прогон упал бы уже
        в середине распознавания, а не до его начала.
        """
        try:
            for module in ("paddle", "paddleocr"):
                if importlib.util.find_spec(module) is None:
                    return Availability(
                        False,
                        f"библиотека {module!r} не установлена — {_INSTALL_HINT}",
                    )
        except Exception as exc:  # битая установка: find_spec бросает наружу
            return Availability(False, f"установка PaddleOCR повреждена: {exc}")
        if not has_local_weights():
            listed = ", ".join(str(path) for path in local_weights_dirs())
            return Availability(
                False,
                "веса PaddleOCR не найдены локально "
                f"(искали в {listed}) — {_INSTALL_HINT}",
            )
        return Availability(True)

    def _load(self) -> Any:
        if self._engine is None:
            from .offline import enforce_offline

            enforce_offline()
            from paddleocr import PaddleOCR  # ленивый импорт

            self._engine = PaddleOCR(**paddleocr_init_kwargs(PaddleOCR, self.lang))
        return self._engine

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        """Распознать печатный текст: по строке на фрагмент."""
        engine = self._load()
        if is_paddleocr_3(type(engine)):
            # В 3.x `ocr()` — устаревшая обёртка над `predict(img, **kwargs)`,
            # и прежний `cls=True` она уже не принимает.
            raw = engine.predict(str(image_path))
        else:  # PaddleOCR 2.x
            raw = engine.ocr(str(image_path), cls=True)
        return parse_paddle_output(raw)

    def unload(self) -> None:
        """Выгрузить модель (design D7)."""
        self._engine = None
