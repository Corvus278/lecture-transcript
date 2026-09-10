"""Выделение прямоугольной области демонстрации по маске «статично и светло».

Маска из `variance` — это ещё не регион: у неё рваные края, внутри дырки от
маркерных штрихов, а снаружи разбросана мелочь (светлые статичные детали
оформления, аватарки-заглушки). Здесь маска приводится к одному
прямоугольнику:

1. морфологическое закрытие — сшивает дырки внутри слайда;
2. морфологическое открытие — сносит мелкие блобы снаружи;
3. крупнейшая связная компонента и её bounding box;
4. проверки правдоподобия — иначе `None` («демонстрации нет»);
5. расширение ядра до внешней границы статичной области (см. ниже).

Проверки правдоподобия намеренно грубые и не завязаны на платформу:
регион должен занимать заметную долю кадра, не быть узкой полоской и быть
достаточно плотно заполнен своей же компонентой (bounding box буквы «Г»
из двух полосок отсекается именно этим).

Почему найденного по маске прямоугольника мало
----------------------------------------------
Маска «статично И светло» выделяет самое светлое поле демонстрации — на
эталонной записи это белая карточка шаблона, а не весь показываемый экран:
рамка шаблона имеет яркость около 122 и в маску не попадает. Лектор же
регулярно дописывает решение маркером **поверх рамки**, за нижней и правой
границей карточки: на эталоне так потеряны ответы на t=1840, t=3613 и
t=4677. Терять содержимое — худшая из ошибок стадии (design D4/Risks),
поэтому найденный по маске прямоугольник считается только **ядром** и
расширяется до внешней границы статичной области (`expand_to_static_border`).

Расширение не знает ничего про цвета шаблона. Оно опирается на устройство
любой видеосетки: плитки разделены тёмными промежутками, а фон соседних
плиток заметно темнее содержимого демонстрации. Порог темноты берётся не
константой, а из самого кадра — между медианой яркости внутри ядра и
медианой яркости за его пределами. На эталоне это даёт 85 при рамке 122 и
фоне плиток 58, и расширенная область во всех окнах записи имеет
соотношение сторон 1.78 — то есть найден ровно транслируемый экран 16:9.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..contracts import Rect
from .variance import (
    DEFAULT_ANALYSIS_WIDTH,
    DEFAULT_MAX_SPREAD,
    DEFAULT_MIN_BRIGHTNESS,
    PixelStats,
    pixel_stats,
    static_bright_mask,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BORDER_LEVEL_RATIO",
    "DEFAULT_BORDER_LINE_RATIO",
    "DEFAULT_CLOSE_PX",
    "DEFAULT_MIN_AREA_RATIO",
    "DEFAULT_MIN_FILL_RATIO",
    "DEFAULT_MIN_SAMPLE_SIZE",
    "DEFAULT_MIN_SIDE_RATIO",
    "DEFAULT_OPEN_PX",
    "expand_to_static_border",
    "find_region",
    "iou",
    "region_from_mask",
    "region_from_stats",
    "union_rect",
]

#: Сторона квадратного ядра закрытия в сетке анализа, px. При ширине сетки
#: 480 это ~36 px исходного кадра — заведомо больше толщины маркерного
#: штриха и межстрочного промежутка, заведомо меньше зазора между
#: областью демонстрации и тайлами участников.
DEFAULT_CLOSE_PX = 9

#: Сторона ядра открытия в сетке анализа, px: сносит блобы мельче ~12 px
#: исходного кадра.
DEFAULT_OPEN_PX = 3

#: Минимальная доля площади кадра (совпадает с `min_region_area_ratio`
#: в конфиге). Область демонстрации эталона занимает 0.39 кадра, случайный
#: светлый статичный мусор — 0.002 и меньше.
DEFAULT_MIN_AREA_RATIO = 0.05

#: Минимальная доля bounding box, занятая самой компонентой. У настоящей
#: области демонстрации — 1.0, у случайного блоба — 0.5..0.75.
DEFAULT_MIN_FILL_RATIO = 0.8

#: Минимальная доля стороны кадра, которую занимает сторона региона:
#: отсекает длинные узкие полосы (баннер, строка статуса), проходящие по
#: площади, но не являющиеся областью демонстрации.
DEFAULT_MIN_SIDE_RATIO = 0.15

#: Где между «фоном за пределами ядра» и «содержимым ядра» проходит граница
#: темноты при расширении ядра до внешней границы демонстрации. 0.15 — то
#: есть линия считается ещё частью демонстрации, если она ярче фона на 15 %
#: расстояния до яркости ядра. На эталоне: фон плиток 58, ядро 235, порог
#: 85; рамка шаблона (122) проходит, фон плиток и тёмный промежуток между
#: плитками (0..61) — нет.
DEFAULT_BORDER_LEVEL_RATIO = 0.15

#: Какая доля пикселей линии должна быть «статичной и не фоном», чтобы
#: линия считалась ещё частью области демонстрации. Половина: маркерный
#: штрих или тёмный текст, пересекающий линию, не должен её отбраковывать.
DEFAULT_BORDER_LINE_RATIO = 0.5

#: Минимальный размер выборки, при котором признак «статично» осмыслен.
#: На одном кадре MAD равен нулю везде, и маска вырождается в «просто
#: светло» — область нашлась бы по любому светлому пятну.
DEFAULT_MIN_SAMPLE_SIZE = 3


def region_from_mask(
    mask: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    min_side_ratio: float = DEFAULT_MIN_SIDE_RATIO,
    close_px: int = DEFAULT_CLOSE_PX,
    open_px: int = DEFAULT_OPEN_PX,
) -> Rect | None:
    """Крупнейший правдоподобный прямоугольник маски в координатах кадра.

    :param mask: маска сетки анализа (0/1), форма (h, w).
    :param frame_width: ширина исходного кадра, px — задаёт масштаб.
    :param frame_height: высота исходного кадра, px.
    :returns: `Rect` в координатах исходного кадра либо `None`, если ни одна
        компонента не тянет на область демонстрации.
    """
    if mask.ndim != 2:
        raise ValueError(f"маска должна быть двумерной, получена форма {mask.shape}")
    binary = (mask > 0).astype(np.uint8)
    if close_px > 1:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8)
        )
    if open_px > 1:
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8)
        )

    grid_h, grid_w = binary.shape
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        logger.warning(
            "область демонстрации не найдена: маска пуста после морфологии"
            " (доля маски до морфологии %.4f)",
            float((mask > 0).mean()),
        )
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in stats[largest])
    if w <= 0 or h <= 0:
        logger.warning("область демонстрации не найдена: вырожденная компонента")
        return None

    box_area = w * h
    fill = area / box_area
    area_ratio = box_area / (grid_h * grid_w)
    side_w, side_h = w / grid_w, h / grid_h
    if fill < min_fill_ratio:
        logger.warning(
            "область демонстрации не найдена: компонента рыхлая, заполнение"
            " bounding box %.2f < %.2f (компонент %d, площадь %.3f кадра)",
            fill,
            min_fill_ratio,
            count - 1,
            area_ratio,
        )
        return None
    if area_ratio < min_area_ratio:
        logger.warning(
            "область демонстрации не найдена: регион мелкий, %.4f площади кадра"
            " < %.4f (компонент %d)",
            area_ratio,
            min_area_ratio,
            count - 1,
        )
        return None
    if side_w < min_side_ratio or side_h < min_side_ratio:
        logger.warning(
            "область демонстрации не найдена: узкая полоса, стороны %.2f×%.2f"
            " кадра при минимуме %.2f",
            side_w,
            side_h,
            min_side_ratio,
        )
        return None

    scale_x = frame_width / grid_w
    scale_y = frame_height / grid_h
    rect = Rect(
        x=int(round(x * scale_x)),
        y=int(round(y * scale_y)),
        width=int(round(w * scale_x)),
        height=int(round(h * scale_y)),
    )
    return _clamp(rect, frame_width, frame_height)


def expand_to_static_border(
    stats: PixelStats,
    core: Rect,
    *,
    max_spread: float = DEFAULT_MAX_SPREAD,
    level_ratio: float = DEFAULT_BORDER_LEVEL_RATIO,
    line_ratio: float = DEFAULT_BORDER_LINE_RATIO,
) -> Rect:
    """Расширить ядро области до внешней границы статичной демонстрации.

    Ядро — самое светлое поле демонстрации; вокруг него у шаблона обычно
    есть рамка средней яркости, поверх которой лектор пишет маркером.
    Прямоугольник растится построчно во все четыре стороны, пока
    очередная линия остаётся «статичной и не фоном», и останавливается на
    тёмном промежутке между плитками видеосетки либо на фоне соседней
    плитки. Порог темноты берётся из самого кадра, поэтому правило не
    зависит от цвета оформления платформы (design D3).

    :param core: прямоугольник ядра в координатах исходного кадра.
    :returns: расширенный прямоугольник в координатах исходного кадра;
        при отсутствии рамки совпадает с ядром.
    """
    brightness, spread = stats.brightness, stats.spread
    grid_h, grid_w = brightness.shape
    scale_x = stats.frame_width / grid_w
    scale_y = stats.frame_height / grid_h

    x0 = max(0, min(grid_w - 1, int(round(core.x / scale_x))))
    y0 = max(0, min(grid_h - 1, int(round(core.y / scale_y))))
    x1 = max(x0 + 1, min(grid_w, int(round((core.x + core.width) / scale_x))))
    y1 = max(y0 + 1, min(grid_h, int(round((core.y + core.height) / scale_y))))

    inside = np.zeros(brightness.shape, dtype=bool)
    inside[y0:y1, x0:x1] = True
    outside = ~inside
    if not outside.any():
        return core
    core_level = float(np.median(brightness[inside]))
    background = float(np.median(brightness[outside]))
    if core_level <= background:
        # Ядро не светлее окружения — расширять не по чему.
        return core
    stop_level = background + level_ratio * (core_level - background)
    belongs = (spread <= max_spread) & (brightness >= stop_level)

    def line_fits(line: np.ndarray) -> bool:
        return bool(line.size) and float(line.mean()) >= line_ratio

    while x0 > 0 and line_fits(belongs[y0:y1, x0 - 1]):
        x0 -= 1
    while x1 < grid_w and line_fits(belongs[y0:y1, x1]):
        x1 += 1
    while y0 > 0 and line_fits(belongs[y0 - 1, x0:x1]):
        y0 -= 1
    while y1 < grid_h and line_fits(belongs[y1, x0:x1]):
        y1 += 1

    expanded = Rect(
        x=int(round(x0 * scale_x)),
        y=int(round(y0 * scale_y)),
        width=int(round((x1 - x0) * scale_x)),
        height=int(round((y1 - y0) * scale_y)),
    )
    return _clamp(expanded, stats.frame_width, stats.frame_height)


def region_from_stats(
    stats: PixelStats,
    *,
    max_spread: float = DEFAULT_MAX_SPREAD,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    min_side_ratio: float = DEFAULT_MIN_SIDE_RATIO,
    close_px: int = DEFAULT_CLOSE_PX,
    open_px: int = DEFAULT_OPEN_PX,
    expand_border: bool = True,
    border_level_ratio: float = DEFAULT_BORDER_LEVEL_RATIO,
    border_line_ratio: float = DEFAULT_BORDER_LINE_RATIO,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
) -> Rect | None:
    """Область демонстрации по готовым картам статистик."""
    if stats.sample_size < min_sample_size:
        logger.warning(
            "область демонстрации не искалась: в выборке %d кадр(ов) при"
            " минимуме %d — признак «статично» на такой выборке вырожден",
            stats.sample_size,
            min_sample_size,
        )
        return None
    mask = static_bright_mask(
        stats, max_spread=max_spread, min_brightness=min_brightness
    )
    if not mask.any():
        logger.warning(
            "область демонстрации не найдена: маска «статично и светло» пуста."
            " Яркость: медиана %.0f, 90-й процентиль %.0f при пороге %.0f;"
            " разброс: медиана %.1f, 10-й процентиль %.1f при пороге %.1f",
            float(np.median(stats.brightness)),
            float(np.percentile(stats.brightness, 90)),
            min_brightness,
            float(np.median(stats.spread)),
            float(np.percentile(stats.spread, 10)),
            max_spread,
        )
        return None
    core = region_from_mask(
        mask,
        frame_width=stats.frame_width,
        frame_height=stats.frame_height,
        min_area_ratio=min_area_ratio,
        min_fill_ratio=min_fill_ratio,
        min_side_ratio=min_side_ratio,
        close_px=close_px,
        open_px=open_px,
    )
    if core is None or not expand_border:
        return core
    return expand_to_static_border(
        stats,
        core,
        max_spread=max_spread,
        level_ratio=border_level_ratio,
        line_ratio=border_line_ratio,
    )


def find_region(
    paths: Sequence[Path],
    *,
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH,
    max_spread: float = DEFAULT_MAX_SPREAD,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    min_side_ratio: float = DEFAULT_MIN_SIDE_RATIO,
    close_px: int = DEFAULT_CLOSE_PX,
    open_px: int = DEFAULT_OPEN_PX,
    expand_border: bool = True,
    border_level_ratio: float = DEFAULT_BORDER_LEVEL_RATIO,
    border_line_ratio: float = DEFAULT_BORDER_LINE_RATIO,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
) -> Rect | None:
    """Область демонстрации по выборке кадров: статистики + маска + регион."""
    stats = pixel_stats(paths, analysis_width=analysis_width)
    return region_from_stats(
        stats,
        max_spread=max_spread,
        min_brightness=min_brightness,
        min_area_ratio=min_area_ratio,
        min_fill_ratio=min_fill_ratio,
        min_side_ratio=min_side_ratio,
        close_px=close_px,
        open_px=open_px,
        expand_border=expand_border,
        border_level_ratio=border_level_ratio,
        border_line_ratio=border_line_ratio,
        min_sample_size=min_sample_size,
    )


def iou(first: Rect, second: Rect) -> float:
    """Отношение площади пересечения к площади объединения двух прямоугольников."""
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = first.area + second.area - inter
    return inter / union if union else 0.0


def union_rect(rects: Sequence[Rect]) -> Rect | None:
    """Наименьший прямоугольник, накрывающий все переданные.

    Нужен для склейки областей соседних окон одной раскладки: шаблон
    оформления у разных слайдов оставляет светлым чуть разное поле, и по
    отдельному окну область получается по пересечению светлых полей, то
    есть слишком тесной. Кроп для OCR обязан не терять содержимого, а
    лишние пиксели рамки OCR не мешают, поэтому берётся объединение.
    """
    boxes = [r for r in rects if r is not None]
    if not boxes:
        return None
    left = min(r.x for r in boxes)
    top = min(r.y for r in boxes)
    right = max(r.x + r.width for r in boxes)
    bottom = max(r.y + r.height for r in boxes)
    return Rect(x=left, y=top, width=right - left, height=bottom - top)


def _clamp(rect: Rect, frame_width: int, frame_height: int) -> Rect:
    """Обрезать прямоугольник по границам кадра."""
    x = max(0, min(rect.x, frame_width - 1))
    y = max(0, min(rect.y, frame_height - 1))
    width = max(1, min(rect.width, frame_width - x))
    height = max(1, min(rect.height, frame_height - y))
    return Rect(x=x, y=y, width=width, height=height)
