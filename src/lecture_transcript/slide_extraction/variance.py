"""Пер-пиксельная статистика выборки кадров: разброс во времени и яркость.

Признак области демонстрации, общий для всех платформ видеосвязи (design D3):
она **статична во времени и светлая**, тогда как видеотайлы участников
подвижны и темны. Модуль считает по выборке кадров окна две карты — карту
временно́го разброса и карту яркости — и строит по ним маску
«низкий разброс И высокая яркость».

Почему разброс считается робастно
---------------------------------
D3 говорит о дисперсии. В буквальном виде она здесь ломается: на эталонной
платформе переключение слайда сопровождается кратким (~4 с) исчезновением
демонстрации. Четыре кадра из трёхсот с яркостью ~58 вместо ~235 поднимают
обычную σ области слайда примерно до 20 — маска рассыпается ровно там, где
она нужна. Поэтому маска строится по робастной оценке σ через медианное
абсолютное отклонение (MAD × 1.4826): она не замечает выбросов, пока их
доля меньше ~25 % выборки. Обычные среднее и дисперсия тоже считаются и
лежат в `PixelStats` — для диагностики и проверок.

Почему анализ идёт в уменьшенном разрешении
-------------------------------------------
Область демонстрации — объект в сотни пикселей, её границы не нужны точнее
одного шага сетки анализа. Уменьшение до 480 px по ширине ускоряет разбор
91-минутной записи примерно в 16 раз (JPEG декодируется сразу в
уменьшенном виде, через `IMREAD_REDUCED_*`) и заодно гасит шум сжатия.
Цена — округление границ региона до шага сетки (4 px при 1920 -> 480).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..contracts import Frame, PipelineError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ANALYSIS_WIDTH",
    "DEFAULT_MAX_SPREAD",
    "DEFAULT_MIN_BRIGHTNESS",
    "MAD_TO_SIGMA",
    "PixelStats",
    "dominant_frame_size",
    "frame_size",
    "load_gray",
    "load_stack",
    "pixel_stats",
    "sample_frames",
    "static_bright_mask",
]

#: Ширина, к которой приводятся кадры перед анализом, px.
DEFAULT_ANALYSIS_WIDTH = 480

#: Верхняя граница робастной σ яркости пикселя, при которой он считается
#: статичным. Обоснование по эталонной записи: внутри области демонстрации
#: 99-й процентиль σ — около 1.2 (неподвижный слайд) и около 28 там, где
#: поверх слайда пишут маркером; снаружи, по видеотайлам, медиана σ — 23 и
#: выше. Порог 8 отделяет фон слайда от тайлов с запасом в обе стороны, а
#: дырки от маркерных штрихов закрывает морфология в `region`.
DEFAULT_MAX_SPREAD = 8.0

#: Нижняя граница медианной яркости пикселя (0..255), при которой он
#: считается светлым. По эталонной записи: внутри области демонстрации
#: медиана яркости — 235, снаружи 90-й процентиль — 129. Порог 150 лежит
#: посередине. Он задаёт только **ядро** области — самое светлое поле;
#: рамку шаблона (≈122) добирает `region.expand_to_static_border`.
DEFAULT_MIN_BRIGHTNESS = 150.0

#: Множитель, приводящий MAD нормального распределения к масштабу σ.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class PixelStats:
    """Карты статистик по выборке кадров, в сетке анализа.

    Все карты имеют одинаковую форму (h, w) в координатах сетки анализа;
    `scale` переводит эту сетку в координаты исходного кадра.
    """

    brightness: np.ndarray  # медиана яркости, float32
    spread: np.ndarray  # робастная σ (MAD × 1.4826), float32
    mean: np.ndarray  # среднее яркости, float32
    variance: np.ndarray  # обычная дисперсия, float32
    frame_width: int  # ширина исходного кадра, px
    frame_height: int  # высота исходного кадра, px
    sample_size: int  # сколько кадров вошло в выборку

    @property
    def shape(self) -> tuple[int, int]:
        """Форма карт (h, w) в сетке анализа."""
        return (int(self.brightness.shape[0]), int(self.brightness.shape[1]))

    @property
    def scale(self) -> float:
        """Во сколько раз сетка анализа мельче исходного кадра."""
        return self.frame_width / float(self.brightness.shape[1])


def frame_size(path: Path) -> tuple[int, int]:
    """Размер кадра (ширина, высота) без полного декодирования файла."""
    from PIL import Image  # noqa: PLC0415 — тяжёлый импорт по требованию

    with Image.open(path) as image:
        return (int(image.width), int(image.height))


def _reduced_flag(source_width: int, target_width: int) -> int:
    """Флаг imread, дающий самое дешёвое декодирование не у́же цели.

    JPEG умеет декодироваться сразу в 1/2, 1/4 и 1/8 размера — это дешевле
    полного декодирования с последующим resize и даёт тот же результат с
    точностью до фильтра.
    """
    flags = (
        (8, cv2.IMREAD_REDUCED_GRAYSCALE_8),
        (4, cv2.IMREAD_REDUCED_GRAYSCALE_4),
        (2, cv2.IMREAD_REDUCED_GRAYSCALE_2),
    )
    for factor, flag in flags:
        if source_width // factor >= target_width:
            return flag
    return cv2.IMREAD_GRAYSCALE


def load_gray(
    path: Path,
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH,
    *,
    source_width: int | None = None,
) -> np.ndarray:
    """Прочитать кадр в градациях серого и привести к ширине `analysis_width`.

    :param source_width: ширина исходного кадра, если она уже известна —
        избавляет от чтения заголовка файла на каждом кадре.
    :returns: массив uint8 формы (h, analysis_width).
    """
    if source_width is None:
        source_width = frame_size(path)[0]
    image = cv2.imread(str(path), _reduced_flag(source_width, analysis_width))
    if image is None:
        raise ValueError(f"кадр не читается: {path}")
    if image.shape[1] != analysis_width:
        height = max(1, int(round(analysis_width * image.shape[0] / image.shape[1])))
        image = cv2.resize(
            image, (analysis_width, height), interpolation=cv2.INTER_AREA
        )
    return image


def dominant_frame_size(paths: Sequence[Path]) -> tuple[int, int]:
    """Размер кадра (ширина, высота), который встречается в выборке чаще всех.

    Нечитаемые файлы пропускаются с предупреждением. Размер берётся не с
    первого кадра: если первым оказался кадр из другого источника, масштаб
    «сетка анализа -> кадр» молча испортил бы область на всю запись.

    :raises PipelineError: ни один кадр выборки не читается.
    """
    counts: dict[tuple[int, int], int] = {}
    order: list[tuple[int, int]] = []
    for path in paths:
        try:
            size = frame_size(Path(path))
        except (OSError, ValueError) as exc:
            logger.warning("кадр пропущен, не читается: %s (%s)", path, exc)
            continue
        if size not in counts:
            order.append(size)
        counts[size] = counts.get(size, 0) + 1
    if not counts:
        raise PipelineError(
            f"ни один кадр выборки не прочитан (кадров на входе: {len(paths)})"
        )
    return max(order, key=lambda size: counts[size])


def load_stack(
    paths: Sequence[Path], analysis_width: int = DEFAULT_ANALYSIS_WIDTH
) -> tuple[np.ndarray, tuple[int, int]]:
    """Стек кадров (n, h, w) float32 и размер исходного кадра (ширина, высота).

    Выборка не обязана быть однородной: битый или отсутствующий файл и
    кадр другого размера выбрасываются из выборки с предупреждением, а не
    роняют разбор всей записи. Размер выборки — десятки кадров, потеря
    одного-двух на статистику не влияет. Норма размера — самый частый размер
    в выборке (`dominant_frame_size`), а не размер первого кадра. Если после
    отсева не осталось ни одного кадра, поднимается `PipelineError`:
    продолжать нечем, и это ошибка пайплайна, а не программная ошибка.
    """
    if not paths:
        raise PipelineError("выборка кадров пуста")
    source_w, source_h = dominant_frame_size(paths)

    layers: list[np.ndarray] = []
    skipped = 0
    for path in paths:
        try:
            if frame_size(Path(path)) != (source_w, source_h):
                skipped += 1
                continue
            layers.append(load_gray(Path(path), analysis_width, source_width=source_w))
        except (OSError, ValueError):
            skipped += 1  # предупреждение уже выдано в dominant_frame_size
            continue
    if skipped:
        logger.warning(
            "из выборки выброшено кадров: %d из %d (нечитаемые или размер"
            " не %dx%d — другое соотношения сторон или разрешение)",
            skipped,
            len(paths),
            source_w,
            source_h,
        )
    if not layers:
        raise PipelineError(
            f"ни один кадр выборки не прочитан (кадров на входе: {len(paths)})"
        )
    return np.stack(layers).astype(np.float32), (source_w, source_h)


def pixel_stats(
    paths: Sequence[Path], *, analysis_width: int = DEFAULT_ANALYSIS_WIDTH
) -> PixelStats:
    """Посчитать карты яркости и временно́го разброса по выборке кадров."""
    stack, (source_w, source_h) = load_stack(paths, analysis_width)
    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median), axis=0)
    return PixelStats(
        brightness=median.astype(np.float32),
        spread=(mad * MAD_TO_SIGMA).astype(np.float32),
        mean=stack.mean(axis=0).astype(np.float32),
        variance=stack.var(axis=0).astype(np.float32),
        frame_width=source_w,
        frame_height=source_h,
        sample_size=int(stack.shape[0]),
    )


def static_bright_mask(
    stats: PixelStats,
    *,
    max_spread: float = DEFAULT_MAX_SPREAD,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
) -> np.ndarray:
    """Маска «статично во времени И светло» — кандидат в область демонстрации.

    :returns: массив uint8 (0/1) формы сетки анализа.
    """
    mask = (stats.spread <= max_spread) & (stats.brightness >= min_brightness)
    return mask.astype(np.uint8)


def sample_frames(
    frames: Sequence[Frame],
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    count: int = 100,
) -> list[Frame]:
    """Равномерная выборка не более `count` кадров из окна [start_s, end_s).

    Выборка детерминирована: индексы берутся по равномерной сетке, без
    случайности, — повторный прогон даёт тот же результат.
    """
    if count <= 0:
        raise ValueError(f"размер выборки должен быть положительным, получено {count}")
    window = [
        frame
        for frame in frames
        if (start_s is None or frame.timestamp_s >= start_s)
        and (end_s is None or frame.timestamp_s < end_s)
    ]
    if len(window) <= count:
        return window
    positions = np.linspace(0, len(window) - 1, count)
    picked = sorted({int(round(pos)) for pos in positions})
    return [window[i] for i in picked]
