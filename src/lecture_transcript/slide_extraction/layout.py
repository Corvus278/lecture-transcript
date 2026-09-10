"""Отслеживание раскладки кадра по всей записи и интервалы стабильной раскладки.

Раскладка меняется в течение записи, поэтому дисперсия считается не по всей
записи разом, а по скользящим окнам, и область демонстрации фиксируется на
интервал стабильной раскладки (design D3).

Порядок работы
--------------
1. Запись режется на окна длиной `window_s`; в каждом по выборке кадров
   ищется область демонстрации (`region.find_region`).
2. По каждому кадру считается дешёвый признак присутствия демонстрации —
   доля светлых пикселей внутри области окна. Это отдельный, покадровый
   признак: область находится по выборке, а включена ли демонстрация именно
   в этом кадре — вопрос к самому кадру.
3. Покадровый признак сглаживается гистерезисом (см. ниже).
4. Интервал разрезается там, где область соседних окон меняется
   существенно (IoU ниже порога), и область интервала берётся как
   объединение областей его окон: у разных слайдов одного шаблона светлое
   поле чуть разное, и по отдельному окну область выходит по их
   пересечению — кроп для OCR так терял бы содержимое.

Гистерезис: что он на самом деле различает
------------------------------------------
На эталонной платформе **переключение слайда сопровождается исчезновением
демонстрации**: лектор снимает демонстрацию, тайлы участников на несколько
секунд разворачиваются на весь кадр, затем демонстрация включается снова
уже со следующим слайдом. Такие провалы — настоящие, а не артефакт
детекта: демонстрации в эти секунды физически нет, и разведка эталона
(`tests/fixtures_manifest.json`) фиксирует их как `no_demo_intervals`
(33 интервала, 247 с суммарно, длина от 3 до 60 с).

Поэтому `min_absence_s` **не** разделяет «артефакт детекта» и «настоящее
выключение» — по длительности они не разделяются вовсе. Второй по длине
настоящий провал на эталоне — 2399..2412 (13 с), и он же служит
контрольной точкой задачи 3.2. Порог различает ровно то, что назван
различать в конфиге: **короткое отсутствие демонстрации, недостаточное для
смены раскладки, и длинное**. Смысл у него один — не дробить интервалы
стабильной раскладки на десятки кусков там, где раскладка на деле одна;
поэтому он и согласован с `min_layout_interval_s`.

Важно: гистерезис сглаживает **только границы интервалов раскладки** и не
подменяет собой покадровую правду. Сырой покадровый признак остаётся в
`LayoutAnalysis.present`; по нему стадия дедупликации выбрасывает кадры,
в которых демонстрации физически нет, и — это принципиально — **режет по
ним границы слайдов**, чтобы отсутствие демонстрации не попадало внутрь
интервала слайда, а оставалось разрывом (спека, «Разрывы между слайдами»).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..contracts import Frame, LayoutInterval, PipelineError, Rect
from .region import (
    DEFAULT_BORDER_LEVEL_RATIO,
    DEFAULT_BORDER_LINE_RATIO,
    DEFAULT_CLOSE_PX,
    DEFAULT_MIN_AREA_RATIO,
    DEFAULT_MIN_FILL_RATIO,
    DEFAULT_MIN_SIDE_RATIO,
    DEFAULT_OPEN_PX,
    find_region,
    iou,
    union_rect,
)
from .variance import (
    DEFAULT_ANALYSIS_WIDTH,
    DEFAULT_MAX_SPREAD,
    DEFAULT_MIN_BRIGHTNESS,
    dominant_frame_size,
    load_gray,
    sample_frames,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MIN_ABSENCE_S",
    "DEFAULT_MIN_LAYOUT_INTERVAL_S",
    "DEFAULT_PRESENCE_RATIO",
    "DEFAULT_SAMPLE_FRAMES",
    "DEFAULT_STABLE_REGION_IOU",
    "DEFAULT_WINDOW_S",
    "LayoutAnalysis",
    "analyze_layout",
    "detect_layout",
    "presence_ratio",
    "smooth_presence",
]

#: Длина окна анализа раскладки, с (совпадает с `variance_window_s`).
DEFAULT_WINDOW_S = 300.0

#: Сколько кадров сэмплировать в окне (совпадает с `variance_sample_frames`).
DEFAULT_SAMPLE_FRAMES = 100

#: Минимальная длительность отсутствия демонстрации, с, при которой
#: интервал раскладки разрезается. Гасится только одиночный кадр —
#: неотличимый от сбоя детекта. На эталоне самое короткое настоящее
#: отсутствие — 3 с, одно-двухкадровых провалов нет вовсе; 2 с оставляют
#: ошибку в сторону лишнего разреза, а не потерянного выключения
#: (см. docstring модуля).
DEFAULT_MIN_ABSENCE_S = 2.0

#: Минимальная длительность интервала стабильной раскладки, с.
DEFAULT_MIN_LAYOUT_INTERVAL_S = 20.0

#: Доля светлых пикселей внутри области, ниже которой демонстрация в кадре
#: считается выключенной. На эталонной записи распределения разведены
#: широко: при включённой демонстрации 5-й процентиль доли — 0.79, при
#: выключенной максимум — 0.13. Порог 0.4 стоит в середине разрыва.
DEFAULT_PRESENCE_RATIO = 0.4

#: IoU, ниже которого область соседнего окна считается новой раскладкой и
#: интервал разрезается. На эталоне области соседних окон расходятся до
#: IoU 0.86 из-за разного светлого поля у разных слайдов шаблона — это одна
#: и та же раскладка, поэтому порог опущен до 0.75.
DEFAULT_STABLE_REGION_IOU = 0.75


@dataclass(frozen=True)
class LayoutAnalysis:
    """Результат разбора раскладки по всей записи.

    `present` идёт параллельно `frames` и хранит **сырой** покадровый
    признак присутствия демонстрации, до гистерезиса. `intervals` —
    сглаженные интервалы стабильной раскладки, покрывающие всю запись
    без разрывов и пересечений.
    """

    frames: tuple[Frame, ...]
    present: tuple[bool, ...]
    intervals: tuple[LayoutInterval, ...]

    def interval_frames(self, interval: LayoutInterval) -> tuple[Frame, ...]:
        """Кадры, попавшие в интервал."""
        return tuple(
            frame
            for frame in self.frames
            if interval.start_s <= frame.timestamp_s < interval.end_s
        )

    def present_frames(self, interval: LayoutInterval) -> tuple[Frame, ...]:
        """Кадры интервала, в которых демонстрация присутствует физически."""
        return tuple(
            frame
            for frame, is_present in zip(self.frames, self.present)
            if is_present and interval.start_s <= frame.timestamp_s < interval.end_s
        )


def presence_ratio(
    frame_gray: np.ndarray,
    region: Rect,
    *,
    scale: float,
    level: float = DEFAULT_MIN_BRIGHTNESS,
) -> float:
    """Доля пикселей ярче `level` внутри области, по кадру в сетке анализа.

    :param scale: во сколько раз сетка анализа мельче исходного кадра.
    """
    x = int(round(region.x / scale))
    y = int(round(region.y / scale))
    w = max(1, int(round(region.width / scale)))
    h = max(1, int(round(region.height / scale)))
    crop = frame_gray[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    return float((crop >= level).mean())


def analyze_layout(
    frames: Sequence[Frame],
    *,
    window_s: float = DEFAULT_WINDOW_S,
    sample_size: int = DEFAULT_SAMPLE_FRAMES,
    min_absence_s: float = DEFAULT_MIN_ABSENCE_S,
    min_layout_interval_s: float = DEFAULT_MIN_LAYOUT_INTERVAL_S,
    presence_level: float = DEFAULT_MIN_BRIGHTNESS,
    min_presence_ratio: float = DEFAULT_PRESENCE_RATIO,
    stable_region_iou: float = DEFAULT_STABLE_REGION_IOU,
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
    region: Rect | None = None,
) -> LayoutAnalysis:
    """Разобрать раскладку по всей записи и вернуть интервалы и покадровый признак.

    :param region: область демонстрации, заданная вручную. Если передана,
        детект области не выполняется вовсе и во всех окнах используется
        она — способ обработать запись, на которой детект не сходится
        (design, Risks: «при неуверенном детекте — явная диагностика и
        возможность задать область вручную»). Покадровый признак
        присутствия демонстрации при этом считается как обычно.
    """
    ordered = tuple(sorted(frames, key=lambda f: f.timestamp_s))
    if not ordered:
        return LayoutAnalysis(frames=(), present=(), intervals=())

    region_kwargs = dict(
        analysis_width=analysis_width,
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
    )
    step_s = _step_seconds(ordered)
    start_s = ordered[0].timestamp_s
    end_s = ordered[-1].timestamp_s + step_s

    if region is not None:
        logger.info(
            "область демонстрации задана вручную: %dx%d+%d+%d — детект пропущен",
            region.width,
            region.height,
            region.x,
            region.y,
        )
        fallback = region
        window_count = max(1, int(np.ceil((end_s - start_s) / window_s)))
        window_regions: list[Rect | None] = [region] * window_count
    else:
        # Опорная область по всей записи: подстраховка для окон, в которых
        # своя область не нашлась (окно целиком без демонстрации либо
        # слишком рваное). Без неё такое окно молча теряет все свои кадры.
        fallback = find_region(
            [f.path for f in sample_frames(ordered, count=sample_size)],
            **region_kwargs,
        )
        if fallback is None:
            logger.warning(
                "опорная область демонстрации по всей записи не найдена —"
                " см. предупреждения выше о том, какая проверка не сошлась."
                " Если демонстрация в записи есть, область можно задать"
                " вручную параметром region"
            )
        window_regions = _window_regions(
            ordered,
            start_s=start_s,
            end_s=end_s,
            window_s=window_s,
            sample_size=sample_size,
            fallback=fallback,
            region_kwargs=region_kwargs,
        )
    per_frame_region = [
        window_regions[_window_index(frame.timestamp_s, start_s, window_s)]
        for frame in ordered
    ]

    present = _frame_presence(
        ordered,
        per_frame_region,
        analysis_width=analysis_width,
        presence_level=presence_level,
        min_presence_ratio=min_presence_ratio,
        source_size=dominant_frame_size(
            [f.path for f in sample_frames(ordered, count=sample_size)]
        ),
    )
    smoothed = smooth_presence(present, step_s=step_s, min_absence_s=min_absence_s)
    intervals = _build_intervals(
        ordered,
        smoothed=smoothed,
        per_frame_region=per_frame_region,
        step_s=step_s,
        end_s=end_s,
        min_layout_interval_s=min_layout_interval_s,
        stable_region_iou=stable_region_iou,
        fallback=fallback,
    )
    logger.info(
        "раскладка разобрана: кадров=%d, интервалов=%d, с демонстрацией=%d",
        len(ordered),
        len(intervals),
        sum(1 for i in intervals if i.region is not None),
    )
    return LayoutAnalysis(
        frames=ordered, present=tuple(present), intervals=tuple(intervals)
    )


def detect_layout(frames: Sequence[Frame], **kwargs) -> list[LayoutInterval]:
    """Интервалы стабильной раскладки — публичная форма результата 3.3."""
    return list(analyze_layout(frames, **kwargs).intervals)


# --------------------------------------------------------------------------
# Внутреннее
# --------------------------------------------------------------------------


def _step_seconds(frames: Sequence[Frame]) -> float:
    """Шаг выборки кадров, с. Медиана разностей устойчивее к пропускам."""
    if len(frames) < 2:
        return 1.0
    deltas = np.diff([f.timestamp_s for f in frames])
    step = float(np.median(deltas))
    return step if step > 0 else 1.0


def _window_index(timestamp_s: float, start_s: float, window_s: float) -> int:
    return int((timestamp_s - start_s) // window_s)


def _window_regions(
    frames: Sequence[Frame],
    *,
    start_s: float,
    end_s: float,
    window_s: float,
    sample_size: int,
    fallback: Rect | None,
    region_kwargs: dict,
) -> list[Rect | None]:
    """Область демонстрации по каждому окну записи."""
    count = max(1, int(np.ceil((end_s - start_s) / window_s)))
    regions: list[Rect | None] = []
    unsure = 0
    for index in range(count):
        lo = start_s + index * window_s
        picked = sample_frames(frames, start_s=lo, end_s=lo + window_s, count=sample_size)
        region = (
            find_region([f.path for f in picked], **region_kwargs) if picked else None
        )
        if region is None:
            unsure += 1
            logger.warning(
                "окно %.0f..%.0f с: область демонстрации не определена"
                " (кадров в выборке %d), взята опорная область по записи: %s",
                lo,
                min(lo + window_s, end_s),
                len(picked),
                "нет и её" if fallback is None else
                f"{fallback.width}x{fallback.height}+{fallback.x}+{fallback.y}",
            )
        regions.append(region if region is not None else fallback)
    if unsure:
        logger.warning(
            "детект области демонстрации неуверенный: %d окно(окон) из %d"
            " без собственной области. Область можно задать вручную"
            " параметром region",
            unsure,
            count,
        )
    return regions


def _frame_presence(
    frames: Sequence[Frame],
    per_frame_region: Sequence[Rect | None],
    *,
    analysis_width: int,
    presence_level: float,
    min_presence_ratio: float,
    source_size: tuple[int, int],
) -> list[bool]:
    """Покадровый признак «демонстрация включена».

    Кадр, который не читается или имеет не тот размер, что у записи,
    считается кадром без демонстрации и отмечается предупреждением: судить
    о нём нечем, а один такой кадр не должен ронять разбор всей записи.
    Если не прочитан ни один кадр — это `PipelineError`.
    """
    source_width, source_height = source_size
    expected_height = max(1, int(round(analysis_width * source_height / source_width)))
    scale = source_width / analysis_width
    flags: list[bool] = []
    bad = 0
    for frame, region in zip(frames, per_frame_region):
        try:
            gray = load_gray(frame.path, analysis_width, source_width=source_width)
        except (OSError, ValueError) as exc:
            bad += 1
            logger.warning(
                "кадр t=%.2f не читается, считается без демонстрации: %s",
                frame.timestamp_s,
                exc,
            )
            flags.append(False)
            continue
        if gray.shape != (expected_height, analysis_width):
            bad += 1
            logger.warning(
                "кадр t=%.2f другого размера, считается без демонстрации: %s",
                frame.timestamp_s,
                frame.path,
            )
            flags.append(False)
            continue
        if region is None:
            flags.append(False)
            continue
        ratio = presence_ratio(gray, region, scale=scale, level=presence_level)
        flags.append(ratio >= min_presence_ratio)
    if frames and bad == len(frames):
        raise PipelineError(
            f"ни один кадр записи не прочитан (кадров: {len(frames)})"
        )
    if bad:
        logger.warning(
            "кадров, пропущенных при проверке присутствия демонстрации: %d из %d",
            bad,
            len(frames),
        )
    return flags


def _runs(flags: Sequence[bool]) -> list[tuple[int, int, bool]]:
    """Разбить последовательность флагов на пробеги (начало, конец+1, значение)."""
    result: list[tuple[int, int, bool]] = []
    start = 0
    for index in range(1, len(flags) + 1):
        if index == len(flags) or flags[index] != flags[start]:
            result.append((start, index, bool(flags[start])))
            start = index
    return result


def smooth_presence(
    flags: Sequence[bool],
    *,
    step_s: float,
    min_absence_s: float,
) -> list[bool]:
    """Убрать короткие провалы признака присутствия.

    Сначала гасятся одиночные всплески «включено» длиной в один-два кадра —
    это шум детекта, и он опасен тем, что разрезал бы длинный настоящий
    провал на два коротких, которые дальше были бы «залечены». Затем
    заполняются пробеги «выключено» короче `min_absence_s` — артефакты
    переключения слайда.

    Провалы длиннее одного-двух кадров не гасятся: на эталонной записи
    между двумя близкими переключениями слайда встречается настоящий
    показ длиной 12 с, и порог «минимальной стабильной раскладки» здесь
    убил бы живой слайд.
    """
    smoothed = list(flags)
    noise_s = 2.0 * step_s
    for start, stop, value in _runs(smoothed):
        if value and (stop - start) * step_s <= noise_s:
            smoothed[start:stop] = [False] * (stop - start)
    for start, stop, value in _runs(smoothed):
        if value or (stop - start) * step_s >= min_absence_s:
            continue
        # Провал у самого края записи заполнять нечем — там нет соседа,
        # подтверждающего, что демонстрация продолжается.
        if start == 0 or stop == len(smoothed):
            continue
        smoothed[start:stop] = [True] * (stop - start)
    return smoothed


def _build_intervals(
    frames: Sequence[Frame],
    *,
    smoothed: Sequence[bool],
    per_frame_region: Sequence[Rect | None],
    step_s: float,
    end_s: float,
    min_layout_interval_s: float,
    stable_region_iou: float,
    fallback: Rect | None,
) -> list[LayoutInterval]:
    """Собрать непрерывное покрытие записи интервалами стабильной раскладки."""
    segments: list[tuple[int, int, bool]] = []
    for start, stop, value in _runs(smoothed):
        if not value:
            segments.append((start, stop, False))
            continue
        pieces = _split_by_region_change(
            per_frame_region[start:stop], offset=start, min_iou=stable_region_iou
        )
        pieces = _merge_short(
            pieces, step_s=step_s, min_length_s=min_layout_interval_s
        )
        segments.extend((lo, hi, True) for lo, hi in pieces)

    intervals: list[LayoutInterval] = []
    for start, stop, value in segments:
        region: Rect | None = None
        if value:
            # Область интервала — объединение областей его окон: шаблон
            # оформления у разных слайдов оставляет светлым чуть разное
            # поле, и по отдельному окну область выходит по их пересечению.
            region = union_rect(
                [r for r in per_frame_region[start:stop] if r is not None]
            ) or fallback
        intervals.append(
            LayoutInterval(
                start_s=float(frames[start].timestamp_s),
                end_s=float(
                    frames[stop].timestamp_s if stop < len(frames) else end_s
                ),
                region=region,
            )
        )
    return intervals


def _split_by_region_change(
    regions: Sequence[Rect | None], *, offset: int, min_iou: float
) -> list[tuple[int, int]]:
    """Разрезать пробег там, где область окна сменилась на существенно иную."""
    bounds = [0]
    for index in range(1, len(regions)):
        previous, current = regions[index - 1], regions[index]
        if previous is None or current is None:
            if previous is not current:
                bounds.append(index)
            continue
        if iou(previous, current) < min_iou:
            bounds.append(index)
    bounds.append(len(regions))
    return [
        (offset + bounds[i], offset + bounds[i + 1]) for i in range(len(bounds) - 1)
    ]


def _merge_short(
    pieces: Sequence[tuple[int, int]], *, step_s: float, min_length_s: float
) -> list[tuple[int, int]]:
    """Слить куски короче `min_length_s` с соседями.

    Кусок короче минимальной стабильной раскладки — не новая раскладка, а
    дрожание границ области на стыке окон. Отдельным интервалом он быть не
    должен, но и терять его кадры нельзя: они приклеиваются к соседу.
    """
    merged: list[list[int]] = []
    for lo, hi in pieces:
        if merged and (hi - lo) * step_s < min_length_s:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    while len(merged) > 1 and (merged[0][1] - merged[0][0]) * step_s < min_length_s:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [(lo, hi) for lo, hi in merged]
