"""Стадия slide-extraction целиком: `FramesArtifact` -> `list[Slide]`.

Порядок работы соответствует design D3/D4:

1. `layout.analyze_layout` — скользящее окно по всей записи, интервалы
   стабильной раскладки и покадровый признак присутствия демонстрации;
2. `dedup.group_frames` — внутри каждого интервала с демонстрацией
   последовательные кадры группируются в логические слайды по perceptual
   hash, репрезентативный кадр группы — последний;
3. `slides.build_slides` — кроп репрезентативного кадра сохраняется в PNG
   со стабильным именем, собирается упорядоченный список `Slide`.

Кадры, в которых демонстрации физически нет, из группировки исключаются:
их тёмное содержимое исказило бы хэш и породило бы лишние слайды. Отрезок
без демонстрации режет группу и остаётся разрывом между слайдами.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from ..contracts import FramesArtifact, Rect, Slide
from .dedup import (
    DEFAULT_HAMMING_THRESHOLD,
    DEFAULT_HASH_SIZE,
    FrameGroup,
    group_frames,
)
from .layout import (
    DEFAULT_MIN_ABSENCE_S,
    DEFAULT_MIN_LAYOUT_INTERVAL_S,
    DEFAULT_PRESENCE_RATIO,
    DEFAULT_SAMPLE_FRAMES,
    DEFAULT_STABLE_REGION_IOU,
    DEFAULT_WINDOW_S,
    LayoutAnalysis,
    analyze_layout,
)
from .region import (
    DEFAULT_CLOSE_PX,
    find_region,
    iou,
    DEFAULT_MIN_AREA_RATIO,
    DEFAULT_MIN_FILL_RATIO,
    DEFAULT_MIN_SIDE_RATIO,
    DEFAULT_OPEN_PX,
)
from .slides import build_slides
from .variance import (
    DEFAULT_ANALYSIS_WIDTH,
    DEFAULT_MAX_SPREAD,
    DEFAULT_MIN_BRIGHTNESS,
    dominant_frame_size,
    sample_frames,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_REFINE_CLOSE_PX",
    "DEFAULT_REFINE_MIN_IOU",
    "DEFAULT_REFINE_SAMPLE_FRAMES",
    "extract_slides",
    "groups_from_layout",
    "region_at_representative",
]

#: Сколько последних кадров группы берётся, чтобы найти область на момент
#: репрезентативного кадра. Внутри группы размер окна демонстрации не
#: меняется: смена размера даёт скачок хэша 98..108 и режет группу.
DEFAULT_REFINE_SAMPLE_FRAMES = 5

#: Ядро закрытия при уточнении области по нескольким кадрам, px сетки.
#: За несколько секунд живые камеры почти неподвижны, их светлые пиксели
#: попадают в маску, и закрытие 9×9 перекидывает маску через тёмный
#: промежуток между плитками (на эталоне — 1–2 px сетки). 3×3 промежуток
#: не перешагивает; при неудаче пробуется обычное закрытие.
DEFAULT_REFINE_CLOSE_PX = 3

#: Минимальное IoU уточнённой области с областью интервала: уточнение
#: подстраивает размер того же окна демонстрации, а не ищет новое.
DEFAULT_REFINE_MIN_IOU = 0.5


def region_at_representative(
    group: FrameGroup,
    *,
    region_kwargs: dict | None = None,
    sample_frames_count: int = DEFAULT_REFINE_SAMPLE_FRAMES,
    min_iou: float = DEFAULT_REFINE_MIN_IOU,
) -> Rect:
    """Область демонстрации на момент репрезентативного кадра группы.

    Область интервала раскладки считается по окнам анализа длиной в минуты,
    а окно демонстрации внутри интервала может сменить размер без провала
    (на эталоне 1272×716 <-> 1360×764 шесть раз). Одна область на интервал
    тогда либо режет слайд (малая область на большом окне), либо захватывает
    полосу с камерами участников (большая на малом). Поэтому область
    переопределяется по последним кадрам группы.

    Признак «статично» на нескольких кадрах вырожден, но здесь он и не
    нужен: присутствие демонстрации в этих кадрах уже установлено, ядро
    ищется по светлому полю, а расширение останавливается на тёмном
    промежутке между плитками. Если уточнение не удалось или нашло
    непохожее окно, остаётся область интервала — с предупреждением.
    """
    kwargs = dict(region_kwargs or {})
    kwargs.pop("min_sample_size", None)
    base_close = kwargs.pop("close_px", DEFAULT_CLOSE_PX)
    paths = [frame.path for frame in group.frames[-sample_frames_count:]]
    for close_px in dict.fromkeys((DEFAULT_REFINE_CLOSE_PX, base_close)):
        found = find_region(paths, min_sample_size=1, close_px=close_px, **kwargs)
        if found is not None and iou(found, group.region) >= min_iou:
            return found
    logger.warning(
        "слайд %.0f..%.0f с: область на момент репрезентативного кадра не"
        " уточнена, оставлена область интервала %dx%d+%d+%d",
        group.start_s,
        group.end_s,
        group.region.width,
        group.region.height,
        group.region.x,
        group.region.y,
    )
    return group.region


def groups_from_layout(
    analysis: LayoutAnalysis,
    *,
    hash_size: int = DEFAULT_HASH_SIZE,
    hamming_threshold: int,
    step_s: float = 1.0,
    min_group_s: float = 0.0,
    min_absence_s: float = DEFAULT_MIN_ABSENCE_S,
    refine_regions: bool = True,
    region_kwargs: dict | None = None,
) -> list[FrameGroup]:
    """Логические слайды по разобранной раскладке, по возрастанию времени.

    :param refine_regions: переопределять область каждой группы на момент её
        репрезентативного кадра (`region_at_representative`). Выключается,
        когда область задана вручную.
    """
    groups: list[FrameGroup] = []
    for interval in analysis.intervals:
        if interval.region is None:
            continue
        present = analysis.present_frames(interval)
        if not present:
            continue
        groups.extend(
            group_frames(
                present,
                interval.region,
                hash_size=hash_size,
                threshold=hamming_threshold,
                step_s=step_s,
                min_group_s=min_group_s,
                min_absence_s=min_absence_s,
            )
        )
    if refine_regions:
        groups = [
            replace(group, region=region_at_representative(group, region_kwargs=region_kwargs))
            for group in groups
        ]
    return sorted(groups, key=lambda g: (g.start_s, g.end_s))


def extract_slides(
    frames: FramesArtifact,
    out_dir: Path | str,
    *,
    source: Path | None = None,
    hamming_threshold: int | None = None,
    hash_size: int | None = None,
    variance_sample_frames: int | None = None,
    variance_window_s: float | None = None,
    min_region_area_ratio: float | None = None,
    min_layout_interval_s: float | None = None,
    min_absence_s: float | None = None,
    min_group_s: float = 0.0,
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH,
    max_spread: float | None = None,
    min_brightness: float | None = None,
    min_presence_ratio: float = DEFAULT_PRESENCE_RATIO,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    min_side_ratio: float = DEFAULT_MIN_SIDE_RATIO,
    stable_region_iou: float = DEFAULT_STABLE_REGION_IOU,
    close_px: int = DEFAULT_CLOSE_PX,
    open_px: int = DEFAULT_OPEN_PX,
    expand_border: bool = True,
    region: Rect | None = None,
    config=None,
) -> list[Slide]:
    """Превратить выборку кадров в упорядоченный список логических слайдов.

    :param frames: артефакт стадии media-ingest.
    :param out_dir: каталог для PNG-кропов слайдов.
    :param source: исходный mp4 — репрезентативный кадр переизвлекается из
        него точечно, чтобы кроп для OCR был без артефактов JPEG.
    :param region: область демонстрации, заданная вручную; детект области
        при этом не выполняется (design, Risks).
    :param config: `lecture_transcript.config.SlideExtractionConfig`; его
        значения используются как дефолты для одноимённых параметров,
        явный аргумент всегда важнее.

    Параметры, дублирующие поля конфига, объявлены со значением `None`:
    только так «не задано» отличается от «задано значение, совпавшее с
    дефолтом», и правка YAML действительно доходит до места использования.
    """
    #: Поле конфига -> имя параметра. Дефолты берутся из констант модулей.
    resolved = dict(
        hamming_threshold=hamming_threshold,
        hash_size=hash_size,
        variance_sample_frames=variance_sample_frames,
        variance_window_s=variance_window_s,
        min_region_area_ratio=min_region_area_ratio,
        min_layout_interval_s=min_layout_interval_s,
        min_absence_s=min_absence_s,
        max_spread=max_spread,
        min_brightness=min_brightness,
    )
    fallbacks = dict(
        hamming_threshold=DEFAULT_HAMMING_THRESHOLD,
        hash_size=DEFAULT_HASH_SIZE,
        variance_sample_frames=DEFAULT_SAMPLE_FRAMES,
        variance_window_s=DEFAULT_WINDOW_S,
        min_region_area_ratio=DEFAULT_MIN_AREA_RATIO,
        min_layout_interval_s=DEFAULT_MIN_LAYOUT_INTERVAL_S,
        min_absence_s=DEFAULT_MIN_ABSENCE_S,
        max_spread=DEFAULT_MAX_SPREAD,
        min_brightness=DEFAULT_MIN_BRIGHTNESS,
    )
    for name, value in resolved.items():
        if value is not None:
            continue
        if config is not None and hasattr(config, name):
            resolved[name] = getattr(config, name)
        else:
            resolved[name] = fallbacks[name]
    hamming_threshold = resolved["hamming_threshold"]
    hash_size = resolved["hash_size"]
    variance_sample_frames = resolved["variance_sample_frames"]
    variance_window_s = resolved["variance_window_s"]
    min_region_area_ratio = resolved["min_region_area_ratio"]
    min_layout_interval_s = resolved["min_layout_interval_s"]
    min_absence_s = resolved["min_absence_s"]
    max_spread = resolved["max_spread"]
    min_brightness = resolved["min_brightness"]
    if region is None and config is not None:
        configured = getattr(config, "region", None)
        if configured is not None:
            region = configured if isinstance(configured, Rect) else Rect(*configured)

    if not frames.frames:
        logger.info("кадров нет — список слайдов пуст")
        return []

    step_s = 1.0 / frames.fps if frames.fps else 1.0
    analysis = analyze_layout(
        frames.frames,
        window_s=variance_window_s,
        sample_size=variance_sample_frames,
        min_absence_s=min_absence_s,
        min_layout_interval_s=min_layout_interval_s,
        presence_level=min_brightness,
        min_presence_ratio=min_presence_ratio,
        stable_region_iou=stable_region_iou,
        analysis_width=analysis_width,
        max_spread=max_spread,
        min_brightness=min_brightness,
        min_area_ratio=min_region_area_ratio,
        min_fill_ratio=min_fill_ratio,
        min_side_ratio=min_side_ratio,
        close_px=close_px,
        open_px=open_px,
        expand_border=expand_border,
        region=region,
    )
    groups = groups_from_layout(
        analysis,
        hash_size=hash_size,
        hamming_threshold=hamming_threshold,
        step_s=step_s,
        min_group_s=min_group_s,
        min_absence_s=min_absence_s,
        refine_regions=region is None,
        region_kwargs=dict(
            analysis_width=analysis_width,
            max_spread=max_spread,
            min_brightness=min_brightness,
            min_area_ratio=min_region_area_ratio,
            min_fill_ratio=min_fill_ratio,
            min_side_ratio=min_side_ratio,
            close_px=close_px,
            open_px=open_px,
            expand_border=expand_border,
        ),
    )
    if not groups:
        logger.warning(
            "демонстрация в записи не обнаружена — список слайдов пуст."
            " Интервалов раскладки %d, из них с областью %d; кадров с"
            " демонстрацией %d из %d. Если демонстрация в записи есть,"
            " см. предупреждения детекта выше и задайте область вручную"
            " параметром region",
            len(analysis.intervals),
            sum(1 for i in analysis.intervals if i.region is not None),
            sum(1 for p in analysis.present if p),
            len(analysis.frames),
        )
        return []

    return build_slides(
        groups,
        out_dir,
        source=source,
        frame_width=dominant_frame_size(
            [f.path for f in sample_frames(frames.frames, count=variance_sample_frames)]
        )[0],
    )
