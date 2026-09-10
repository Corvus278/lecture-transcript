"""Стадия slide-extraction: кадры видео -> упорядоченный список слайдов.

Публичный API стадии:

* `extract_slides` — стадия целиком, `FramesArtifact` -> `list[Slide]`;
* `analyze_layout` / `detect_layout` — интервалы стабильной раскладки и
  область демонстрации на каждом из них (design D3);
* `find_region` / `region_from_mask` — область демонстрации по выборке
  кадров либо по готовой маске;
* `pixel_stats` / `static_bright_mask` — карты временно́го разброса и
  яркости и маска «статично и светло», по которой ищется область;
* `group_frames` — группировка кадров в логические слайды по perceptual
  hash, репрезентативный кадр группы — последний (design D4);
* `build_slides` — PNG-кропы со стабильными именами и список `Slide`.

Типы данных берутся из `lecture_transcript.contracts` и реэкспортируются
здесь для удобства вызывающего кода.
"""

from __future__ import annotations

from ..contracts import Frame, FramesArtifact, LayoutInterval, PipelineError, Rect, Slide
from .dedup import (
    DEFAULT_HAMMING_THRESHOLD,
    DEFAULT_HASH_SIZE,
    FrameGroup,
    crop_hash,
    group_frames,
    hash_distances,
    hash_sequence,
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
    detect_layout,
    presence_ratio,
    smooth_presence,
)
from .pipeline import extract_slides, groups_from_layout
from .region import (
    DEFAULT_BORDER_LEVEL_RATIO,
    DEFAULT_BORDER_LINE_RATIO,
    DEFAULT_MIN_AREA_RATIO,
    DEFAULT_MIN_FILL_RATIO,
    DEFAULT_MIN_SAMPLE_SIZE,
    DEFAULT_MIN_SIDE_RATIO,
    expand_to_static_border,
    find_region,
    iou,
    region_from_mask,
    region_from_stats,
)
from .slides import (
    FFMPEG_TIMEOUT_S,
    SLIDE_FILENAME_TEMPLATE,
    SlideExtractionError,
    build_slides,
    scale_region,
    validate_slides,
)
from .variance import (
    DEFAULT_ANALYSIS_WIDTH,
    DEFAULT_MAX_SPREAD,
    DEFAULT_MIN_BRIGHTNESS,
    PixelStats,
    load_gray,
    pixel_stats,
    sample_frames,
    static_bright_mask,
)

__all__ = [
    # стадия целиком
    "extract_slides",
    "groups_from_layout",
    # раскладка и область демонстрации
    "analyze_layout",
    "detect_layout",
    "LayoutAnalysis",
    "presence_ratio",
    "smooth_presence",
    "find_region",
    "expand_to_static_border",
    "region_from_mask",
    "region_from_stats",
    "iou",
    # статистики кадров
    "PixelStats",
    "pixel_stats",
    "static_bright_mask",
    "load_gray",
    "sample_frames",
    # дедупликация
    "FrameGroup",
    "group_frames",
    "crop_hash",
    "hash_sequence",
    "hash_distances",
    # сохранение
    "build_slides",
    "validate_slides",
    "scale_region",
    "SLIDE_FILENAME_TEMPLATE",
    "FFMPEG_TIMEOUT_S",
    # типы контракта
    "Frame",
    "FramesArtifact",
    "LayoutInterval",
    "Rect",
    "Slide",
    # ошибки
    "PipelineError",
    "SlideExtractionError",
    # дефолты
    "DEFAULT_ANALYSIS_WIDTH",
    "DEFAULT_BORDER_LEVEL_RATIO",
    "DEFAULT_BORDER_LINE_RATIO",
    "DEFAULT_HAMMING_THRESHOLD",
    "DEFAULT_HASH_SIZE",
    "DEFAULT_MAX_SPREAD",
    "DEFAULT_MIN_ABSENCE_S",
    "DEFAULT_MIN_AREA_RATIO",
    "DEFAULT_MIN_BRIGHTNESS",
    "DEFAULT_MIN_FILL_RATIO",
    "DEFAULT_MIN_LAYOUT_INTERVAL_S",
    "DEFAULT_MIN_SAMPLE_SIZE",
    "DEFAULT_MIN_SIDE_RATIO",
    "DEFAULT_SAMPLE_FRAMES",
    "DEFAULT_STABLE_REGION_IOU",
    "DEFAULT_PRESENCE_RATIO",
    "DEFAULT_WINDOW_S",
]
