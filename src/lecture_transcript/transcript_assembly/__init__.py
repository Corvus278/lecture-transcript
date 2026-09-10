"""Стадия `merge`: слияние речи и слайдов и генерация `transcript.md`.

Публичный API стадии. Внутренние модули друг на друга ссылаются только вниз
по цепочке merge -> cleanup -> paragraphs -> render -> pipeline; другие стадии
пайплайна импортируют лишь `contracts`.
"""

from .cleanup import (
    ALLOWED_DOUBLES,
    AMBIGUOUS_FILLER_PHRASES,
    FILLER_PHRASES,
    FILLER_WORDS,
    MATH_TOKENS,
    NUMERAL_WORDS,
    CleanupPolicy,
    cleanup_sections,
    cleanup_words,
)
from .consolidate import ConsolidationPolicy, consolidate_sections
from .merge import BoundaryPolicy, merge_sections
from .paragraphs import (
    ParagraphPolicy,
    fill_paragraphs,
    paragraph_text,
    split_paragraphs,
    split_words,
)
from .pipeline import (
    assemble,
    consolidation_policy_from_config,
    policies_from_config,
    write_transcript,
)
from .render import (
    RenderPolicy,
    format_interval,
    format_timecode,
    relative_image_path,
    render_markdown,
)

__all__ = [
    "ALLOWED_DOUBLES",
    "AMBIGUOUS_FILLER_PHRASES",
    "FILLER_PHRASES",
    "FILLER_WORDS",
    "MATH_TOKENS",
    "NUMERAL_WORDS",
    "BoundaryPolicy",
    "CleanupPolicy",
    "ConsolidationPolicy",
    "ParagraphPolicy",
    "RenderPolicy",
    "assemble",
    "cleanup_sections",
    "cleanup_words",
    "consolidate_sections",
    "consolidation_policy_from_config",
    "fill_paragraphs",
    "format_interval",
    "format_timecode",
    "merge_sections",
    "paragraph_text",
    "policies_from_config",
    "relative_image_path",
    "render_markdown",
    "split_paragraphs",
    "split_words",
    "write_transcript",
]
