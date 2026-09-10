"""Стадия `merge` целиком: слияние, чистка, абзацы, рендер, запись файла.

Порядок стадий (он же порядок проверок):

    merge_sections   — раскладка слов по секциям, инвариант полноты 6.1
    consolidate_sections — короткая речь вне слайдов и повторы одного слайда
                       сливаются с соседями (полнота 6.1 сохраняется)
    cleanup_sections — консервативная чистка филлеров 6.3 (уже после 6.1)
    fill_paragraphs  — разбиение на абзацы 6.4
    render_markdown  — шаблон transcript.md 6.5

Детерминированность (6.6): стадия не читает часы, не использует случайность,
не обходит неупорядоченные множества при формировании вывода и не печатает
абсолютные пути. Два прогона на одних входных данных дают побайтово
идентичный файл.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ..config import TranscriptAssemblyConfig
from ..contracts import Slide, SlideOcr, Transcript, Transcription
from .cleanup import CleanupPolicy, cleanup_sections
from .consolidate import ConsolidationPolicy, consolidate_sections
from .merge import BoundaryPolicy, merge_sections
from .paragraphs import ParagraphPolicy, fill_paragraphs
from .render import RenderPolicy, format_timecode, render_markdown

__all__ = [
    "assemble",
    "write_transcript",
    "policies_from_config",
    "consolidation_policy_from_config",
]


def policies_from_config(
    config: TranscriptAssemblyConfig | None,
) -> tuple[BoundaryPolicy, CleanupPolicy, ParagraphPolicy, RenderPolicy]:
    """Развернуть секцию конфига в политики отдельных шагов."""
    config = config or TranscriptAssemblyConfig()
    return (
        BoundaryPolicy(tolerance_s=config.section_boundary_tolerance_s),
        CleanupPolicy(enabled=config.filler_cleanup),
        ParagraphPolicy(pause_s=config.paragraph_pause_s),
        RenderPolicy(timecode_every_s=config.timecode_every_s),
    )


def consolidation_policy_from_config(
    config: TranscriptAssemblyConfig | None,
) -> ConsolidationPolicy:
    """Политика укрупнения секций из конфига.

    Поле `short_offslide_section_s` добавляется в `TranscriptAssemblyConfig`
    отдельно (зона core); пока его нет, берётся дефолт политики.
    """
    default = ConsolidationPolicy()
    return ConsolidationPolicy(
        short_offslide_section_s=getattr(
            config, "short_offslide_section_s", default.short_offslide_section_s
        )
    )


def assemble(
    transcription: Transcription,
    slides: Sequence[Slide],
    slide_ocrs: Sequence[SlideOcr] = (),
    config: TranscriptAssemblyConfig | None = None,
    *,
    source_path: Path,
    output_dir: Path | None = None,
    duration_s: float | None = None,
    extra_metadata: Mapping[str, str] | None = None,
) -> Transcript:
    """Собрать `Transcript` из речи и слайдов.

    `output_dir` нужен, чтобы посчитать пути к PNG слайдов относительно
    каталога транскрипта; если он не задан, пути берутся как есть.
    """
    boundary, cleanup, paragraph, render = policies_from_config(config)

    sections = merge_sections(transcription.words, slides, slide_ocrs, boundary)
    sections = consolidate_sections(
        sections, consolidation_policy_from_config(config)
    )
    sections = cleanup_sections(sections, cleanup)
    sections = fill_paragraphs(sections, paragraph)

    metadata = _build_metadata(
        transcription=transcription,
        slides=slides,
        slide_ocrs=slide_ocrs,
        source_path=source_path,
        duration_s=duration_s,
        extra_metadata=extra_metadata,
    )
    markdown = render_markdown(
        sections,
        metadata=metadata,
        output_dir=output_dir,
        policy=render,
        paragraph_policy=paragraph,
    )
    return Transcript(
        sections=sections,
        markdown=markdown,
        source_path=source_path,
        metadata=metadata,
    )


def write_transcript(
    transcript: Transcript,
    output_dir: Path,
    config: TranscriptAssemblyConfig | None = None,
) -> Path:
    """Записать `transcript.md` в каталог и вернуть путь к нему."""
    config = config or TranscriptAssemblyConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / config.output_filename
    target.write_text(transcript.markdown, encoding="utf-8", newline="\n")
    return target


def _build_metadata(
    *,
    transcription: Transcription,
    slides: Sequence[Slide],
    slide_ocrs: Sequence[SlideOcr],
    source_path: Path,
    duration_s: float | None,
    extra_metadata: Mapping[str, str] | None,
) -> dict[str, str]:
    """Шапка файла. Только имя исходного файла — абсолютный путь не переносим."""
    metadata: dict[str, str] = {"source": source_path.name}
    if duration_s is not None:
        metadata["duration"] = format_timecode(duration_s, with_hours=True)
    if transcription.backend:
        metadata["asr_backend"] = transcription.backend
    ocr_backends = sorted({ocr.backend for ocr in slide_ocrs if ocr.backend})
    if ocr_backends:
        metadata["ocr_backend"] = ", ".join(ocr_backends)
    metadata["punctuation"] = "есть" if transcription.has_punctuation else "нет"
    metadata["slides"] = str(len(slides))
    metadata["words"] = str(len(transcription.words))
    for key in sorted(extra_metadata or {}):
        metadata[key] = str((extra_metadata or {})[key])
    return metadata
