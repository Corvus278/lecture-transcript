"""Шаблон итогового `transcript.md` (задача 6.5, design D5).

Структура файла:

    # Транскрипт лекции
    <шапка с метаданными: исходный файл, длительность, бэкенды>

    ## Слайд 12 — Пример 2. — 0:40:00–0:48:00
    > **Пример 2.** $\\sqrt{x^2-x} = -5$
    > ![Слайд 12](slides/012.png)

    <речь абзацами, с промежуточными таймкодами в длинных секциях>

    ## Речь вне слайдов — 0:48:00–0:48:40
    <речь абзацами>

Решения по формату
------------------
* **Ссылка на PNG обязательна и всегда рендерится нами**, а не берётся из
  `SlideOcr.markdown`: путь пересчитывается относительно каталога транскрипта
  (`os.path.relpath`) и записывается через `/`. Абсолютные пути в вывод не
  попадают — иначе файл нельзя перенести вместе с каталогом слайдов.
  Из врезки OCR картинки вырезаются, чтобы ссылка не задвоилась.
* Врезка содержимого слайда — цитата (`> `), в ней Markdown из
  `SlideOcr.markdown` как есть, включая формулы `$...$` (design D5).
* Слайд без OCR или с `unreliable=True` всё равно получает заголовок,
  ссылку на PNG и явную пометку о ненадёжности/отсутствии текста.
* Таймкод ставится перед абзацем, если он начинается не раньше чем через
  `timecode_every_s` секунд после предыдущего таймкода (в начале секции
  таймкодом считается её заголовок). Правило действует и для первого абзаца
  секции: если лектор молчал полслайда, заголовок как ориентир врёт на всю
  эту паузу. В короткой секции таймкодов внутри нет.
* Никаких «сейчас», случайностей и абсолютных путей: вывод зависит только от
  входных данных (задача 6.6).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from ..contracts import Slide, SlideOcr, TranscriptSection
from .consolidate import OCR_LOW_CONFIDENCE_MARK, group_same_content_slides, ocr_body
from .paragraphs import ParagraphPolicy, paragraph_text, split_words

__all__ = [
    "METADATA_LABELS",
    "METADATA_ORDER",
    "TITLE_MAX_CHARS",
    "RenderPolicy",
    "format_timecode",
    "format_interval",
    "render_markdown",
    "slide_title",
]

# Длина темы слайда в заголовке: помещается в строку оглавления редактора
# вместе с номером и интервалом, но не превращает заголовок в абзац.
TITLE_MAX_CHARS = 60

# Формулы `$$...$$` и `$...$` (экранированный `\$` формулу не открывает).
_TITLE_FORMULA = re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$.*?(?<!\\)\$|(?<!\\)\$")
# Разметка выделения (неэкранированная).
_TITLE_EMPHASIS = re.compile(r"(?<!\\)(\*\*|__|\*|_)")
# Разметка начала строки: заголовок, цитата, маркер списка.
_TITLE_LEADING_MARKUP = re.compile(r"^\s*(?:#+|>|[-+*])\s+")

# Порядок и подписи полей шапки. Ключи вне списка выводятся после него
# в алфавитном порядке — чтобы вывод оставался детерминированным.
METADATA_ORDER: tuple[str, ...] = (
    "source",
    "duration",
    "asr_backend",
    "ocr_backend",
    "punctuation",
    "slides",
    "words",
)
METADATA_LABELS: dict[str, str] = {
    "source": "Исходный файл",
    "duration": "Длительность",
    "asr_backend": "ASR-бэкенд",
    "ocr_backend": "OCR-бэкенд",
    "punctuation": "Пунктуация",
    "slides": "Слайдов",
    "words": "Слов распознано",
}


@dataclass(frozen=True)
class RenderPolicy:
    """Параметры шаблона.

    timecode_every_s: порог, с которого внутри секции ставятся промежуточные
        таймкоды (`TranscriptAssemblyConfig.timecode_every_s`, дефолт 120 с).
    title: заголовок первого уровня.
    group_same_content_slides: показывать подряд идущие слайды с одинаковым
        содержимым OCR одной секцией (см. `consolidate`, правило 2).
    """

    timecode_every_s: float = 120.0
    title: str = "Транскрипт лекции"
    group_same_content_slides: bool = True


# --------------------------------------------------------------------------
# Таймкоды
# --------------------------------------------------------------------------


def format_timecode(seconds: float, *, with_hours: bool = False) -> str:
    """`MM:SS`, либо `H:MM:SS` при `with_hours` или длительности от часа."""
    total = int(max(0.0, seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours or with_hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_interval(start_s: float, end_s: float) -> str:
    """Интервал секции: `0:40:00–0:48:00` (короткое тире — en dash).

    Формат времени во всём файле один — `H:MM:SS`, как у промежуточных
    таймкодов и длительности в шапке: запись лекции длиннее часа, а `01:00`
    без часов легко прочитать как час.
    """
    left = format_timecode(start_s, with_hours=True)
    right = format_timecode(end_s, with_hours=True)
    return f"{left}–{right}"


# --------------------------------------------------------------------------
# Пути
# --------------------------------------------------------------------------


def relative_image_path(image_path: Path, output_dir: Path | None) -> str:
    """Путь к PNG относительно каталога транскрипта, всегда через `/`."""
    if output_dir is None:
        return PurePosixPath(image_path.as_posix()).as_posix()
    try:
        relative = os.path.relpath(image_path, output_dir)
    except ValueError:  # разные тома в Windows — оставляем как есть
        return image_path.as_posix()
    return PurePosixPath(Path(relative).as_posix()).as_posix()


def _markdown_link(text: str, target: str) -> str:
    if any(ch in target for ch in " ()"):
        target = f"<{target}>"
    return f"![{text}]({target})"


# --------------------------------------------------------------------------
# Рендер
# --------------------------------------------------------------------------


def render_markdown(
    sections: Sequence[TranscriptSection],
    *,
    metadata: Mapping[str, str] | None = None,
    output_dir: Path | None = None,
    policy: RenderPolicy | None = None,
    paragraph_policy: ParagraphPolicy | None = None,
) -> str:
    """Собрать текст `transcript.md`. Файл всегда оканчивается переводом строки."""
    policy = policy or RenderPolicy()
    lines: list[str] = [f"# {policy.title}", ""]
    lines.extend(_render_metadata(metadata or {}))

    groups = (
        group_same_content_slides(sections)
        if policy.group_same_content_slides
        else tuple((section,) for section in sections)
    )
    for group in groups:
        lines.extend(_render_group(group, policy, paragraph_policy, output_dir))

    text = "\n".join(lines).rstrip("\n")
    return text + "\n"


def _render_metadata(metadata: Mapping[str, str]) -> list[str]:
    if not metadata:
        return []
    known = [key for key in METADATA_ORDER if key in metadata]
    extra = sorted(key for key in metadata if key not in METADATA_ORDER)
    lines = [
        f"- **{METADATA_LABELS.get(key, key)}:** {metadata[key]}"
        for key in (*known, *extra)
    ]
    lines.append("")
    return lines


def _render_group(
    group: Sequence[TranscriptSection],
    policy: RenderPolicy,
    paragraph_policy: ParagraphPolicy | None,
    output_dir: Path | None,
) -> list[str]:
    """Группа слайдов с одинаковым содержимым — одна секция вывода.

    Один заголовок (диапазон номеров), интервал — объединение, врезка OCR —
    от последнего слайда (D4: последний кадр полнее), речь подряд. Ссылки —
    на все различающиеся по содержимому PNG группы в порядке времени:
    одинаковый OCR не значит одинаковый слайд, рукописное OCR выбрасывает
    (D5). Побайтово одинаковые PNG показываются один раз.
    """
    if len(group) == 1:
        return _render_section(group[0], policy, paragraph_policy, output_dir)
    last = group[-1]
    merged = TranscriptSection(
        start_s=min(section.start_s for section in group),
        end_s=max(section.end_s for section in group),
        slide=last.slide,
        slide_ocr=last.slide_ocr,
        words=tuple(word for section in group for word in section.words),
    )
    images = _distinct_images([s.slide for s in group if s.slide is not None])
    first = group[0].slide.index if group[0].slide is not None else None
    return _render_section(
        merged, policy, paragraph_policy, output_dir, images=images, first_index=first
    )


def _distinct_images(slides: Sequence[Slide]) -> tuple[Slide, ...]:
    """Слайды с различающимися PNG, в исходном порядке; дубль — первый."""
    seen: set[tuple[str, str]] = set()
    result: list[Slide] = []
    for slide in slides:
        key = _image_key(slide.image_path)
        if key in seen:
            continue
        seen.add(key)
        result.append(slide)
    return tuple(result)


def _image_key(path: Path) -> tuple[str, str]:
    """Идентичность PNG — по содержимому файла, не по имени. Файл не
    читается — сравнивать не с чем, ключ по пути (дубль не схлопнется)."""
    try:
        return ("sha256", hashlib.sha256(Path(path).read_bytes()).hexdigest())
    except OSError:
        return ("path", Path(path).as_posix())


def _render_section(
    section: TranscriptSection,
    policy: RenderPolicy,
    paragraph_policy: ParagraphPolicy | None,
    output_dir: Path | None,
    *,
    images: Sequence[Slide] | None = None,
    first_index: int | None = None,
) -> list[str]:
    lines: list[str] = [_section_heading(section, first_index), ""]

    if section.slide is not None:
        lines.extend(
            _render_slide_block(
                section.slide,
                section.slide_ocr,
                output_dir,
                images if images is not None else (section.slide,),
            )
        )

    lines.extend(_render_speech(section, policy, paragraph_policy))
    return lines


def _section_heading(section: TranscriptSection, first_index: int | None = None) -> str:
    interval = format_interval(section.start_s, section.end_s)
    if section.slide is None:
        return f"## Речь вне слайдов — {interval}"
    label = f"Слайд {section.slide.index}"
    if first_index is not None and first_index != section.slide.index:
        label = f"Слайды {first_index}–{section.slide.index}"
    title = slide_title(section.slide_ocr)
    if title:
        return f"## {label} — {title} — {interval}"
    return f"## {label} — {interval}"


def slide_title(ocr: SlideOcr | None, max_chars: int = TITLE_MAX_CHARS) -> str:
    """Первая текстовая строка OCR для заголовка секции — навигация по теме.

    Из строки убираются формулы, пометки неуверенности и разметка выделения;
    строка, в которой после этого не осталось букв или цифр (чистая формула),
    пропускается. Экранирование slide-ocr (`\\$`, `\\*`) сохраняется: в
    заголовке оно отрисуется как обычный символ. Длинная строка обрезается
    по границе слова до `max_chars` с многоточием.
    """
    if ocr is None:
        return ""
    for line in ocr_body(ocr.markdown).splitlines():
        text = line.replace(OCR_LOW_CONFIDENCE_MARK, " ")
        text = _TITLE_FORMULA.sub(" ", text)
        text = _TITLE_EMPHASIS.sub("", text)
        text = _TITLE_LEADING_MARKUP.sub("", text)
        text = " ".join(text.split())
        if not any(ch.isalnum() for ch in text):
            continue
        if len(text) > max_chars:
            cut = text[:max_chars]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            text = cut.rstrip(" ,;:—–-\\") + "…"
        return text
    return ""


def _render_slide_block(
    slide: Slide,
    ocr: SlideOcr | None,
    output_dir: Path | None,
    images: Sequence[Slide] = (),
) -> list[str]:
    body: list[str] = []
    if ocr is None:
        body.append("_Содержимое слайда не распознано — смотрите изображение._")
    else:
        content = ocr_body(ocr.markdown)
        # Пометка о ненадёжности ставится до проверки на пустоту: «OCR не
        # отработал» и «OCR отработал и сам себе не доверяет» — разные
        # состояния, и оба должны быть обозначены явно.
        if ocr.unreliable:
            body.append("_Распознавание ненадёжно, сверьтесь с изображением слайда._")
            body.append("")
        if not content:
            body.append("_Текст на слайде не распознан — смотрите изображение._")
        else:
            body.extend(line.rstrip() for line in content.splitlines())

    for image in images or (slide,):
        body.append("")
        body.append(
            _markdown_link(
                f"Слайд {image.index}", relative_image_path(image.image_path, output_dir)
            )
        )

    quoted = [f"> {line}".rstrip() for line in body]
    quoted.append("")
    return quoted


def _render_speech(
    section: TranscriptSection,
    policy: RenderPolicy,
    paragraph_policy: ParagraphPolicy | None,
) -> list[str]:
    # Группы пересчитываются той же чистой функцией, что заполняла
    # `section.paragraphs`: тексты совпадают, но здесь нужны ещё и таймкоды.
    groups = split_words(section.words, paragraph_policy)
    if not groups:
        return ["_Речи в этом интервале нет._", ""]

    lines: list[str] = []
    last_timecode = section.start_s
    for group in groups:
        if not group:
            continue
        start_s = group[0].start_s
        # Первый абзац секции тоже получает таймкод, если речь началась много
        # позже начала слайда: иначе для длинного молчаливого слайда время
        # абзаца определяется только интервалом в заголовке, с ошибкой во всю
        # длительность секции.
        if start_s - last_timecode >= policy.timecode_every_s:
            lines.append(f"**[{format_timecode(start_s, with_hours=True)}]**")
            lines.append("")
            last_timecode = start_s
        text = paragraph_text(group)
        if text:
            lines.append(text)
            lines.append("")
    return lines
