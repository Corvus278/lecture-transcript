"""Интеграция slide-ocr -> transcript-assembly: настоящий `assemble_slide_ocr`
и настоящий рендер транскрипта вместе.

Тесты рендера в `test_transcript_assembly.py` подают `SlideOcr.markdown`,
написанный руками. Здесь `SlideOcr` собирает сама стадия slide-ocr, поэтому
ловятся расхождения на стыке зон — так была найдена двойная пометка
ненадёжности. Импорт двух пакетов допустим в тестах; код стадий друг друга не
импортирует.
"""

from __future__ import annotations

import re
from pathlib import Path

from lecture_transcript.config import TranscriptAssemblyConfig
from lecture_transcript.contracts import OcrFragment, Rect, Slide, Transcription, Word
from lecture_transcript.slide_ocr.assemble import assemble_slide_ocr
from lecture_transcript.transcript_assembly import assemble

REGION = Rect(x=0, y=0, width=1280, height=720)


def make_slide(out: Path, index: int, start_s: float, end_s: float, png: bytes) -> Slide:
    path = out / "slides" / f"slide_{index:03d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return Slide(
        index=index,
        start_s=start_s,
        end_s=end_s,
        region=REGION,
        representative_timestamp_s=end_s,
        image_path=path,
    )


def fragments(confidence: float) -> list[OcrFragment]:
    return [
        OcrFragment(
            text="Линейное уравнение",
            kind="text",
            confidence=confidence,
            bbox=Rect(x=10, y=10, width=400, height=40),
        ),
        OcrFragment(
            text="$\\sqrt{x^2 - 19x} = -5$",
            kind="formula",
            confidence=confidence,
            bbox=Rect(x=10, y=80, width=400, height=60),
        ),
    ]


def words_spec(spec: list[tuple[str, float, float]]) -> tuple[Word, ...]:
    return tuple(Word(text=t, start_s=s, end_s=e) for t, s, e in spec)


def run(out: Path, slides: list[Slide], ocrs, words: tuple[Word, ...]):
    transcription = Transcription(words=words, backend="asr", has_punctuation=True)
    return assemble(
        transcription,
        slides,
        ocrs,
        TranscriptAssemblyConfig(),
        source_path=Path("lecture.mp4"),
        output_dir=out,
    )


WORDS = words_spec(
    [
        ("Решаем", 1.0, 1.5),
        ("уравнение.", 1.6, 2.2),
        ("Демонстрация", 31.0, 31.6),  # провал демонстрации 30–36 с
        ("пропала.", 31.7, 32.3),
        ("Дописали", 37.0, 37.6),
        ("ответ.", 37.7, 38.3),
    ]
)


def test_real_slide_ocr_same_content_different_png_one_section_two_images(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    first = make_slide(out, 19, 0.0, 30.0, b"blank template")
    second = make_slide(out, 20, 36.0, 60.0, b"template + handwritten answer")
    ocrs = [
        assemble_slide_ocr(19, first.image_path, fragments(0.95), backend="hybrid"),
        # Тот же печатный текст, но неуверенный: пометки `_(?)_` и unreliable.
        assemble_slide_ocr(20, second.image_path, fragments(0.30), backend="hybrid"),
    ]
    assert ocrs[1].unreliable and not ocrs[0].unreliable

    transcript = run(out, [first, second], ocrs, WORDS)
    md = transcript.markdown

    # Структура: короткая речь вне слайдов ушла в слайд 19, слайды не слиты.
    assert [s.slide.index for s in transcript.sections if s.slide] == [19, 20]
    assert all(s.slide is not None for s in transcript.sections)
    assert tuple(w for s in transcript.sections for w in s.words) == WORDS

    # Вывод: одна секция, тема в заголовке, оба различающихся PNG.
    assert re.findall(r"^## .*$", md, re.M) == [
        "## Слайды 19–20 — Линейное уравнение — 0:00:00–0:01:00"
    ]
    assert "Речь вне слайдов" not in md
    assert md.index("![Слайд 19](slides/slide_019.png)") < md.index(
        "![Слайд 20](slides/slide_020.png)"
    )
    # Врезка одна, пометка ненадёжности ровно одна и без вложенной цитаты.
    assert len(re.findall(r"^> \$\\sqrt", md, re.M)) == 1
    assert len(re.findall(r"ненадёжн|неуверенн", md, re.I)) == 1
    assert "> >" not in md


def test_real_slide_ocr_same_content_identical_png_shown_once(tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = make_slide(out, 19, 0.0, 30.0, b"identical")
    second = make_slide(out, 20, 36.0, 60.0, b"identical")
    ocrs = [
        assemble_slide_ocr(19, first.image_path, fragments(0.95)),
        assemble_slide_ocr(20, second.image_path, fragments(0.95)),
    ]

    md = run(out, [first, second], ocrs, WORDS).markdown

    assert len(re.findall(r"^## ", md, re.M)) == 1
    assert md.count("![Слайд") == 1


def test_real_slide_ocr_empty_results_are_not_grouped(tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = make_slide(out, 1, 0.0, 30.0, b"graph")
    second = make_slide(out, 2, 36.0, 60.0, b"photo")
    ocrs = [
        assemble_slide_ocr(1, first.image_path, []),
        assemble_slide_ocr(2, second.image_path, []),
    ]

    md = run(out, [first, second], ocrs, WORDS).markdown

    assert re.findall(r"^## .*$", md, re.M) == [
        "## Слайд 1 — 0:00:00–0:00:32",
        "## Слайд 2 — 0:00:36–0:01:00",
    ]
    assert md.count("не распознан") == 2


def test_real_slide_ocr_escaped_text_title_has_no_formula_opener(tmp_path: Path) -> None:
    out = tmp_path / "out"
    slide = make_slide(out, 5, 0.0, 30.0, b"price")
    ocr = assemble_slide_ocr(
        5,
        slide.image_path,
        [OcrFragment(text="Цена 100$ за *штуку*", kind="text", confidence=0.95)],
    )

    md = run(out, [slide], [ocr], WORDS[:2]).markdown

    heading = re.findall(r"^## .*$", md, re.M)[0]
    assert "Цена 100\\$ за" in heading
    assert not re.search(r"(?<!\\)\$", heading)
