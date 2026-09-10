#!/usr/bin/env python3
"""Сухой сквозной прогон пайплайна на эталонной записи (аналог задач 7.1, 7.5).

Это настоящий `lecture_transcript.cli.main` на всей записи. Подменены только
три модельных бэкенда: они зарегистрированы в штатных реестрах и выбраны
штатными флагами/конфигом:

* OCR слайдов   -> ``dry-run-ocr``  (реестр ``slide_ocr.registry``); текст
  и уверенность выводятся из содержимого PNG (хэш грубой миниатюры), а не из
  номера слайда — одинаковые кропы получают одинаковый текст, как с настоящим
  OCR;
* ASR           -> ``dry-run-asr``  (реестр ``speech_transcription.registry``,
  наследник ``IntervalAsrBackend`` — нарезка по VAD, дробление длинных
  интервалов и пересчёт таймкодов в абсолютные идут настоящим кодом);
* пунктуатор    -> фейк под именем штатного ``sbert_punc_case_ru``
  (стадия ``punctuation`` берёт пунктуатор по умолчанию, отдельного флага нет);
* VAD — штатный энергетический (``vad_backend: energy``), silero не нужен.

Что сухой прогон ПРОВЕРЯЕТ по-настоящему
----------------------------------------
* извлечение WAV и кадров ffmpeg на всей записи, их длительность и объём;
* детект области демонстрации, дедупликацию и кропы PNG слайдов;
* энергетический VAD и нарезку речи на куски для ASR;
* граф стадий, кэш (промах, попадание, публикация `merge` из кэша), выгрузку
  бэкендов между стадиями;
* стыковку контрактов между стадиями на реальном объёме данных;
* слияние речи со слайдами, допуск на границах, чистку филлеров, абзацы,
  промежуточные таймкоды и рендер `transcript.md`: структуру секций, врезки
  слайдов, пометки ненадёжности, ссылки на PNG;
* время CPU-стадий и пиковую RAM процесса; детерминированность вывода.

Что сухой прогон принципиально НЕ проверяет
-------------------------------------------
* качество OCR (PaddleOCR, pix2tex) и ASR (GigaAM, Whisper), пунктуатора —
  текст транскрипта синтетический;
* реальные таймкоды слов: фейковый ASR раскладывает слова равномерно
  (~2.5 слова/с) внутри интервалов VAD, настоящая речь неравномерна;
* поведение silero-VAD (энергетический VAD режет речь иначе);
* загрузку/выгрузку весов, VRAM, время моделей на GPU (задачи 7.2–7.4);
* пороги из Open Questions (7.6) — они зависят от настоящих confidence
  и настоящих пауз;
* nvdec: на macOS декод идёт через videotoolbox/CPU.

Использование::

    .venv/bin/python tools/dry_run.py --out-dir /путь/вне/репо/out \\
        [--cache-dir /путь/вне/репо/cache] [--input запись.mp4] [--no-cache]

Путь к записи по умолчанию — ``LECTURE_REFERENCE_MP4`` либо
``~/Downloads/wr_20260909_1150.mp4``. Выход и кэш внутри репозитория
запрещены: артефакты полной записи весят сотни мегабайт.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]

try:  # editable-установка в .venv; иначе — из исходников
    import lecture_transcript  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from lecture_transcript import cli
from lecture_transcript.contracts import Availability, Glossary, OcrFragment, Rect, Word
from lecture_transcript.slide_ocr import registry as ocr_registry
from lecture_transcript.slide_ocr.assemble import RecognizedFragments
from lecture_transcript.speech_transcription import punctuation as punctuation_mod
from lecture_transcript.speech_transcription import registry as asr_registry
from lecture_transcript.speech_transcription.base import IntervalAsrBackend

DEFAULT_REFERENCE = Path("~/Downloads/wr_20260909_1150.mp4")
FAKE_OCR_NAME = "dry-run-ocr"
FAKE_ASR_NAME = "dry-run-asr"

# --------------------------------------------------------------------------
# Фейковый OCR
# --------------------------------------------------------------------------

#: Миниатюра для хэша содержимого: 32x18 (16:9), оттенки серого, 6 уровней.
#: Перед уменьшением кроп обрезается до светлой карточки слайда, чтобы рамка
#: плитки и сдвиг окна на несколько пикселей не меняли хэш. Подобрано на
#: 53 кропах эталона: побайтово одинаковые 019/020 и почти одинаковые 021/022
#: совпадают, среди разных слайдов совпадений нет (пороги 180 и 200, 6 и 8
#: уровней дают то же). Смену раскладки окна с изменением масштаба (кропы
#: 1360 против 1272, «Итоги лекции» 046-049) хэш НЕ переживает — эти слайды
#: получают разный текст.
CONTENT_THUMB_SIZE = (32, 18)
CONTENT_LEVELS = 6
CARD_BRIGHTNESS = 180
CARD_MIN_FILL = 0.5

#: Профили слайдов по хэшу содержимого — чтобы в выводе встретились состояния
#: врезки, которые умеет сборка slide_ocr. Пустого профиля нет: у каждого
#: слайда есть заголовок и формула, иначе разные слайды с пустым OCR выглядели
#: бы «одинаковыми по содержимому».
OCR_PROFILES = {
    0: "надёжный: заголовок, формула, текст",
    1: "формула под порогом -> пометка (?) у фрагмента, слайд надёжный",
    2: "два фрагмента из трёх под порогом -> слайд ненадёжный",
    3: "надёжный, текст с символами разметки (экранирование)",
    4: "бэкенд отбросил строку (как гибрид рукописное) -> ненадёжный",
    5: "надёжный: заголовок, формула, текст",
}


def content_digest(image_path: Path) -> str:
    """Хэш содержимого кропа, устойчивый к рамке и сдвигу окна на пару пикселей."""
    with Image.open(image_path) as image:
        gray = image.convert("L")
    pixels = np.asarray(gray)
    bright = pixels > CARD_BRIGHTNESS
    rows = np.flatnonzero(bright.mean(axis=1) > CARD_MIN_FILL)
    cols = np.flatnonzero(bright.mean(axis=0) > CARD_MIN_FILL)
    if rows.size >= 10 and cols.size >= 10:
        gray = gray.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))
    thumb = np.asarray(gray.resize(CONTENT_THUMB_SIZE, Image.Resampling.BOX), dtype=np.uint16)
    quantized = (thumb * CONTENT_LEVELS // 256).astype(np.uint8)
    return hashlib.sha1(quantized.tobytes()).hexdigest()


class FakeOcrBackend:
    """Детерминированный OCR: фрагменты зависят только от содержимого PNG."""

    name = FAKE_OCR_NAME

    def __init__(self) -> None:
        self.calls = 0
        self.unloads = 0

    def check_availability(self) -> Availability:
        return Availability(True)

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        self.calls += 1
        digest = content_digest(image_path)
        short = digest[:6]
        value = int(digest[6:14], 16)
        profile = value % len(OCR_PROFILES)
        a, b, c = 2 + value % 7, 1 + (value >> 3) % 9, 1 + (value >> 7) % 20
        title = OcrFragment(
            text=f"Слайд-тема {short}",
            kind="text",
            confidence=0.97,
            bbox=Rect(40, 20, 600, 40),
        )
        formula = OcrFragment(
            text=f"$\\frac{{{a}}}{{{b}}}x^2 - {c}x = 0$",
            kind="formula",
            confidence=0.88,
            bbox=Rect(40, 90, 500, 50),
        )
        body = OcrFragment(
            text=f"Найдите все корни уравнения {short}",
            kind="text",
            confidence=0.93,
            bbox=Rect(40, 160, 700, 36),
        )
        if profile == 1:
            formula = OcrFragment(formula.text, "formula", 0.45, formula.bbox)
        elif profile == 2:
            body = OcrFragment(body.text, "text", 0.52, body.bbox)
            formula = OcrFragment(formula.text, "formula", 0.41, formula.bbox)
        elif profile == 3:
            body = OcrFragment(
                f"Цена 100$ за *штуку*, итог | {short}_{c}", "text", 0.95, body.bbox
            )
        elif profile == 4:
            return RecognizedFragments((title, formula, body), dropped=1)
        # порядок выдачи намеренно не сверху вниз: сортировку делает сборка
        return (body, formula, title)

    def unload(self) -> None:
        self.unloads += 1


# --------------------------------------------------------------------------
# Фейковый ASR
# --------------------------------------------------------------------------

WORDS_PER_S = 2.5
WORD_DURATION_S = 0.3

#: Фразы с филлерами и самоповтором, чтобы задействовать чистку 6.3.
#: После каждой фразы идёт уникальный маркер `меткаN` (N — сквозной номер
#: фразы): по нему проверяются полнота и порядок речи в транскрипте.
PHRASES: tuple[tuple[str, ...], ...] = (
    ("итак", "рассмотрим", "следующее", "уравнение"),
    ("ээ", "здесь", "нужно", "найти", "область", "допустимых", "значений"),
    ("ну", "вот", "получаем", "квадратное", "уравнение"),
    ("мы", "мы", "подставим", "корень", "и", "проверим"),
    ("это", "как", "бы", "главный", "шаг", "решения"),
    ("запишем", "ответ", "в", "тетрадь"),
    ("обратите", "внимание", "на", "знак", "перед", "корнем"),
)
MARKER_PREFIX = "метка"


def word_stream() -> Iterator[str]:
    """Бесконечный детерминированный поток слов: фраза, маркер, фраза, ..."""
    for number in itertools.count(1):
        # шаг 3 взаимно прост с 7: соседние фразы никогда не совпадают,
        # поэтому на стыке фраз не возникает случайных повторов n-грамм
        yield from PHRASES[(number * 3) % len(PHRASES)]
        yield f"{MARKER_PREFIX}{number}"


class FakeAsrBackend(IntervalAsrBackend):
    """ASR без модели: слова равномерно внутри каждого куска речи VAD.

    Нарезку интервалов, дробление по `max_chunk_s` и перевод таймкодов в
    абсолютные делает настоящий `IntervalAsrBackend`.
    """

    name = FAKE_ASR_NAME
    provides_punctuation = False  # как GigaAM: нужна стадия punctuation
    supports_glossary = False  # как GigaAM 0.1.0
    max_chunk_s = 20.0  # как GIGAAM_MAX_CHUNK_S

    def __init__(self) -> None:
        super().__init__()
        self._stream: Iterator[str] = word_stream()
        self.chunks = 0
        self.generated = 0
        self.unloads = 0

    def check_availability(self) -> Availability:
        return Availability(True)

    def _prepare(self, sample_rate: int) -> None:
        self._stream = word_stream()
        self.chunks = 0
        self.generated = 0

    def _transcribe_chunk(
        self, samples: np.ndarray, sample_rate: int, glossary: Glossary | None
    ) -> Sequence[Word]:
        self.chunks += 1
        duration = samples.shape[0] / sample_rate
        slot = 1.0 / WORDS_PER_S
        count = int(duration * WORDS_PER_S)
        words = []
        for k in range(count):
            start = round(k * slot + 0.05, 3)
            words.append(
                Word(next(self._stream), start, round(start + WORD_DURATION_S, 3), 0.9)
            )
        self.generated += count
        return words

    def unload(self) -> None:
        self.unloads += 1


# --------------------------------------------------------------------------
# Фейковый пунктуатор
# --------------------------------------------------------------------------

_MARKER_RE = re.compile(rf"\b({MARKER_PREFIX}\d+)\b")


class FakePunctuator:
    """Точка после маркера фразы, запятая после «итак», заглавная в начале
    предложения. Помнит, закрыто ли предложение на конце предыдущего куска:
    `restore_punctuation` отдаёт куски по порядку."""

    name = "dry-run-punctuator"

    def __init__(self) -> None:
        self._sentence_closed = True
        self.chunks = 0

    def check_availability(self) -> Availability:
        return Availability(True)

    def restore(self, text: str) -> str:
        self.chunks += 1
        tokens = text.split(" ")
        out: list[str] = []
        for token in tokens:
            word = token
            if self._sentence_closed and word:
                word = word[0].upper() + word[1:]
                self._sentence_closed = False
            if _MARKER_RE.fullmatch(token):
                word += "."
                self._sentence_closed = True
            elif token == "итак":
                word += ","
            out.append(word)
        return " ".join(out)

    def unload(self) -> None:
        self._sentence_closed = True


# --------------------------------------------------------------------------
# Регистрация и запуск
# --------------------------------------------------------------------------


def register_fakes() -> tuple[FakeOcrBackend, FakeAsrBackend, FakePunctuator]:
    ocr = FakeOcrBackend()
    asr = FakeAsrBackend()
    punctuator = FakePunctuator()
    ocr_registry.register(lambda: ocr, name=FAKE_OCR_NAME)
    asr_registry.register(FAKE_ASR_NAME, lambda: asr, replace=True)
    name = punctuation_mod.DEFAULT_PUNCTUATOR_NAME
    punctuation_mod.register_punctuator(name, lambda: punctuator, replace=True)
    punctuation_mod._INSTANCES.pop(name, None)  # сбросить кэш экземпляра
    return ocr, asr, punctuator


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
    except ValueError:
        return False
    return True


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.environ.get("LECTURE_REFERENCE_MP4") or DEFAULT_REFERENCE),
        help="запись лекции (по умолчанию LECTURE_REFERENCE_MP4 или эталон в ~/Downloads)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="выходной каталог")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="каталог кэша (по умолчанию <out>/.cache)"
    )
    parser.add_argument("--no-cache", action="store_true", help="не читать кэш")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.expanduser()
    cache_dir = args.cache_dir.expanduser() if args.cache_dir else out_dir / ".cache"
    for path in (out_dir, cache_dir):
        if _inside_repo(path):
            print(f"отказ: {path} внутри репозитория {ROOT}", file=sys.stderr)
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir.parent / f"{out_dir.name}.dry_run_config.yaml"
    config_path.write_text(
        "# сгенерировано tools/dry_run.py\n"
        "speech_transcription:\n"
        "  vad_backend: energy\n",
        encoding="utf-8",
    )

    ocr, asr, punctuator = register_fakes()
    cli_argv = [
        str(args.input.expanduser()),
        "--out-dir", str(out_dir),
        "--cache-dir", str(cache_dir),
        "--config", str(config_path),
        "--ocr-backend", FAKE_OCR_NAME,
        "--asr-backend", FAKE_ASR_NAME,
        "--log-level", args.log_level,
    ]
    if args.no_cache:
        cli_argv.append("--no-cache")

    started = time.perf_counter()
    code = cli.main(cli_argv)
    wall = time.perf_counter() - started

    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # на macOS ru_maxrss в байтах, на Linux — в килобайтах
    scale = 1 if sys.platform == "darwin" else 1024
    summary = {
        "exit_code": code,
        "wall_s": round(wall, 2),
        "maxrss_self_mb": round(self_usage.ru_maxrss * scale / 2**20, 1),
        "maxrss_children_mb": round(child_usage.ru_maxrss * scale / 2**20, 1),
        "ocr_calls": ocr.calls,
        "ocr_unloads": ocr.unloads,
        "asr_chunks": asr.chunks,
        "asr_words_generated": asr.generated,
        "asr_unloads": asr.unloads,
        "punctuator_chunks": punctuator.chunks,
        "ocr_profiles": OCR_PROFILES,
    }
    (out_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
