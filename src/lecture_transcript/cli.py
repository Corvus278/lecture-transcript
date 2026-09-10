"""Точка входа `lecture-transcript`: разбор аргументов и оркестрация стадий.

Стадии образуют граф из design D2 (`config.STAGE_DEPENDENCIES`)::

    audio  -> vad ------------------> asr -> punctuation --+
    frames -> slides -> ocr -> глоссарий ^                  +-> merge

`audio` и `frames` — корневые и независимые: параметры одной не инвалидируют
артефакт другой.

Каждая стадия проходит через кэш (design D8): попадание — стадия не считается,
промах — стадия исполняется и результат фиксируется атомарно. Между стадиями
у бэкендов вызывается `unload()` (design D7: одновременно в VRAM живёт не
больше одной модели).

Границы владения: сам CLI не реализует ни одной стадии и ни одного бэкенда.
Реестры бэкендов (`slide_ocr.registry`, `speech_transcription.registry`) и
реализации стадий импортируются **лениво и защищённо** — отсутствующий модуль
логируется как «не реализована», прогон продолжается. Это позволяет каркасу
работать до появления остальных пакетов.

Соглашение о стыковке: пакет стадии экспортирует ``run_stage(ctx)``, где
``ctx`` — `StageContext` ниже; возвращаемое значение попадает в
``ctx.results[<стадия>]`` и в JSON-полезную нагрузку кэша. Файлы стадия пишет
в ``ctx.build_dir``, а пути в нагрузке хранит относительно него; каталог, от
которого их разворачивать, лежит в ``ctx.stage_dirs[<стадия>]`` (каталог
ячейки кэша для уже посчитанных стадий).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .cache import Cache, CacheError
from .config import (
    STAGE_DEPENDENCIES,
    STAGES,
    ConfigError,
    PipelineConfig,
    load_config,
    parse_region,
)
from .contracts import (
    OcrFragment,
    PipelineError,
    Rect,
    Slide,
    SlideOcr,
    Transcription,
    Word,
)
from .logging_setup import LOG_LEVELS, setup_logging, stage_timer

__all__ = ["build_parser", "main", "StageContext"]

LOGGER = logging.getLogger("lecture_transcript.cli")

HWACCELS = ("auto", "cuda", "videotoolbox", "none")
CACHE_IDENTITIES = ("stat", "sample", "content")

# Стадия -> модуль пакета, который её реализует. Функция-точка входа —
# `run_stage(ctx)`; отсутствие модуля или функции не является ошибкой прогона.
_STAGE_MODULES: dict[str, str] = {
    "audio": "lecture_transcript.media_ingest",
    "frames": "lecture_transcript.media_ingest",
    "slides": "lecture_transcript.slide_extraction",
    "ocr": "lecture_transcript.slide_ocr",
    "glossary": "lecture_transcript.slide_ocr",
    "vad": "lecture_transcript.speech_transcription",
    "asr": "lecture_transcript.speech_transcription",
    "punctuation": "lecture_transcript.speech_transcription",
    "merge": "lecture_transcript.transcript_assembly",
}

# Имя функции стадии в модуле: сначала специфичное, потом общее.
_STAGE_ENTRYPOINTS: dict[str, tuple[str, ...]] = {
    "audio": ("run_audio_stage",),
    "frames": ("run_frames_stage", "run_stage"),
    "slides": ("run_slides_stage", "run_stage"),
    "ocr": ("run_ocr_stage", "run_stage"),
    "glossary": ("run_glossary_stage",),
    "vad": ("run_vad_stage",),
    "asr": ("run_asr_stage",),
    "punctuation": ("run_punctuation_stage",),
    "merge": ("run_merge_stage", "run_stage"),
}

# --------------------------------------------------------------------------
# Реестры бэкендов — ленивый и защищённый доступ
# --------------------------------------------------------------------------


def _registry(kind: str):
    """Модуль реестра бэкендов или None, если он ещё не реализован.

    Глушится любое исключение импорта, а не только ImportError: битая
    установка CUDA-библиотек падает `OSError`, и `--help` не должен из-за
    этого переставать работать. Причина уходит в лог на DEBUG.
    """
    module_name = {
        "ocr": "lecture_transcript.slide_ocr.registry",
        "asr": "lecture_transcript.speech_transcription.registry",
    }[kind]
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 — CLI обязан пережить любой отказ
        LOGGER.debug("реестр %s-бэкендов не импортируется: %s", kind, exc)
        return None


def _call_registry(kind: str, func_name: str) -> list[str]:
    registry = _registry(kind)
    if registry is None:
        return []
    func = getattr(registry, func_name, None)
    if not callable(func):
        return []
    try:
        return list(func())
    except Exception as exc:  # noqa: BLE001 — см. _registry
        LOGGER.debug("%s.%s(): %s", kind, func_name, exc)
        return []


def known_backends(kind: str) -> list[str]:
    """Все ЗАРЕГИСТРИРОВАННЫЕ имена бэкендов.

    Именно по этому списку проверяется имя из конфига и `--help`: бэкенд
    может быть зарегистрирован, но недоступен (нет весов, нет torch), и это
    другой класс ошибки, чем опечатка в имени.
    """
    return _call_registry(kind, "list_backend_names")


def available_backends(kind: str) -> list[str]:
    """Имена бэкендов, готовых к работе прямо сейчас (зависимости на месте)."""
    return _call_registry(kind, "available_backend_names")


def _backends_help(kind: str) -> str:
    names = known_backends(kind)
    if not names:
        return "реестр бэкендов ещё не доступен, значение берётся из конфига"
    text = "доступны: " + ", ".join(names)
    ready = available_backends(kind)
    if not ready:
        return text + " (ни один не готов в этом окружении: не установлены зависимости)"
    if set(ready) != set(names):
        return text + "; готовы сейчас: " + ", ".join(ready)
    return text


def _get_backend(kind: str, name: str):
    """Экземпляр бэкенда или None, если реестр/бэкенд недоступны."""
    registry = _registry(kind)
    if registry is None:
        return None
    getter = getattr(registry, "get_backend", None)
    if not callable(getter):
        return None
    try:
        return getter(name)
    except Exception as exc:  # noqa: BLE001 — стадия сама решит, что делать
        LOGGER.warning("не удалось получить %s-бэкенд %r: %s", kind, name, exc)
        return None


def _get_punctuator():
    """Пунктуатор по умолчанию — держатель модели стадии `punctuation` (D7)."""
    try:
        module = importlib.import_module("lecture_transcript.speech_transcription")
        return module.get_punctuator()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("пунктуатор недоступен: %s", exc)
        return None


# Стадия -> как получить объект, держащий модель, чтобы выгрузить её после
# стадии (design D7). Стадии без записи здесь модель в памяти не удерживают:
# VAD грузит silero внутри вызова и отпускает по выходу из функции.
_STAGE_MODEL_HOLDERS: dict[str, Callable[[PipelineConfig], Any]] = {
    "ocr": lambda config: _get_backend("ocr", config.slide_ocr.backend),
    "asr": lambda config: _get_backend(
        "asr", config.speech_transcription.asr_backend
    ),
    "punctuation": lambda config: _get_punctuator(),
}


def _unload(backend: Any) -> None:
    if backend is None:
        return
    unload = getattr(backend, "unload", None)
    if callable(unload):
        try:
            unload()
        except Exception as exc:  # выгрузка не должна ронять прогон
            LOGGER.warning("не удалось выгрузить бэкенд %r: %s", backend, exc)


# --------------------------------------------------------------------------
# Аргументы
# --------------------------------------------------------------------------


#: Значение `--slide-region auto`: сбросить область из конфига на автодетект.
_AUTO_REGION = "auto"


def _region_arg(value: str):
    """Тип аргумента `--slide-region`: "X,Y,W,H" либо "auto"."""
    if value.strip().lower() == _AUTO_REGION:
        return _AUTO_REGION
    try:
        return parse_region(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов CLI."""
    parser = argparse.ArgumentParser(
        prog="lecture-transcript",
        description="Локальный пайплайн: запись лекции (mp4) -> transcript.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Стадии (design D2), в порядке исполнения:\n  "
            + " -> ".join(STAGES)
            + "\n\nOCR-бэкенды: "
            + _backends_help("ocr")
            + "\nASR-бэкенды: "
            + _backends_help("asr")
            + "\n"
        ),
    )
    parser.add_argument("input", type=Path, help="входной файл записи (mp4)")
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="каталог для transcript.md, слайдов и промежуточных артефактов",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML поверх дефолтной конфигурации"
    )

    backends = parser.add_argument_group("выбор бэкендов")
    backends.add_argument(
        "--asr-backend",
        metavar="NAME",
        default=None,
        help="бэкенд распознавания речи; " + _backends_help("asr"),
    )
    backends.add_argument(
        "--ocr-backend",
        metavar="NAME",
        default=None,
        help="бэкенд распознавания слайдов; " + _backends_help("ocr"),
    )

    media = parser.add_argument_group("приём медиа")
    media.add_argument(
        "--frames-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="частота выборки кадров видео, кадр/с",
    )
    media.add_argument(
        "--hwaccel",
        choices=HWACCELS,
        default=None,
        help="аппаратное декодирование ffmpeg",
    )

    slides = parser.add_argument_group("слайды")
    slides.add_argument(
        "--slide-region",
        type=_region_arg,
        default=None,
        metavar="X,Y,W,H",
        help=(
            "область демонстрации вручную, в пикселях кадра; "
            "auto — автодетект (по умолчанию)"
        ),
    )

    cache = parser.add_argument_group("кэш (design D8)")
    cache.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="каталог кэша; по умолчанию <out-dir>/.cache",
    )
    cache.add_argument(
        "--no-cache",
        action="store_true",
        help="не читать кэш (результаты всё равно пишутся на диск)",
    )
    cache.add_argument(
        "--cache-identity",
        choices=CACHE_IDENTITIES,
        default=None,
        help="как определяется идентичность входного файла",
    )

    flow = parser.add_argument_group("управление стадиями")
    flow.add_argument(
        "--from-stage",
        choices=STAGES,
        default=None,
        metavar="STAGE",
        help="начать с этой стадии (предыдущие берутся из кэша)",
    )
    flow.add_argument(
        "--only-stage",
        choices=STAGES,
        default=None,
        metavar="STAGE",
        help="выполнить только эту стадию",
    )

    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="уровень логирования",
    )
    return parser


def apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Наложить значения из командной строки поверх конфигурации."""
    media = config.media_ingest
    if args.frames_fps is not None:
        media = dataclasses.replace(media, frames_fps=args.frames_fps)
    if args.hwaccel is not None:
        media = dataclasses.replace(media, hwaccel=args.hwaccel)

    extraction = config.slide_extraction
    region_arg = getattr(args, "slide_region", None)
    if region_arg == _AUTO_REGION:
        extraction = dataclasses.replace(extraction, region=None)
    elif region_arg is not None:
        extraction = dataclasses.replace(extraction, region=region_arg)

    slide_ocr = config.slide_ocr
    if args.ocr_backend is not None:
        slide_ocr = dataclasses.replace(slide_ocr, backend=args.ocr_backend)

    speech = config.speech_transcription
    if args.asr_backend is not None:
        speech = dataclasses.replace(speech, asr_backend=args.asr_backend)

    cache = config.cache
    if args.cache_dir is not None:
        cache = dataclasses.replace(cache, directory=str(args.cache_dir))
    if args.cache_identity is not None:
        cache = dataclasses.replace(cache, identity=args.cache_identity)

    return dataclasses.replace(
        config,
        media_ingest=media,
        slide_extraction=extraction,
        slide_ocr=slide_ocr,
        speech_transcription=speech,
        cache=cache,
    )


def selected_stages(args: argparse.Namespace) -> tuple[str, ...]:
    """Какие стадии реально исполнять (остальные — только чтение кэша)."""
    if args.only_stage is not None:
        return (args.only_stage,)
    if args.from_stage is not None:
        return STAGES[STAGES.index(args.from_stage) :]
    return STAGES


# --------------------------------------------------------------------------
# Оркестрация
# --------------------------------------------------------------------------


@dataclass
class StageContext:
    """Всё, что нужно стадии: вход, куда писать, конфиг, результаты предыдущих.

    Пути внутри `results` относительны каталога своей стадии, поэтому рядом
    идёт `stage_dirs`: `stage_dirs[<стадия>]` — каталог, относительно которого
    разворачивается `results[<стадия>]`. Для уже посчитанных стадий это
    каталог ячейки кэша, для текущей — её `build_dir`.
    """

    stage: str
    source: Path
    out_dir: Path
    build_dir: Path  # временный каталог стадии; фиксируется кэшем после успеха
    config: PipelineConfig
    results: dict[str, Any] = field(default_factory=dict)
    stage_dirs: dict[str, Path] = field(default_factory=dict)
    backend: Any = None


# --------------------------------------------------------------------------
# Адаптеры стадий: публичный API пакета -> контракт `run_stage(ctx)`
#
# Пакеты стадий отдают доменные функции (`extract_slides`, `recognize_slides`,
# `transcribe_speech`, `assemble`, ...), а не точку входа оркестратора. Ниже —
# тонкая обвязка: развернуть нагрузку кэша в контрактные объекты, вызвать
# функцию пакета, сложить результат в `ctx.build_dir` и вернуть JSON-нагрузку
# с относительными путями. Логика стадий здесь не дублируется.
#
# Как только пакет объявит свой `run_*_stage(ctx)`, он победит адаптер:
# `_stage_callable` сначала ищет точку входа в пакете.
# --------------------------------------------------------------------------


def _rect_json(rect: Rect | None) -> dict[str, int] | None:
    if rect is None:
        return None
    return {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}


def _rect_obj(data: Mapping[str, int] | None) -> Rect | None:
    return None if data is None else Rect(**dict(data))


def _slide_json(slide: Slide, base: Path) -> dict[str, Any]:
    return {
        "index": slide.index,
        "start_s": slide.start_s,
        "end_s": slide.end_s,
        "region": _rect_json(slide.region),
        "representative_timestamp_s": slide.representative_timestamp_s,
        "image": str(Path(slide.image_path).relative_to(base)),
    }


def _slide_obj(data: Mapping[str, Any], base: Path) -> Slide:
    return Slide(
        index=int(data["index"]),
        start_s=float(data["start_s"]),
        end_s=float(data["end_s"]),
        region=_rect_obj(data["region"]),
        representative_timestamp_s=float(data["representative_timestamp_s"]),
        image_path=base / data["image"],
    )


def _ocr_json(ocr: SlideOcr, base: Path) -> dict[str, Any]:
    return {
        "slide_index": ocr.slide_index,
        "image": str(Path(ocr.image_path).relative_to(base)),
        "markdown": ocr.markdown,
        "unreliable": ocr.unreliable,
        "backend": ocr.backend,
        "fragments": [
            {
                "text": f.text,
                "kind": f.kind,
                "confidence": f.confidence,
                "bbox": _rect_json(f.bbox),
                "low_confidence": f.low_confidence,
            }
            for f in ocr.fragments
        ],
    }


def _ocr_obj(data: Mapping[str, Any], base: Path) -> SlideOcr:
    return SlideOcr(
        slide_index=int(data["slide_index"]),
        image_path=base / data["image"],
        markdown=data["markdown"],
        unreliable=bool(data["unreliable"]),
        backend=data.get("backend", ""),
        fragments=tuple(
            OcrFragment(
                text=f["text"],
                kind=f["kind"],
                confidence=float(f["confidence"]),
                bbox=_rect_obj(f["bbox"]),
                low_confidence=bool(f["low_confidence"]),
            )
            for f in data["fragments"]
        ),
    )


def _transcription_json(transcription: Transcription) -> dict[str, Any]:
    return {
        "backend": transcription.backend,
        "has_punctuation": transcription.has_punctuation,
        "used_glossary": transcription.used_glossary,
        "warnings": list(transcription.warnings),
        "words": [
            [w.text, w.start_s, w.end_s, w.confidence] for w in transcription.words
        ],
    }


def _transcription_obj(data: Mapping[str, Any]) -> Transcription:
    return Transcription(
        backend=data["backend"],
        has_punctuation=bool(data["has_punctuation"]),
        used_glossary=bool(data["used_glossary"]),
        warnings=tuple(data.get("warnings", ())),
        words=tuple(
            Word(
                text=w[0],
                start_s=float(w[1]),
                end_s=float(w[2]),
                confidence=None if w[3] is None else float(w[3]),
            )
            for w in data["words"]
        ),
    )


def _module(name: str):
    """Пакет стадии; отсутствие — понятная ошибка, а не ImportError наружу."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise PipelineError(f"пакет {name} недоступен: {exc}") from exc


def _adapt_slides(ctx: StageContext) -> dict[str, Any]:
    media = _module("lecture_transcript.media_ingest")
    extraction = _module("lecture_transcript.slide_extraction")
    frames = media.frames_artifact(ctx.results["frames"], ctx.stage_dirs["frames"])
    if frames is None:
        # Аудио-режим (spec media-ingest): видео нет — ветка слайдов даёт
        # пустой, но корректный результат, речь обрабатывается как обычно.
        # Предупреждение о неполноте — одно на прогон, в конце run_pipeline.
        LOGGER.debug("стадия slides: видеодорожки нет — список слайдов пуст")
        return {"slides": []}
    cfg = ctx.config.slide_extraction
    # Пороги (max_spread, min_brightness, min_absence_s, hash_size, ...) стадия
    # читает из `config` сама; область — отдельный аргумент контракта.
    slides = extraction.extract_slides(
        frames,
        ctx.build_dir / "slides",
        source=ctx.source,
        config=cfg,
        region=None if cfg.region is None else Rect(*cfg.region),
    )
    return {"slides": [_slide_json(s, ctx.build_dir) for s in slides]}


def _adapt_ocr(ctx: StageContext) -> dict[str, Any]:
    ocr_pkg = _module("lecture_transcript.slide_ocr")
    cfg = ctx.config.slide_ocr
    slides = [
        _slide_obj(item, ctx.stage_dirs["slides"])
        for item in ctx.results["slides"]["slides"]
    ]
    if not slides:
        # Нечего распознавать — бэкенд OCR не трогаем вовсе, даже проверкой
        # доступности: файлу без видео PaddleOCR не нужен.
        return {"ocr": []}
    recognized = ocr_pkg.recognize_slides(
        slides,
        backend_name=cfg.backend,
        config=ocr_pkg.AssembleConfig(
            low_confidence_threshold=cfg.confidence_threshold,
            unreliable_ratio=cfg.unreliable_fragment_ratio,
        ),
    )
    # пути к PNG остаются в ячейке стадии `slides` — она предшественница ocr,
    # поэтому её ключ уже входит в ключ этой стадии.
    base = ctx.stage_dirs["slides"]
    return {"ocr": [_ocr_json(item, base) for item in recognized]}


def _adapt_glossary(ctx: StageContext) -> dict[str, Any]:
    ocr_pkg = _module("lecture_transcript.slide_ocr")
    cfg = ctx.config.slide_ocr
    recognized = [
        _ocr_obj(item, ctx.stage_dirs["slides"]) for item in ctx.results["ocr"]["ocr"]
    ]
    if not recognized:
        # пустой глоссарий -> ASR штатно идёт без него
        return {"terms": []}
    glossary = ocr_pkg.build_glossary(
        recognized,
        ocr_pkg.GlossaryConfig(
            min_term_length=cfg.glossary_min_term_length,
            max_terms=cfg.glossary_max_terms,
            stopwords=tuple(cfg.glossary_stopwords),
        ),
    )
    return {"terms": list(glossary.terms)}


def _recording_duration(results: Mapping[str, Any]) -> float | None:
    """Длительность записи из нагрузки `audio` или `frames` (обе её несут)."""
    for stage in ("audio", "frames"):
        payload = results.get(stage)
        if isinstance(payload, Mapping) and payload.get("duration_s") is not None:
            return float(payload["duration_s"])
    return None


def _adapt_merge(ctx: StageContext) -> dict[str, Any]:
    assembly = _module("lecture_transcript.transcript_assembly")
    slides_base = ctx.stage_dirs["slides"]
    slides = [_slide_obj(i, slides_base) for i in ctx.results["slides"]["slides"]]
    recognized = [_ocr_obj(i, slides_base) for i in ctx.results["ocr"]["ocr"]]
    transcription = _transcription_obj(ctx.results["punctuation"])

    # Всё, что стадия производит, — transcript.md и PNG слайдов — собирается
    # в её ячейке кэша. В <out-dir> это публикует `_publish_merge`, одинаково
    # при промахе и при попадании: иначе после удаления transcript.md
    # повторный прогон из кэша вернул бы код 0 без результата.
    images_dir = ctx.build_dir / "slides"
    images_dir.mkdir(parents=True, exist_ok=True)
    moved: dict[int, Path] = {}
    for slide in slides:
        target = images_dir / Path(slide.image_path).name
        if Path(slide.image_path).is_file():
            shutil.copy2(slide.image_path, target)
        moved[slide.index] = target
    slides = [
        dataclasses.replace(s, image_path=moved[s.index]) for s in slides
    ]
    recognized = [
        dataclasses.replace(o, image_path=moved.get(o.slide_index, o.image_path))
        for o in recognized
    ]

    transcript = assembly.assemble(
        transcription,
        slides,
        recognized,
        ctx.config.transcript_assembly,
        source_path=ctx.source,
        output_dir=ctx.build_dir,
        duration_s=_recording_duration(ctx.results),
    )
    written = assembly.write_transcript(
        transcript, ctx.build_dir, ctx.config.transcript_assembly
    )
    return {
        "transcript": written.name,
        "sections": len(transcript.sections),
    }


#: Имена PNG слайдов: `slide_%03d.png` из slide_extraction (>999 — длиннее).
_SLIDE_PNG_RE = re.compile(r"^slide_\d{3,}\.png$")


def _publish_merge(payload: Any, cell_dir: Path, out_dir: Path) -> None:
    """Выложить результат `merge` из ячейки кэша в <out-dir>.

    Копируется transcript.md и PNG слайдов. Из <out-dir>/slides убираются
    только осиротевшие файлы по шаблону `slide_NNN.png` — всё прочее, что
    человек положил туда сам, не трогается.
    """
    name = payload.get("transcript") if isinstance(payload, Mapping) else None
    if name:
        produced = cell_dir / name
        if produced.is_file():
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, out_dir / name)

    cell_images = cell_dir / "slides"
    fresh = (
        {p.name: p for p in cell_images.iterdir() if _SLIDE_PNG_RE.match(p.name)}
        if cell_images.is_dir()
        else {}
    )
    images_dir = out_dir / "slides"
    if images_dir.is_dir():
        for stale in images_dir.iterdir():
            if (
                stale.is_file()
                and _SLIDE_PNG_RE.match(stale.name)
                and stale.name not in fresh
            ):
                stale.unlink()
    if fresh:
        images_dir.mkdir(parents=True, exist_ok=True)
        for filename, image in fresh.items():
            shutil.copy2(image, images_dir / filename)


#: Стадия -> публикация результата из ячейки кэша в <out-dir>.
_STAGE_PUBLISHERS: dict[str, Callable[[Any, Path, Path], None]] = {
    "merge": _publish_merge,
}


def _stages_needing_cache(selected: Sequence[str]) -> list[str]:
    """Стадии вне выборки, чей результат выборке нужен, — брать их можно
    только из кэша."""
    chosen = set(selected)
    return sorted(
        {
            parent
            for stage in selected
            for parent in STAGE_DEPENDENCIES.get(stage, ())
            if parent not in chosen
        }
    )


#: Стадия -> адаптер, если пакет не объявил свою точку входа.
_STAGE_ADAPTERS: dict[str, Callable[[StageContext], Any]] = {
    "slides": _adapt_slides,
    "ocr": _adapt_ocr,
    "glossary": _adapt_glossary,
    # vad / asr / punctuation: точки входа run_*_stage объявил сам пакет
    # speech_transcription — дублёр здесь молча разошёлся бы с ними.
    "merge": _adapt_merge,
}


def _stage_callable(stage: str) -> Callable[[StageContext], Any] | None:
    """Функция стадии из её пакета либо None, если стадия ещё не реализована."""
    try:
        module = importlib.import_module(_STAGE_MODULES[stage])
    except ImportError as exc:
        LOGGER.debug("пакет стадии %s не импортируется: %s", stage, exc)
        return None
    for attr in _STAGE_ENTRYPOINTS[stage]:
        func = getattr(module, attr, None)
        if callable(func):
            return func
    # пакет ещё не объявил точку входа — используем адаптер оркестратора
    return _STAGE_ADAPTERS.get(stage)


def _publish_stage(
    stage: str, payload: Any, cell_dir: Path, out_dir: Path, missing: list[str]
) -> None:
    publisher = _STAGE_PUBLISHERS.get(stage)
    if publisher is None:
        return
    try:
        publisher(payload, cell_dir, out_dir)
    except OSError as exc:
        LOGGER.error("стадия %s: не удалось выложить результат в %s: %s", stage, out_dir, exc)
        missing.append(stage)


#: Корень пакета lecture_transcript: пакеты стадий лежат в нём подкаталогами.
_PACKAGE_ROOT = Path(__file__).resolve().parent


def _package_fingerprint(directory: Path) -> str:
    """Отпечаток исходников каталога: .py по содержимому, в стабильном порядке.

    В хэш входит и относительный путь файла (переименование — тоже правка), и
    его байты. Порядок обхода файловой системы не влияет: пути сортируются.
    """
    digest = hashlib.sha256()
    if not directory.exists():
        digest.update(b"<absent>")
        return digest.hexdigest()[:16]
    files = [directory] if directory.is_file() else list(directory.rglob("*.py"))
    base = directory.parent if directory.is_file() else directory
    for path in sorted(files, key=lambda item: item.relative_to(base).as_posix()):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def stage_code_fingerprints(stages: Sequence[str]) -> dict[str, str]:
    """Отпечаток кода каждой стадии — один раз на прогон (design D8).

    Стадия зависит от исходников своего пакета; если вместо точки входа пакета
    работает адаптер оркестратора — ещё и от всего `cli.py`: адаптеры зовут
    общие кодеки нагрузок, и перечислять функции поштучно ненадёжно — забытая
    функция молча отдала бы устаревший результат из кэша.
    """
    memo: dict[Path, str] = {}

    def fingerprint(path: Path) -> str:
        if path not in memo:
            memo[path] = _package_fingerprint(path)
        return memo[path]

    result: dict[str, str] = {}
    for stage in stages:
        module = _STAGE_MODULES.get(stage)
        package = _PACKAGE_ROOT / module.rsplit(".", 1)[-1] if module else None
        parts = [fingerprint(package) if package is not None else "<no-package>"]
        func = _stage_callable(stage)
        if func is not None and func is _STAGE_ADAPTERS.get(stage):
            parts.append(fingerprint(Path(__file__).resolve()))
        result[stage] = "+".join(parts)
    return result


def run_pipeline(args: argparse.Namespace, config: PipelineConfig) -> int:
    """Пройти цепочку стадий. Возвращает код выхода процесса."""
    source = args.input.expanduser()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(config.cache.directory)
    if not cache_dir.is_absolute():
        cache_dir = out_dir / cache_dir

    cache = Cache(
        cache_dir,
        source,
        enabled=not args.no_cache,
        identity=config.cache.identity,
    )
    # Отпечаток кода — часть ключа: правка шаблона вывода обязана
    # инвалидировать сборку, а не молча отдавать прежний transcript.md.
    code = stage_code_fingerprints(STAGES)
    cells = cache.chain(
        (
            (stage, {**config.stage_params(stage), "__code__": code[stage]})
            for stage in STAGES
        ),
        dependencies=STAGE_DEPENDENCIES,
    )

    to_run = set(selected_stages(args))
    results: dict[str, Any] = {}
    # Каталог, относительно которого разворачиваются пути из results.
    stage_dirs: dict[str, Path] = {}
    executed: list[str] = []
    missing: list[str] = []
    backend: Any = None

    LOGGER.info("вход: %s", source)
    LOGGER.info("выход: %s", out_dir)
    LOGGER.info("кэш: %s (%s)", cache_dir, "выключен" if args.no_cache else "включён")

    try:
        for stage in STAGES:
            cell = cells[stage]
            if stage not in to_run:
                if cell.hit():
                    results[stage] = cell.load()
                    stage_dirs[stage] = cell.path
                    LOGGER.info("стадия %s: пропущена, взята из кэша", stage)
                elif args.no_cache:
                    LOGGER.info(
                        "стадия %s: пропущена (чтение кэша выключено флагом --no-cache)",
                        stage,
                    )
                else:
                    LOGGER.info("стадия %s: пропущена (нет в кэше)", stage)
                continue

            if cell.hit():
                results[stage] = cell.load()
                stage_dirs[stage] = cell.path
                LOGGER.info("стадия %s: попадание в кэш (%s)", stage, cell.key)
                _publish_stage(stage, results[stage], cell.path, out_dir, missing)
                continue

            func = _stage_callable(stage)
            if func is None:
                LOGGER.warning("стадия %s: не реализована, пропускаю", stage)
                missing.append(stage)
                continue

            absent = [
                parent
                for parent in STAGE_DEPENDENCIES.get(stage, ())
                if parent not in results
            ]
            if absent:
                LOGGER.error(
                    "стадия %s: нет результата предшественниц (%s) — пропускаю",
                    stage,
                    ", ".join(absent),
                )
                missing.append(stage)
                continue

            holder = _STAGE_MODEL_HOLDERS.get(stage)
            if holder is not None:
                backend = holder(config)
                if backend is None:
                    LOGGER.warning(
                        "стадия %s: держатель модели недоступен, "
                        "стадия получит backend=None",
                        stage,
                    )

            ctx = StageContext(
                stage=stage,
                source=source,
                out_dir=out_dir,
                build_dir=cell.new_build_dir(),
                config=config,
                results=results,
                stage_dirs=stage_dirs,
                backend=backend,
            )
            # Пока стадия считается, её база — каталог сборки; после
            # фиксации той же базой становится каталог ячейки кэша.
            stage_dirs[stage] = ctx.build_dir
            try:
                with stage_timer(stage, LOGGER):
                    payload = func(ctx)
            except (TypeError, AttributeError, KeyError) as exc:
                # API пакета стадии разошёлся с тем, что ожидает адаптер:
                # это дефект стыковки, а не сбой прогона — говорим прямо.
                LOGGER.error(
                    "стадия %s: API пакета не стыкуется с оркестратором (%s: %s); "
                    "нужна точка входа run_%s_stage(ctx) в самом пакете",
                    stage,
                    type(exc).__name__,
                    exc,
                    stage,
                )
                missing.append(stage)
                _unload(backend)
                backend = None
                continue
            stage_dirs[stage] = cell.save(payload, build_dir=ctx.build_dir)
            results[stage] = payload
            executed.append(stage)
            _publish_stage(stage, payload, stage_dirs[stage], out_dir, missing)

            # Выгрузка модели между стадиями обязательна, а не опциональна
            # (D7): в VRAM одновременно живёт не больше одной модели.
            _unload(backend)
            backend = None
    finally:
        _unload(backend)

    LOGGER.info(
        "выполнено стадий: %d из %d", len(executed), len(to_run.intersection(STAGES))
    )
    frames_payload = results.get("frames")
    if isinstance(frames_payload, Mapping) and (
        frames_payload.get("has_video") is False
        or ("frames" in frames_payload and frames_payload["frames"] is None)
    ):
        LOGGER.warning(
            "аудио-режим: во входном файле нет видеодорожки — транскрипт "
            "собран без слайдов и их содержимого, результат неполон"
        )
    if "merge" in to_run and "merge" in results and "merge" not in missing:
        expected = out_dir / config.transcript_assembly.output_filename
        if not expected.is_file():
            LOGGER.error(
                "стадия merge отмечена выполненной, но %s на диске нет "
                "(ячейка кэша повреждена?) — снесите .cache/merge и перезапустите",
                expected,
            )
            missing.append("merge")
    if missing:
        # Прогон, не давший результата, обязан отдать ненулевой код: успех
        # без transcript.md — худший вид тихого отказа.
        LOGGER.error(
            "не выполнены стадии: %s — результат прогона неполон",
            ", ".join(missing),
        )
        return 1
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.from_stage is not None and args.only_stage is not None:
        parser.error("--from-stage и --only-stage взаимоисключающие")
    if args.no_cache:
        needed = _stages_needing_cache(selected_stages(args))
        if needed:
            flag = "--from-stage" if args.from_stage else "--only-stage"
            value = args.from_stage or args.only_stage
            parser.error(
                f"--no-cache несовместим с {flag} {value}: стадии "
                f"{', '.join(needed)} не входят в выборку и могут быть взяты "
                "только из кэша, а чтение кэша выключено. Уберите --no-cache "
                "или расширьте выборку"
            )

    setup_logging(args.log_level)

    try:
        config = load_config(args.config)
        config = apply_overrides(config, args)
        _validate_backend("ocr", config.slide_ocr.backend)
        _validate_backend("asr", config.speech_transcription.asr_backend)
        if not args.input.expanduser().is_file():
            raise PipelineError(f"входной файл не найден: {args.input}")
        return run_pipeline(args, config)
    except (ConfigError, CacheError, PipelineError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("прервано пользователем")
        return 130


def _validate_backend(kind: str, name: str) -> None:
    """Отвергнуть неизвестное имя бэкенда ДО начала прогона (задачи 4.1, 5.1).

    Проверяется по списку ЗАРЕГИСТРИРОВАННЫХ имён, а не готовых к работе:
    иначе на машине без весов любая опечатка проходит валидацию и всплывает
    через минуты декодирования. Псевдонимы разворачиваются реестром.
    """
    names = known_backends(kind)
    if not names:
        return
    registry = _registry(kind)
    resolve = getattr(registry, "resolve_name", None)
    if callable(resolve):
        try:
            resolve(name)
        except Exception:  # noqa: BLE001 — реестр сообщает отказ исключением
            raise PipelineError(
                f"неизвестный {kind}-бэкенд {name!r}; "
                f"зарегистрированы: {', '.join(names)}"
            ) from None
        return
    if name not in names:
        raise PipelineError(
            f"неизвестный {kind}-бэкенд {name!r}; "
            f"зарегистрированы: {', '.join(names)}"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
