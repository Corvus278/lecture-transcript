"""Точки входа стадий `audio` и `frames` для оркестратора (design D8).

Оркестратор (`lecture_transcript.cli`) сам решает, считать стадию или взять
её из кэша: перед вызовом он проверяет `StageCache.hit()`, а после успешного
вызова фиксирует каталог сборки через `StageCache.save()`. Поэтому здесь не
нужно ни проверок существования артефактов, ни ключей — нужно ровно два
свойства:

1. все файлы стадии пишутся **в `ctx.build_dir`** (временный каталог ячейки
   кэша), а не в `out_dir`: только тогда кэш зафиксирует их атомарно и
   переиспользует при повторном прогоне без единого вызова ffmpeg;
2. возвращаемая полезная нагрузка — JSON-совместимая, с путями
   **относительно каталога стадии**, чтобы кэш оставался переносимым
   (`meta.json` переживает перенос каталога кэша).

Разворачивают нагрузку обратно в контракты `audio_artifact` и
`frames_artifact`: им нужен каталог стадии — при промахе это
`ctx.build_dir`, при попадании — каталог ячейки кэша.

Почему входа два, а не один
---------------------------

Аудио и кадры извлекаются из одного файла, но зависят от разных параметров
и стоят разного. Пока они лежали в одной ячейке кэша, ключ этой ячейки был
вынужден включать и `frames_fps`, и `audio_track`: иначе смена дорожки
вернула бы попадание со старым WAV — тихую порчу. В результате смена
аудиодорожки инвалидировала 5500 кадров и заново декодировала 830 МБ
видео, а смена частоты выборки заставляла переизвлекать аудио.

Две стадии — две ячейки, каждая со своим ключом: `audio` зависит только от
`audio_track`/`audio_sample_rate`, `frames` — только от `frames_fps`/`hwaccel`.
Ни один из входов не читает параметры другого, поэтому ячейки инвалидируются
независимо.

Ни одна из стадий не потребляет результат другой, значит в цепочке кэша обе
обязаны быть **корневыми** (`previous=None`): если связать их линейно,
ключ второй включит ключ первой, и независимость ячеек пропадёт.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from ..contracts import AudioArtifact, Frame, FramesArtifact
from ._ffmpeg import MediaProcessingError
from .audio import extract_audio
from .frames import extract_frames
from .probe import probe

logger = logging.getLogger(__name__)

__all__ = [
    "run_audio_stage",
    "run_frames_stage",
    "run_stage",
    "AUDIO_NAME",
    "FRAMES_DIR_NAME",
    "audio_artifact",
    "frames_artifact",
]

#: Имена артефактов внутри каталога своей стадии.
AUDIO_NAME = "audio.wav"
FRAMES_DIR_NAME = "frames"


class StageContext(Protocol):
    """То, что оркестратор передаёт стадии (структурно — `cli.StageContext`)."""

    source: Path
    out_dir: Path
    build_dir: Path
    config: Any


def _build_dir(ctx: StageContext) -> Path:
    build_dir = Path(ctx.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def run_audio_stage(ctx: StageContext) -> dict[str, Any]:
    """Стадия `audio`: нормализованное аудио WAV 16 кГц моно.

    Зависит только от `media_ingest.audio_track` и
    `media_ingest.audio_sample_rate`. Отсутствие аудиодорожки —
    `MissingAudioTrackError`: транскрипт речи получить невозможно.
    """
    cfg = ctx.config.media_ingest
    build_dir = _build_dir(ctx)

    media = probe(ctx.source)
    audio = extract_audio(
        media,
        build_dir / AUDIO_NAME,
        cfg.audio_track,
        sample_rate=cfg.audio_sample_rate,
    )

    return {
        "source": str(media.path),
        "duration_s": media.duration_s,
        "has_video": media.video is not None,
        "audio": {
            "path": AUDIO_NAME,
            "sample_rate": audio.sample_rate,
            "duration_s": audio.duration_s,
            "source_track_index": audio.source_track_index,
        },
    }


def run_frames_stage(ctx: StageContext) -> dict[str, Any]:
    """Стадия `frames`: кадры видеотрека с таймкодами.

    Зависит только от `media_ingest.frames_fps` и `media_ingest.hwaccel`.
    Отсутствие видеодорожки ошибкой не является: пайплайн продолжает работу
    в аудио-режиме, стадия отдаёт `frames=None` и предупреждает о неполноте
    результата.
    """
    cfg = ctx.config.media_ingest
    build_dir = _build_dir(ctx)

    media = probe(ctx.source)

    frames: FramesArtifact | None = None
    if media.video is None:
        logger.warning(
            "стадия frames: кадры не извлекаются — во входном файле нет "
            "видеодорожки; результат будет неполным (без слайдов и OCR)"
        )
    else:
        frames = extract_frames(
            media,
            build_dir / FRAMES_DIR_NAME,
            fps=cfg.frames_fps,
            hwaccel=cfg.hwaccel,
        )

    return {
        "source": str(media.path),
        "duration_s": media.duration_s,
        "has_video": media.video is not None,
        "frames": None
        if frames is None
        else {
            "directory": FRAMES_DIR_NAME,
            "fps": frames.fps,
            "count": len(frames.frames),
            "suffix": frames.frames[0].path.suffix,
        },
    }


#: Общее имя, которое оркестратор ищет после специфичного `run_frames_stage`.
#: Сохранено ради совместимости: `media_ingest` подключён к стадии `frames`.
run_stage = run_frames_stage


def audio_artifact(payload: dict[str, Any], base_dir: Path | str) -> AudioArtifact:
    """Развернуть нагрузку стадии `audio` обратно в `AudioArtifact`.

    `base_dir` — каталог стадии: `ctx.build_dir` для только что посчитанной
    стадии либо каталог ячейки кэша при попадании.
    """
    data = payload["audio"]
    path = Path(base_dir) / data["path"]
    if not path.is_file():
        raise MediaProcessingError(f"аудиоартефакт стадии не найден: {path}")
    return AudioArtifact(
        path=path,
        sample_rate=int(data["sample_rate"]),
        duration_s=float(data["duration_s"]),
        source_track_index=int(data["source_track_index"]),
    )


def frames_artifact(
    payload: dict[str, Any], base_dir: Path | str
) -> FramesArtifact | None:
    """Развернуть нагрузку стадии `frames` обратно в `FramesArtifact`.

    None — во входном файле не было видеодорожки (аудио-режим).
    Список кадров восстанавливается с диска тем же `sorted(glob(...))`,
    что и при извлечении, поэтому в `meta.json` не хранится 5500 имён.
    """
    data = payload.get("frames")
    if data is None:
        return None
    directory = Path(base_dir) / data["directory"]
    if not directory.is_dir():
        raise MediaProcessingError(f"каталог кадров стадии не найден: {directory}")

    fps = float(data["fps"])
    paths = sorted(directory.glob(f"frame_*{data['suffix']}"))
    expected = int(data["count"])
    if len(paths) != expected:
        logger.warning(
            "в каталоге кадров %s найдено %d файлов вместо %d — "
            "артефакт стадии неполон",
            directory,
            len(paths),
            expected,
        )
    frames = tuple(
        Frame(index=i, timestamp_s=i / fps, path=path) for i, path in enumerate(paths)
    )
    return FramesArtifact(directory=directory, fps=fps, frames=frames)
