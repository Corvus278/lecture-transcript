"""Извлечение кадров видеотрека с заданной частотой выборки.

Решения этого модуля
--------------------

**Формат кадров — JPEG q=2 по умолчанию, PNG опцией.**
Кадры при 1 fps — промежуточная выборка: по ней считается дисперсия для
поиска области демонстрации (D3) и perceptual hash для дедупликации (D4).
Обе операции устойчивы к артефактам JPEG высокого качества, а вот объём
критичен: 5500 кадров 1080p в PNG — порядка 10 ГБ на одну запись, в JPEG
q=2 — порядка 1,5 ГБ. При этом кроп репрезентативного кадра для OCR
дешевле переизвлечь точечно (одна секунда seek по исходнику) в PNG, чем
хранить всю выборку без потерь. Поэтому `image_format="png"` оставлен
параметром для случаев, когда качество всей выборки важнее объёма.

**Временны́е метки — детерминированно из индекса и fps.**
Фильтр `fps=N` строит равномерную сетку от начала записи и подставляет в
каждый её узел ближайший исходный кадр. Значит метка узла `i` равна
`i / fps` по построению, а фактический кадр отстоит от неё не более чем на
половину интервала между исходными кадрами — то есть заведомо лучше
интервала выборки, как того требует спека. Плюс такой способ даёт строгую
монотонность меток по определению, тогда как реальные pts из
`-show_frames` могут повторяться при дублировании кадров и требуют второго
прохода по файлу.

Нулевая точка меток — **начало контейнера**, а не `start_time` видеопотока.
У записей вебинаров видео обычно стартует в нуле или с задержкой в единицы
десятков миллисекунд (у эталонной записи — 0.050 с), что укладывается в
требуемую спекой точность «не хуже интервала выборки» (1 с). Тот же ноль
используется для аудио (см. `audio._ALIGN_FILTER`), поэтому кадры и слова
ASR остаются в одной системе отсчёта.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from ..contracts import Frame, FramesArtifact, MediaInfo, MissingVideoTrackError
from ._ffmpeg import (
    DEFAULT_HWACCEL,
    FFMPEG,
    MediaProcessingError,
    atomic_dir,
    hwaccel_args,
    resolve_hwaccel,
    run,
    stderr_tail,
)

logger = logging.getLogger(__name__)

__all__ = ["extract_frames", "ImageFormat"]

ImageFormat = Literal["jpg", "png"]

_EXTENSION: dict[str, str] = {"jpg": "jpg", "png": "png"}
_FILENAME_TEMPLATE = "frame_%06d.{ext}"


def extract_frames(
    media: MediaInfo,
    out_dir: Path | str,
    fps: float = 1.0,
    *,
    hwaccel: str = DEFAULT_HWACCEL,
    image_format: ImageFormat = "jpg",
    quality: int = 2,
    scale_width: int | None = None,
) -> FramesArtifact:
    """Извлечь кадры видеотрека в `out_dir` с частотой `fps`.

    :param hwaccel: ``auto`` | ``cuda`` | ``videotoolbox`` | ``none``.
        Явно запрошенное и недоступное ускорение — ошибка
        `HwaccelUnavailableError` (без тихого отката на CPU: на 91-минутной
        записи такой откат стоит десятки минут и должен быть заметен).
        В режиме ``auto`` откат на CPU — штатное поведение.
    :param quality: качество JPEG в шкале ffmpeg ``-q:v`` (2 — лучшее
        практическое). Для PNG игнорируется.
    :param scale_width: если задано, кадры масштабируются по ширине с
        сохранением пропорций.

    Имена файлов — ``frame_000000.jpg`` и далее, лексикографический порядок
    совпадает с временны́м. Каталог собирается во временном месте и
    переименовывается целиком: при ошибке `out_dir` не появляется, а
    существовавший ранее каталог остаётся нетронутым.
    """
    out_dir = Path(out_dir)
    if fps <= 0:
        raise ValueError(f"частота выборки должна быть положительной, получено {fps}")
    if image_format not in _EXTENSION:
        raise ValueError(f"неподдерживаемый формат кадров: {image_format!r}")
    if media.video is None:
        raise MissingVideoTrackError(
            f"во входном файле нет видеодорожки: {media.path} — извлекать кадры не из чего"
        )

    accel = resolve_hwaccel(hwaccel)
    ext = _EXTENSION[image_format]
    pattern = _FILENAME_TEMPLATE.format(ext=ext)

    filters = [f"fps={fps}"]
    if scale_width:
        filters.append(f"scale={int(scale_width)}:-2")

    with atomic_dir(out_dir) as tmp_dir:
        _run_extraction(
            media=media,
            tmp_dir=tmp_dir,
            pattern=pattern,
            filters=filters,
            accel=accel,
            image_format=image_format,
            quality=quality,
            allow_cpu_retry=(hwaccel or "").lower() == "auto",
        )
        produced = sorted(tmp_dir.glob(f"frame_*.{ext}"))
        if not produced:
            raise MediaProcessingError(
                f"ffmpeg не извлёк ни одного кадра из {media.path} при fps={fps}"
            )
        names = [p.name for p in produced]

    frames = tuple(
        Frame(index=i, timestamp_s=i / fps, path=out_dir / name)
        for i, name in enumerate(names)
    )
    logger.info(
        "кадры извлечены: %s — %d шт., fps=%s, формат=%s, hwaccel=%s",
        out_dir,
        len(frames),
        fps,
        image_format,
        accel,
    )
    return FramesArtifact(directory=out_dir, fps=float(fps), frames=frames)


def _run_extraction(
    *,
    media: MediaInfo,
    tmp_dir: Path,
    pattern: str,
    filters: list[str],
    accel: str,
    image_format: str,
    quality: int,
    allow_cpu_retry: bool,
) -> None:
    """Один вызов ffmpeg; в режиме ``auto`` — один повтор на CPU при сбое."""
    result = run(
        _build_args(
            media=media,
            pattern=pattern,
            filters=filters,
            accel=accel,
            image_format=image_format,
            quality=quality,
        ),
        cwd=tmp_dir,
    )
    if result.returncode == 0:
        return

    if allow_cpu_retry and accel != "none":
        # hwaccel числится у ffmpeg, но фактически не заработал (нет устройства,
        # неподдерживаемый профиль кодека). В автоматическом режиме это не повод
        # падать, но повод громко предупредить.
        logger.warning(
            "hwaccel=%s не отработал (%s), повтор на CPU", accel, stderr_tail(result, lines=2)
        )
        for stale in tmp_dir.iterdir():
            stale.unlink()
        result = run(
            _build_args(
                media=media,
                pattern=pattern,
                filters=filters,
                accel="none",
                image_format=image_format,
                quality=quality,
            ),
            cwd=tmp_dir,
        )
        if result.returncode == 0:
            return

    raise MediaProcessingError(
        f"не удалось извлечь кадры из {media.path}: {stderr_tail(result)}"
    )


def _build_args(
    *,
    media: MediaInfo,
    pattern: str,
    filters: list[str],
    accel: str,
    image_format: str,
    quality: int,
) -> list[str]:
    """Аргументы ffmpeg. Запускать строго с ``cwd`` = каталог сборки.

    Шаблон имени кадра остаётся относительным: muxer ``image2``
    разворачивает printf-спецификаторы по всему переданному пути, поэтому
    ``%`` в имени любого родительского каталога (``Лекция 100% готово``)
    ломает вывод. Вход, наоборот, приводится к абсолютному пути — рабочий
    каталог процесса другой.
    """
    assert media.video is not None  # проверено в extract_frames
    args = [FFMPEG, "-nostdin", "-hide_banner", "-v", "error", "-y"]
    args += hwaccel_args(accel)
    args += ["-i", str(Path(media.path).resolve()), "-map", f"0:{media.video.index}"]
    args += ["-an", "-sn", "-dn", "-vf", ",".join(filters)]
    if image_format == "jpg":
        args += ["-q:v", str(quality)]
    args += ["-start_number", "0", pattern]
    return args
