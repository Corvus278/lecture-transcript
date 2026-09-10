"""Сохранение кропов слайдов в PNG и сборка списка `Slide`.

Откуда берётся пиксель кропа
----------------------------
Выборка кадров (`media_ingest.extract_frames`) по умолчанию отдаёт JPEG
q=2: 5500 кадров 1080p в PNG заняли бы порядка 10 ГБ против ~1.5 ГБ в
JPEG, а дисперсия и perceptual hash к артефактам сжатия устойчивы. Но
кроп репрезентативного кадра уходит в OCR формул, который к «звону» JPEG
вокруг тонких штрихов как раз чувствителен, да и спека требует PNG.
Поэтому репрезентативный кадр **переизвлекается точечно из исходного
mp4**: один seek на его таймкод, один кадр, кроп фильтром ffmpeg, PNG без
потерь. Это один вызов ffmpeg на слайд — десятки вызовов на запись,
несопоставимо дешевле хранения всей выборки в PNG.

Если исходник недоступен (или ffmpeg не отработал), кроп режется из кадра
выборки — с потерей качества, но без потери самого слайда; такой слайд
отмечается в логе предупреждением.

Имена файлов
------------
`slide_001.png`, `slide_002.png`, … — нумерация по возрастанию времени
начала слайда. Имя зависит только от порядкового номера, а порядок
детерминирован, поэтому повторный прогон с теми же параметрами даёт те же
имена (требование спеки «Стабильность имён при повторном запуске»).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from ..contracts import PipelineError, Rect, Slide
from .dedup import FrameGroup

logger = logging.getLogger(__name__)

__all__ = [
    "SLIDE_FILENAME_TEMPLATE",
    "SlideExtractionError",
    "build_slides",
    "scale_region",
    "validate_slides",
]

#: Шаблон имени файла кропа; %d — порядковый номер слайда, 1-based.
SLIDE_FILENAME_TEMPLATE = "slide_%03d.png"

#: Потолок времени на один вызов ffmpeg (один seek и один кадр), с.
#: Без него зависший ffmpeg вешает стадию целиком.
FFMPEG_TIMEOUT_S = 120.0

_FFMPEG = "ffmpeg"


class SlideExtractionError(PipelineError):
    """Кроп слайда не удалось ни переизвлечь, ни вырезать из кадра выборки."""


def scale_region(region: Rect, *, from_width: int, to_width: int) -> Rect:
    """Пересчитать прямоугольник между разрешениями с сохранением пропорций.

    Нужно, когда кадры выборки извлечены уменьшенными, а кроп режется из
    исходника в полном разрешении.
    """
    if from_width <= 0 or to_width <= 0:
        raise ValueError("ширина кадра должна быть положительной")
    if from_width == to_width:
        return region
    factor = to_width / from_width
    return Rect(
        x=int(round(region.x * factor)),
        y=int(round(region.y * factor)),
        width=max(1, int(round(region.width * factor))),
        height=max(1, int(round(region.height * factor))),
    )


def build_slides(
    groups: Sequence[FrameGroup],
    out_dir: Path | str,
    *,
    source: Path | None = None,
    frame_width: int | None = None,
    filename_template: str = SLIDE_FILENAME_TEMPLATE,
) -> list[Slide]:
    """Сохранить кропы групп в PNG и собрать упорядоченный список слайдов.

    :param groups: группы кадров, в любом порядке — сортируются по времени.
    :param out_dir: каталог для изображений; создаётся при необходимости.
    :param source: исходный mp4 для точечного переизвлечения кадра в PNG.
    :param frame_width: ширина кадров выборки, в координатах которых заданы
        регионы групп. Нужна, только если она отличается от ширины исходника.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(groups, key=lambda g: (g.start_s, g.end_s))
    _drop_stale(out_dir, filename_template, keep=len(ordered))
    source_width = _source_width(source) if source is not None else None

    slides: list[Slide] = []
    for number, group in enumerate(ordered, start=1):
        image_path = out_dir / (filename_template % number)
        crop_region = group.region
        if source_width is not None and frame_width and frame_width != source_width:
            crop_region = scale_region(
                group.region, from_width=frame_width, to_width=source_width
            )
        crop_region = _write_crop(
            group=group,
            crop_region=crop_region,
            source=source,
            image_path=image_path,
        )
        slides.append(
            Slide(
                index=number,
                start_s=float(group.start_s),
                end_s=float(group.end_s),
                # Именно тот прямоугольник, по которому вырезан PNG:
                # `region` и `image_path` обязаны быть в одной системе
                # координат, иначе наложение одного на другое даст мусор.
                region=crop_region,
                representative_timestamp_s=float(group.representative.timestamp_s),
                image_path=image_path,
            )
        )
    validate_slides(slides)
    logger.info("слайды сохранены: %s — %d шт.", out_dir, len(slides))
    return slides


def validate_slides(slides: Sequence[Slide]) -> None:
    """Проверить, что интервалы упорядочены, непусты и не пересекаются."""
    previous: Slide | None = None
    for slide in slides:
        if slide.end_s <= slide.start_s:
            raise SlideExtractionError(
                f"слайд {slide.index}: пустой интервал "
                f"[{slide.start_s}, {slide.end_s})"
            )
        if previous is not None:
            if slide.index != previous.index + 1:
                raise SlideExtractionError(
                    f"нумерация слайдов не сплошная: {previous.index} -> {slide.index}"
                )
            if slide.start_s < previous.end_s:
                raise SlideExtractionError(
                    f"интервалы слайдов {previous.index} и {slide.index} "
                    f"пересекаются: {previous.end_s} > {slide.start_s}"
                )
        if not (slide.start_s <= slide.representative_timestamp_s < slide.end_s):
            raise SlideExtractionError(
                f"слайд {slide.index}: репрезентативный кадр "
                f"{slide.representative_timestamp_s} вне интервала"
            )
        previous = slide


# --------------------------------------------------------------------------
# Внутреннее
# --------------------------------------------------------------------------


def _drop_stale(out_dir: Path, filename_template: str, *, keep: int) -> None:
    """Удалить кропы прошлого прогона с номерами больше `keep`.

    Если повторный прогон дал меньше слайдов, оставшиеся от прошлого раза
    `slide_NNN.png` выглядят как настоящие слайды — и в транскрипт уходят
    ссылки на них.
    """
    number = keep + 1
    while True:
        stale = out_dir / (filename_template % number)
        if not stale.exists():
            break
        stale.unlink()
        logger.info("удалён устаревший кроп прошлого прогона: %s", stale.name)
        number += 1


def _write_crop(
    *,
    group: FrameGroup,
    crop_region: Rect,
    source: Path | None,
    image_path: Path,
) -> Rect:
    """Записать PNG-кроп репрезентативного кадра группы.

    :returns: прямоугольник, по которому кроп фактически вырезан, — он и
        уходит в `Slide.region`, чтобы координаты соответствовали картинке.
    """
    timestamp_s = group.representative.timestamp_s
    if source is not None and shutil.which(_FFMPEG):
        grabbed = _grab_png(source, timestamp_s, crop_region, image_path)
        if grabbed is not None:
            return grabbed
        logger.warning(
            "не удалось переизвлечь кадр t=%.2f из %s, режу кроп из кадра выборки"
            " (JPEG-артефакты попадут в OCR)",
            timestamp_s,
            source,
        )
    _crop_frame_png(group.representative.path, group.region, image_path)
    return group.region


def _grab_png(
    source: Path,
    timestamp_s: float,
    region: Rect,
    image_path: Path,
    *,
    timeout_s: float = FFMPEG_TIMEOUT_S,
) -> Rect | None:
    """Один кадр исходника на заданном таймкоде, обрезанный, в PNG.

    :returns: прямоугольник, по которому кадр фактически вырезан (после
        клампа по размеру исходника), либо `None`, если ffmpeg не отработал.

    Фильтр `crop` ffmpeg не падает, если окно вылезает за границы кадра, —
    он молча сдвигает его внутрь, и PNG перестаёт соответствовать
    `Slide.region`. Поэтому окно клампится по реальному размеру исходника
    заранее, а расхождение попадает в лог.
    """
    size = _source_size(source)
    if size is not None:
        clamped = _clamp_region(region, *size)
        if clamped != region:
            logger.warning(
                "область кропа %dx%d+%d+%d выходит за кадр %dx%d,"
                " обрезана до %dx%d+%d+%d",
                region.width, region.height, region.x, region.y,
                size[0], size[1],
                clamped.width, clamped.height, clamped.x, clamped.y,
            )
        region = clamped
    args = [
        _FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{timestamp_s:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"crop={region.width}:{region.height}:{region.x}:{region.y}",
        "-pix_fmt",
        "rgb24",
        str(image_path),
    ]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "ffmpeg не уложился в %.0f с на кадре t=%.2f из %s",
            timeout_s,
            timestamp_s,
            source,
        )
        return None
    if result.returncode != 0 or not image_path.exists():
        logger.debug("ffmpeg: %s", (result.stderr or "").strip()[-400:])
        return None
    return region


def _clamp_region(region: Rect, width: int, height: int) -> Rect:
    """Прямоугольник, гарантированно лежащий внутри кадра width×height."""
    x = max(0, min(region.x, width - 1))
    y = max(0, min(region.y, height - 1))
    return Rect(
        x=x,
        y=y,
        width=max(1, min(region.width, width - x)),
        height=max(1, min(region.height, height - y)),
    )


def _crop_frame_png(frame_path: Path, region: Rect, image_path: Path) -> None:
    """Запасной путь: кроп из кадра выборки средствами Pillow."""
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(frame_path) as image:
            box = (
                region.x,
                region.y,
                region.x + region.width,
                region.y + region.height,
            )
            image.convert("RGB").crop(box).save(image_path, format="PNG")
    except OSError as exc:  # pragma: no cover — битый кадр выборки
        raise SlideExtractionError(
            f"не удалось сохранить кроп слайда из {frame_path}: {exc}"
        ) from exc


def _source_size(source: Path) -> tuple[int, int] | None:
    """Размер видеопотока исходника (ширина, высота); None, если не открылся."""
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height) if width and height else None
    finally:
        capture.release()


def _source_width(source: Path) -> int | None:
    """Ширина видеопотока исходника; None, если файл не открылся."""
    size = _source_size(source)
    return size[0] if size else None
