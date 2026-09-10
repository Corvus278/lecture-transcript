"""Группировка кадров в логические слайды по perceptual hash (design D4).

Лектор дописывает поверх слайда, поэтому хэш одного логического слайда
плывёт постепенно: каждая дописанная строка меняет его чуть-чуть, а смена
слайда — сразу и сильно. Отсюда правило: **сравниваются соседние кадры**,
и новая группа начинается там, где hamming distance между соседями
перескакивает порог. Сравнение с первым кадром группы такое правило не
заменяет: за десять минут дописывания слайд уходит от своей заготовки
дальше, чем иной слайд от следующего, и группа развалилась бы на середине.

Репрезентативный кадр — **последний** в группе (D4): он содержит наиболее
полное состояние слайда. Наивный scene-detect берёт первый и получает
пустую заготовку без решения.

`hash_size` зафиксирован равным 16: хэш 16×16 = 256 бит. При 8×8 (64 бита)
дописанная строка формулы теряется в шуме квантования — расстояние от
дописывания и от смены слайда перестают разделяться. Порог hamming
задаётся в конфиге (`slide_extraction.hamming_threshold`) и подобран на
эталонной записи в задаче 3.7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..contracts import Frame, Rect
from .layout import DEFAULT_MIN_ABSENCE_S

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_HAMMING_THRESHOLD",
    "DEFAULT_HASH_SIZE",
    "FrameGroup",
    "crop_hash",
    "group_frames",
    "hash_distances",
    "hash_sequence",
]

#: Сторона матрицы perceptual hash. См. docstring модуля.
DEFAULT_HASH_SIZE = 16

#: Порог hamming distance между соседними кадрами, выше которого начинается
#: новый логический слайд (задача 3.7). Измерено на эталонной записи по
#: кропу всей плитки демонстрации: дописывание маркером внутри непрерывного
#: показа даёт до 30, самая слабая настоящая смена слайда — 56 (t=676->681).
#: Любой порог из 31..46 даёт одни и те же 53 группы без единой склейки.
#: Взята середина зазора 30|56 — 43: design D4/Risks требует не заходить
#: выше середины, ошибка в сторону дробления безопаснее ошибки в сторону
#: склейки. Провал демонстрации режет группу всегда, независимо от порога.
#: Значение продублировано в `default_config.yaml`
#: (`slide_extraction.hamming_threshold`); конфиг — источник правды для
#: прогонов, эта константа — для прямых вызовов модуля.
DEFAULT_HAMMING_THRESHOLD = 43


@dataclass(frozen=True)
class FrameGroup:
    """Группа последовательных кадров с одним содержимым области демонстрации."""

    frames: tuple[Frame, ...]
    region: Rect
    start_s: float
    end_s: float

    @property
    def representative(self) -> Frame:
        """Последний кадр группы — самое полное состояние слайда (D4)."""
        return self.frames[-1]

    def __len__(self) -> int:
        return len(self.frames)


def crop_hash(path: Path, region: Rect, *, hash_size: int = DEFAULT_HASH_SIZE):
    """Perceptual hash кропа области демонстрации одного кадра."""
    import imagehash  # noqa: PLC0415 — тяжёлый импорт по требованию
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as image:
        box = (
            region.x,
            region.y,
            region.x + region.width,
            region.y + region.height,
        )
        crop = image.convert("L").crop(box)
        return imagehash.phash(crop, hash_size=hash_size)


def hash_sequence(
    frames: Sequence[Frame], region: Rect, *, hash_size: int = DEFAULT_HASH_SIZE
) -> list:
    """Хэши кропов по последовательности кадров, в том же порядке."""
    return [crop_hash(frame.path, region, hash_size=hash_size) for frame in frames]


def hash_distances(hashes: Sequence) -> list[int]:
    """Hamming distance между соседними хэшами: длина len(hashes) - 1."""
    return [int(hashes[i] - hashes[i - 1]) for i in range(1, len(hashes))]


def group_frames(
    frames: Sequence[Frame],
    region: Rect,
    *,
    hash_size: int = DEFAULT_HASH_SIZE,
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
    step_s: float = 1.0,
    min_group_s: float = 0.0,
    min_absence_s: float = DEFAULT_MIN_ABSENCE_S,
) -> list[FrameGroup]:
    """Разбить последовательные кадры на логические слайды.

    :param frames: кадры одного интервала стабильной раскладки, по возрастанию
        времени, уже очищенные от кадров без демонстрации.
    :param step_s: шаг выборки кадров, с — им продлевается конец группы,
        чтобы интервал накрывал последний свой кадр целиком.
    :param min_group_s: группы короче отбрасываются как переходный мусор;
        0 — не отбрасывать (ошибка в сторону дробления безопаснее ошибки в
        сторону склейки, design D4).

    Границы групп режутся не только по скачку хэша, но и по **пропуску в
    последовательности кадров**: `frames` уже очищены от кадров без
    демонстрации, поэтому пропуск — это отрезок записи, где демонстрации
    нет. Такой отрезок не должен принадлежать ни одному слайду (спека,
    «Разрывы между слайдами»), поэтому группа закрывается на своём
    последнем кадре плюс `step_s`, а не на метке следующего кадра.

    Пропуск короче `min_absence_s` группу не режет — это тот же порог, что
    гасит короткие провалы в интервалах раскладки: одиночный тёмный кадр
    (сбой детекта, артефакт кодека) не должен порождать лишний слайд и
    секунду «вне слайдов» в транскрипте. Сам такой кадр в хэширование не
    попадает — его выбросили до вызова.
    """
    ordered = [f for f in sorted(frames, key=lambda f: f.timestamp_s)]
    if not ordered:
        return []

    hashes = hash_sequence(ordered, region, hash_size=hash_size)
    distances = hash_distances(hashes)
    # Допуск на дрожание меток кадров: соседними считаются кадры, отстоящие
    # не больше чем на полтора шага выборки.
    max_gap_s = step_s * 1.5
    bounds = [0]
    for index in range(1, len(ordered)):
        gap_s = ordered[index].timestamp_s - ordered[index - 1].timestamp_s
        # Длительность отсутствия — промежуток между кадрами минус шаг.
        absence_s = gap_s - step_s
        is_absence = gap_s > max_gap_s and absence_s >= min_absence_s
        if is_absence or distances[index - 1] >= threshold:
            bounds.append(index)
    bounds.append(len(ordered))

    groups: list[FrameGroup] = []
    for lo, hi in zip(bounds, bounds[1:]):
        chunk = tuple(ordered[lo:hi])
        end_s = chunk[-1].timestamp_s + step_s
        if (end_s - chunk[0].timestamp_s) < min_group_s:
            continue
        groups.append(
            FrameGroup(
                frames=chunk,
                region=region,
                start_s=float(chunk[0].timestamp_s),
                end_s=float(end_s),
            )
        )
    logger.debug(
        "кадров=%d -> групп=%d (hash_size=%d, порог=%d)",
        len(ordered),
        len(groups),
        hash_size,
        threshold,
    )
    return groups
