"""Проверки стадии slide-extraction на полной эталонной записи.

Тяжёлые: извлечение 5482 кадров и разбор всей записи. Помечены
``reference`` и ``slow``. Кадры извлекаются один раз на сессию; если они
уже где-то лежат, путь можно передать через ``LECTURE_SLIDE_FRAMES_DIR``,
чтобы не переизвлекать (каталог должен содержать ``frame_%06d.jpg``,
1 кадр/с, начиная с нулевой секунды).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from lecture_transcript.config import load_config
from lecture_transcript.contracts import Frame, FramesArtifact, Rect
from lecture_transcript.media_ingest import extract_frames, probe
from lecture_transcript.slide_extraction import (
    FrameGroup,
    analyze_layout,
    crop_hash,
    extract_slides,
    find_region,
    groups_from_layout,
    sample_frames,
)

pytestmark = [pytest.mark.reference, pytest.mark.slow]

ENV_FRAMES_DIR = "LECTURE_SLIDE_FRAMES_DIR"


# --------------------------------------------------------------------------
# Фикстуры
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def reference_frames(reference_mp4, tmp_path_factory) -> FramesArtifact:
    """Кадры эталонной записи, 1 кадр/с."""
    existing = os.environ.get(ENV_FRAMES_DIR)
    if existing:
        directory = Path(existing).expanduser()
        paths = sorted(directory.glob("frame_*.jpg"))
        if not paths:
            pytest.skip(f"в {directory} нет кадров frame_*.jpg")
        return FramesArtifact(
            directory=directory,
            fps=1.0,
            frames=tuple(
                Frame(index=i, timestamp_s=float(i), path=p)
                for i, p in enumerate(paths)
            ),
        )
    out = tmp_path_factory.mktemp("reference_frames")
    return extract_frames(probe(reference_mp4), out / "frames", fps=1.0)


@pytest.fixture(scope="session")
def reference_layout(reference_frames):
    """Разбор раскладки по всей записи — считается один раз."""
    return analyze_layout(reference_frames.frames)


@pytest.fixture(scope="session")
def reference_groups(reference_layout):
    """Логические слайды при пороге из конфига — считаются один раз."""
    threshold = load_config().slide_extraction.hamming_threshold
    return groups_from_layout(reference_layout, hamming_threshold=threshold, step_s=1.0)


@pytest.fixture(scope="session")
def expected_region(fixtures_manifest) -> Rect:
    spec = fixtures_manifest["slide_region_1080p"]
    return Rect(x=spec["x"], y=spec["y"], width=spec["width"], height=spec["height"])


def _window(frames, center_s: float, half_width_s: float = 30.0):
    return [
        f.path
        for f in sample_frames(
            frames,
            start_s=center_s - half_width_s,
            end_s=center_s + half_width_s,
            count=100,
        )
    ]


def _assert_region_covers(found: Rect | None, expected: Rect, tolerance: int = 16):
    """Найденная область содержит разведанную и не раздута на видеосетку.

    `slide_region_1080p` манифеста — светлое поле шаблона (белая карточка).
    Область демонстрации шире: рамка шаблона вокруг карточки — часть
    транслируемого экрана, и лектор пишет поверх неё.
    """
    assert found is not None
    assert found.x <= expected.x + tolerance, found
    assert found.y <= expected.y + tolerance, found
    assert found.x + found.width >= expected.x + expected.width - tolerance, found
    assert found.y + found.height >= expected.y + expected.height - tolerance, found
    assert found.area <= expected.area * 1.5, found


def _absence_runs(layout) -> list[tuple[float, float]]:
    """Сырые пробеги «демонстрации нет» по покадровому признаку."""
    runs: list[tuple[float, float]] = []
    start: float | None = None
    for frame, present in zip(layout.frames, layout.present):
        if not present and start is None:
            start = frame.timestamp_s
        if present and start is not None:
            runs.append((start, frame.timestamp_s))
            start = None
    if start is not None:
        runs.append((start, layout.frames[-1].timestamp_s + 1.0))
    return runs


def _overlaps(first, second) -> bool:
    return not (first[1] <= second[0] or second[1] <= first[0])


# --------------------------------------------------------------------------
# 3.1 / 3.2 — область демонстрации на контрольных таймкодах
# --------------------------------------------------------------------------


def _close(first: Rect, second: Rect, tolerance: int = 16) -> bool:
    return all(
        abs(a - b) <= tolerance
        for a, b in (
            (first.x, second.x),
            (first.y, second.y),
            (first.x + first.width, second.x + second.width),
            (first.y + first.height, second.y + second.height),
        )
    )


def _assert_region_around_core(region: Rect | None, core: Rect | None) -> None:
    """Область содержит светлое ядро момента и шире его на рамку шаблона."""
    assert core is not None and region is not None
    assert region.x <= core.x and region.y <= core.y, (region, core)
    assert region.x + region.width >= core.x + core.width + 16, (region, core)
    assert region.y + region.height >= core.y + core.height + 40, (region, core)
    assert region.area <= core.area * 1.5, (region, core)


def test_region_found_at_probe_timecodes(
    reference_layout, reference_frames, expected_region, fixtures_manifest
):
    """t≈600 и t≈4200 — демонстрация есть, область накрывает светлое поле (3.2).

    Окно демонстрации на эталоне бывает двух размеров: 1272×716 (t≈600) и
    1360×764 (t≈4200); `slide_region_1080p` манифеста снят со второго.
    Поэтому область интервала сверяется со светлым ядром своего момента, а
    с манифестом — там, где ядро с ним совпадает (хотя бы на одной точке).
    """
    probes = fixtures_manifest["timecodes"]["task_3_2_probes"]["with_slide_s"]
    matched_manifest = 0
    for center_s in probes:
        interval = next(
            i
            for i in reference_layout.intervals
            if i.start_s <= center_s < i.end_s
        )
        core = find_region(
            _window(reference_frames.frames, center_s), expand_border=False
        )
        _assert_region_around_core(interval.region, core)
        if _close(core, expected_region):
            _assert_region_covers(interval.region, expected_region)
            matched_manifest += 1
    assert matched_manifest >= 1


def test_single_window_region_is_the_whole_shared_screen(
    reference_frames, expected_region, fixtures_manifest
):
    """Область по одному окну — вся плитка демонстрации 16:9, а не карточка (3.2).

    Расширение ядра до внешней границы статичной области опирается не на
    цвет шаблона, а на устройство видеосетки. Проверка того, что найдено
    именно транслируемое окно, — его соотношение сторон при обоих размерах
    окна демонстрации.
    """
    probes = fixtures_manifest["timecodes"]["task_3_2_probes"]["with_slide_s"]
    matched_manifest = 0
    for center_s in probes:
        paths = _window(reference_frames.frames, center_s)
        core = find_region(paths, expand_border=False)
        region = find_region(paths)
        _assert_region_around_core(region, core)
        assert region.width / region.height == pytest.approx(16 / 9, abs=0.02), region
        if _close(core, expected_region):
            _assert_region_covers(region, expected_region)
            matched_manifest += 1
    assert matched_manifest >= 1


#: Слайды, у которых ответ дописан маркером под белой карточкой шаблона —
#: поверх рамки. До правки этих ответов в PNG не было (аудит, находка 1).
ANSWERS_OUTSIDE_CARD_S = (1840.0, 3613.0, 4677.0)


def _ink_rows(path: Path, box: tuple[int, int, int, int], level: int = 100) -> int:
    with Image.open(path) as image:
        crop = np.asarray(image.convert("L").crop(box))
    return int((crop < level).sum())


def test_region_keeps_answers_written_outside_the_card(
    reference_frames, expected_region
):
    """Ответ, дописанный под карточкой, попадает в область демонстрации (3.2).

    Мера — тёмные пиксели в полосе между низом карточки и низом найденной
    области: там написан ответ («Отв: −5; 0», «Ответ: 2», «Ответ: 3…»).
    До правки низ области совпадал с низом карточки и полосы не было.
    """
    card_bottom = expected_region.y + expected_region.height
    by_time = {f.timestamp_s: f for f in reference_frames.frames}
    for center_s in ANSWERS_OUTSIDE_CARD_S:
        region = find_region(_window(reference_frames.frames, center_s))
        assert region is not None
        bottom = region.y + region.height
        assert bottom >= card_bottom + 40, (center_s, region)
        ink = _ink_rows(
            by_time[center_s].path,
            (region.x, card_bottom, region.x + region.width, bottom),
        )
        assert ink >= 1000, (center_s, ink)


def test_region_absent_in_no_demo_interval(reference_frames, fixtures_manifest):
    """t≈2400 — демонстрации нет, область не находится (3.2)."""
    probe_s = fixtures_manifest["timecodes"]["task_3_2_probes"]["without_slide_s"]
    interval = next(
        (a, b)
        for a, b in fixtures_manifest["timecodes"]["no_demo_intervals_s"]
        if a <= probe_s < b
    )
    paths = [
        f.path
        for f in reference_frames.frames
        if interval[0] <= f.timestamp_s < interval[1]
    ]
    assert find_region(paths) is None


def test_frame_presence_matches_probes(reference_layout, fixtures_manifest):
    """Покадровый признак присутствия согласован с разведанными таймкодами."""
    probes = fixtures_manifest["timecodes"]["task_3_2_probes"]
    present = {
        f.timestamp_s: p
        for f, p in zip(reference_layout.frames, reference_layout.present)
    }
    for center_s in probes["with_slide_s"]:
        assert present[center_s] is True
    assert present[probes["without_slide_s"]] is False


# --------------------------------------------------------------------------
# 3.3 — раскладка по всей записи
# --------------------------------------------------------------------------

#: Сколько раз на эталоне пропадает демонстрация. 33 интервала перечислены
#: в манифесте как `no_demo_intervals_s`; ещё 12 манифест относит только к
#: `slide_switches_s`, но глазами в каждом из них в кадре одна видеосетка
#: без демонстрации (244–249, 282–285, 383–387, 400–404, 410–414, 446–450,
#: 473–477, 700–705, 721–725, 1397–1401, 1408–1417, 1439–1443).
EXPECTED_ABSENCES = 45


def test_layout_marks_every_absence_of_demonstration(
    reference_layout, fixtures_manifest
):
    """Каждое пропадание демонстрации — отдельный интервал без области (3.3).

    Контрольная точка задачи 3.2 — t=2400 (провал 2399–2412, 13 с) —
    обязана быть отсутствием. До правки гистерезис 20 с заливал все
    провалы, кроме начального, и раскладка состояла из двух интервалов.
    """
    timecodes = fixtures_manifest["timecodes"]
    intervals = reference_layout.intervals
    absent = [(i.start_s, i.end_s) for i in intervals if i.region is None]

    probe_s = timecodes["task_3_2_probes"]["without_slide_s"]
    at_probe = next(i for i in intervals if i.start_s <= probe_s < i.end_s)
    assert at_probe.region is None, at_probe

    manifest_absent = [tuple(x) for x in timecodes["no_demo_intervals_s"]]
    switches = [tuple(x) for x in timecodes["slide_switches_s"]]
    missed = [m for m in manifest_absent if not any(_overlaps(m, a) for a in absent)]
    unexplained = [
        a for a in absent if not any(_overlaps(a, m) for m in manifest_absent + switches)
    ]
    assert not missed, f"разведанные отсутствия не найдены: {missed}"
    assert not unexplained, f"отсутствия, которых нет в разведке: {unexplained}"
    assert len(absent) == EXPECTED_ABSENCES

    transitions = sum(
        1
        for previous, current in zip(intervals, intervals[1:])
        if (previous.region is None) != (current.region is None)
    )
    # Запись начинается без демонстрации и заканчивается с ней.
    assert transitions == 2 * EXPECTED_ABSENCES - 1


def test_raw_absences_are_not_healed(reference_layout):
    """Интервалы раскладки без области совпадают с сырыми провалами (3.3).

    По длительности переключение слайда и «настоящее» выключение не
    различаются: после провала 2399–2412 демонстрация возвращается на то же
    место, что и после любой смены слайда, — и провал этот настоящий.
    """
    absent = [
        (i.start_s, i.end_s) for i in reference_layout.intervals if i.region is None
    ]
    runs = _absence_runs(reference_layout)
    assert absent == runs
    assert min(end - start for start, end in runs) >= 3.0

    intervals = reference_layout.intervals
    index = next(k for k, i in enumerate(intervals) if i.start_s == 2399.0)
    assert intervals[index - 1].region == intervals[index + 1].region


def test_layout_intervals_cover_record(reference_layout, reference_frames):
    """Интервалы покрывают запись подряд, без дыр и пересечений (3.3)."""
    intervals = reference_layout.intervals
    assert intervals[0].start_s == reference_frames.frames[0].timestamp_s
    for previous, current in zip(intervals, intervals[1:]):
        assert previous.end_s == current.start_s
    assert intervals[-1].end_s >= reference_frames.frames[-1].timestamp_s


# --------------------------------------------------------------------------
# 3.5 — репрезентативный кадр на настоящем интервале дописывания
# --------------------------------------------------------------------------


def test_representative_of_marker_interval_holds_the_answer(reference_layout):
    """«Метод замены переменной» (2468..3017): в конце дописано решение (3.5).

    Слайда «Пример 2. √(x²−x) = −5» из design.md на этом интервале нет —
    проверяется самый длинный реальный интервал дописывания маркером.
    Мера — число тёмных пикселей внутри области демонстрации; порог 100,
    чтобы рамка шаблона (≈122) не заслоняла написанное.
    """
    interval = next(
        i for i in reference_layout.intervals if i.start_s <= 2500.0 < i.end_s
    )
    region = interval.region
    frames = [
        f
        for f in reference_layout.present_frames(interval)
        if 2468.0 <= f.timestamp_s < 3018.0
    ]
    assert len(frames) > 400

    ink = [_ink(f.path, region) for f in frames]
    assert ink[-1] > ink[0] * 1.5, (ink[0], ink[-1])
    assert ink[-1] >= max(ink) * 0.98
    assert float(np.corrcoef(np.arange(len(ink)), ink)[0, 1]) > 0.9

    first = crop_hash(frames[0].path, region)
    distances = [int(crop_hash(f.path, region) - first) for f in frames]
    assert distances[-1] >= max(distances) * 0.8


def _ink(path: Path, region: Rect, level: int = 100) -> int:
    with Image.open(path) as image:
        crop = image.convert("L").crop(
            (region.x, region.y, region.x + region.width, region.y + region.height)
        )
        return int((np.asarray(crop) < level).sum())


# --------------------------------------------------------------------------
# 3.7 — порог hamming и число логических слайдов
# --------------------------------------------------------------------------

#: Число групп при пороге из конфига. Это не число логических слайдов:
#: ручной подсчёт по репрезентативным кадрам даёт 43. Лишние 10 делений —
#: безопасные (одно и то же содержимое по обе стороны границы):
#: 6 — смена масштаба окна демонстрации без смены слайда (t≈797, 1225, 1531,
#: 5126, 5155, 5246), 2 — сдвиг окна (t≈1368, 1369), 2 — возврат того же
#: слайда после провала демонстрации (t≈1401, 2034). Все 49 границ прежнего
#: разбиения сохранены, то есть новых склеек нет.
EXPECTED_GROUP_COUNT = 53

#: Измерено на эталоне по кропу всей плитки демонстрации: максимум hamming
#: distance между соседними кадрами при дописывании маркером внутри
#: непрерывного показа и минимум при настоящей смене слайда.
WRITING_MAX_DISTANCE = 30
WEAKEST_SWITCH_DISTANCE = 56

#: Пороги, дающие одно и то же разбиение (31..46; 46 и 52 — сдвиг окна на
#: том же слайде, выше них группы сливаются по одному сдвигу за раз).
#: Проверяются края и порог из конфига.
THRESHOLD_PLATEAU = (31, 43, 46)


def test_configured_threshold_sits_below_the_middle_of_the_gap():
    """Порог из конфига в зазоре и не выше его середины (3.7, design D4).

    Ошибка в сторону дробления безопаснее ошибки в сторону склейки, поэтому
    порог не должен заходить в верхнюю половину зазора.
    """
    threshold = load_config().slide_extraction.hamming_threshold
    assert WRITING_MAX_DISTANCE < threshold <= WEAKEST_SWITCH_DISTANCE
    middle = (WRITING_MAX_DISTANCE + WEAKEST_SWITCH_DISTANCE) / 2
    assert threshold <= middle, "порог в верхней половине зазора: склейка ближе"


def test_group_count_matches_review(reference_groups):
    """При пороге из конфига — разведанное число групп (3.7).

    Логических слайдов 43; как из них получаются 53 группы — см.
    `EXPECTED_GROUP_COUNT`.
    """
    assert len(reference_groups) == EXPECTED_GROUP_COUNT


def test_threshold_plateau_is_flat(reference_layout):
    """Края плато и порог из конфига дают тот же результат (3.7)."""
    counts = {
        threshold: len(
            groups_from_layout(
                reference_layout, hamming_threshold=threshold, step_s=1.0
            )
        )
        for threshold in THRESHOLD_PLATEAU
    }
    assert set(counts.values()) == {EXPECTED_GROUP_COUNT}, counts


#: Средний модуль разности яркости (MAE) кропа в 1/4 разрешения. Измерено на
#: эталоне: внутри групп кадр отличается от представителя не больше чем на
#: 3.96 (дописывание маркером, группа с t=2468); на границах групп с разным
#: содержимым — не меньше 5.94. Метрика не зависит от phash, которым
#: группы строились, поэтому проверка не повторяет саму себя.
MAE_SAME_SLIDE_LIMIT = 5.0


def _gray_crop(path: Path, region: Rect) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4).astype(np.float32)
    return image[
        region.y // 4 : (region.y + region.height) // 4,
        region.x // 4 : (region.x + region.width) // 4,
    ]


def _max_mae_to_representative(group: FrameGroup, cache: dict) -> float:
    def crop(frame: Frame) -> np.ndarray:
        key = (frame.timestamp_s, group.region)
        if key not in cache:
            cache[key] = _gray_crop(frame.path, group.region)
        return cache[key]

    last = crop(group.representative)
    return max(float(np.abs(crop(frame) - last).mean()) for frame in group.frames)


def test_no_group_merges_two_different_slides(reference_groups):
    """Внутри группы нет второго слайда: склеек нет (3.7).

    Каждый кадр группы сравнивается с представителем по пиксельному MAE.
    Чтобы проверка не была декоративной, она прогоняется и на искусственных
    склейках соседних групп с разным содержимым — каждую из них она обязана
    поймать. Прежний вариант (phash первого и последнего кадра < 80) такие
    склейки пропускал.
    """
    cache: dict = {}
    worst = {
        (g.start_s, g.end_s): _max_mae_to_representative(g, cache)
        for g in reference_groups
    }
    merged = {k: v for k, v in worst.items() if v >= MAE_SAME_SLIDE_LIMIT}
    assert not merged, f"группы с чужим содержимым: {merged}"

    caught = 0
    for previous, current in zip(reference_groups, reference_groups[1:]):
        if previous.region != current.region:
            continue
        boundary = float(
            np.abs(
                _gray_crop(previous.representative.path, previous.region)
                - _gray_crop(current.representative.path, current.region)
            ).mean()
        )
        if boundary < MAE_SAME_SLIDE_LIMIT:
            continue  # по обе стороны одно содержимое — склейка безвредна
        fake = FrameGroup(
            frames=previous.frames + current.frames,
            region=previous.region,
            start_s=previous.start_s,
            end_s=current.end_s,
        )
        assert _max_mae_to_representative(fake, cache) >= MAE_SAME_SLIDE_LIMIT, (
            previous.start_s,
            current.start_s,
        )
        caught += 1
    assert caught >= 30, f"искусственных склеек проверено слишком мало: {caught}"


# --------------------------------------------------------------------------
# 3.6 — итоговый список слайдов на всей записи
# --------------------------------------------------------------------------


def test_full_run_produces_ordered_disjoint_slides(
    reference_frames, reference_mp4, fixtures_manifest, tmp_path_factory
):
    """Прогон стадии на всей записи: слайды упорядочены, разрывы не съедены (3.6).

    Отсутствие демонстрации не принадлежит ни одному слайду (спека,
    «Разрывы между слайдами»); PNG трёх слайдов с ответом под карточкой
    содержат всю плитку демонстрации, а `Slide.region` совпадает с PNG.
    """
    out = tmp_path_factory.mktemp("reference_slides")
    slides = extract_slides(reference_frames, out, source=reference_mp4)

    assert len(slides) == EXPECTED_GROUP_COUNT
    assert [s.index for s in slides] == list(range(1, len(slides) + 1))
    assert [s.image_path.name for s in slides] == [
        f"slide_{i:03d}.png" for i in range(1, len(slides) + 1)
    ]
    for previous, current in zip(slides, slides[1:]):
        assert previous.start_s < previous.end_s
        assert previous.end_s <= current.start_s

    for start, end in fixtures_manifest["timecodes"]["no_demo_intervals_s"]:
        middle = (start + end) / 2
        inside = [s.index for s in slides if s.start_s <= middle < s.end_s]
        assert not inside, f"отсутствие {start}–{end} внутри слайдов {inside}"

    by_representative = {s.representative_timestamp_s: s for s in slides}
    for slide in slides:
        assert slide.image_path.exists()
        with Image.open(slide.image_path) as image:
            assert image.format == "PNG"
            assert image.size == (slide.region.width, slide.region.height)
    card = fixtures_manifest["slide_region_1080p"]
    for moment in ANSWERS_OUTSIDE_CARD_S:
        slide = by_representative[moment]
        assert slide.region.y + slide.region.height >= card["y"] + card["height"] + 40


# --------------------------------------------------------------------------
# Аудит, итерация 2: область на момент репрезентативного кадра
# --------------------------------------------------------------------------


def _edge_fits_gutter(gray: np.ndarray, region: Rect) -> tuple[bool, tuple]:
    """Правый и нижний края области лежат на промежутке между плитками.

    Внутри у края — рамка шаблона (≈122); снаружи в пределах 8 px есть
    линия, в которой не меньше 40 % пикселей темнее 40, — промежуток между
    плитками. Долю, а не медиану: промежуток местами перекрыт подсвеченной
    рамкой плитки говорящего (t=243). Малая область на большом окне даёт
    снаружи содержимое слайда (тёмных почти нет), большая на малом — фон
    соседних плиток внутри у края.
    """
    x, y, w, h = region.x, region.y, region.width, region.height
    rows = slice(y + h // 4, y + 3 * h // 4)
    cols = slice(x + w // 4, x + 3 * w // 4)
    right_in = float(np.median(gray[rows, x + w - 14 : x + w - 6]))
    right_out = max(
        [float((gray[rows, c] < 40).mean()) for c in range(x + w, min(x + w + 8, gray.shape[1]))]
        or [1.0]
    )
    bottom_in = float(np.median(gray[y + h - 14 : y + h - 6, cols]))
    bottom_out = max(
        [float((gray[r, cols] < 40).mean()) for r in range(y + h, min(y + h + 8, gray.shape[0]))]
        or [1.0]
    )
    values = (right_in, right_out, bottom_in, bottom_out)
    fits = right_in > 90 and right_out >= 0.4 and bottom_in > 90 and bottom_out >= 0.4
    return fits, values


def test_every_group_region_is_the_window_at_its_representative(reference_groups):
    """Область каждой группы — окно демонстрации на момент её представителя.

    Окно демонстрации на эталоне шесть раз меняет размер без провала
    (1272×716 <-> 1360×764). До правки область бралась по окну анализа или
    интервалу и на слайдах 014–016, 046, 048 резала содержимое, а на 019–021,
    024 захватывала полосу с камерами участников.
    """
    misfits = {}
    for group in reference_groups:
        gray = cv2.imread(str(group.representative.path), cv2.IMREAD_GRAYSCALE)
        fits, values = _edge_fits_gutter(gray, group.region)
        ratio = group.region.width / group.region.height
        if not fits or abs(ratio - 16 / 9) > 0.02:
            misfits[group.representative.timestamp_s] = (group.region, values)
    assert not misfits, misfits
