"""Быстрые тесты стадии slide-extraction — на коротких фрагментах эталона.

Проверки, требующие всей записи, живут в
``tests/test_slide_extraction_reference.py`` под маркерами
``reference``/``slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lecture_transcript.contracts import Frame, FramesArtifact, Rect
from lecture_transcript.media_ingest import extract_frames, probe
from lecture_transcript.slide_extraction import (
    DEFAULT_HAMMING_THRESHOLD,
    build_slides,
    crop_hash,
    detect_layout,
    extract_slides,
    find_region,
    group_frames,
    pixel_stats,
    sample_frames,
    smooth_presence,
    static_bright_mask,
    validate_slides,
)

# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------


def frames_of(clip: Path, tmp_path_factory) -> FramesArtifact:
    """Кадры фрагмента, 1 fps, извлечённые один раз на сессию."""
    out = tmp_path_factory.mktemp(f"frames_{clip.stem}")
    return extract_frames(probe(clip), out / "frames", fps=1.0, hwaccel="none")


def ink_pixels(path: Path, region: Rect, level: int = 100) -> int:
    """Сколько тёмных пикселей внутри области — мера «сколько написано».

    Порог 100, а не 128: область демонстрации теперь включает рамку
    шаблона (яркость ≈122), и при пороге 128 она одна давала бы вклад
    больше, чем всё написанное маркером.
    """
    with Image.open(path) as image:
        crop = image.convert("L").crop(
            (region.x, region.y, region.x + region.width, region.y + region.height)
        )
        return int((np.asarray(crop) < level).sum())


@pytest.fixture(scope="session")
def slide_frames(clip_slide, tmp_path_factory) -> FramesArtifact:
    return frames_of(clip_slide, tmp_path_factory)


@pytest.fixture(scope="session")
def no_slide_frames(clip_no_slide, tmp_path_factory) -> FramesArtifact:
    return frames_of(clip_no_slide, tmp_path_factory)


@pytest.fixture(scope="session")
def switch_frames(clip_slide_switch, tmp_path_factory) -> FramesArtifact:
    return frames_of(clip_slide_switch, tmp_path_factory)


@pytest.fixture(scope="session")
def marker_frames(clip_marker, tmp_path_factory) -> FramesArtifact:
    return frames_of(clip_marker, tmp_path_factory)


@pytest.fixture(scope="session")
def expected_region(fixtures_manifest) -> Rect:
    spec = fixtures_manifest["slide_region_1080p"]
    return Rect(
        x=spec["x"], y=spec["y"], width=spec["width"], height=spec["height"]
    )


def assert_region_covers(
    found: Rect | None,
    expected: Rect,
    *,
    tolerance: int = 16,
    max_area_factor: float = 1.5,
) -> None:
    """Область накрывает разведанную и не разрастается на видеосетку.

    `slide_region_1080p` из манифеста — это **светлое поле** шаблона
    (белая карточка). Настоящая область демонстрации шире: у шаблона есть
    рамка, поверх которой лектор дописывает ответы, и терять их нельзя
    (см. `region.expand_to_static_border`). Поэтому проверяется не
    совпадение, а включение: найденная область обязана целиком содержать
    разведанную, но не может раздуться настолько, чтобы захватить тайлы
    участников.
    """
    assert found is not None
    assert found.x <= expected.x + tolerance, f"левый край срезан: {found}"
    assert found.y <= expected.y + tolerance, f"верхний край срезан: {found}"
    assert (
        found.x + found.width >= expected.x + expected.width - tolerance
    ), f"правый край срезан: {found}"
    assert (
        found.y + found.height >= expected.y + expected.height - tolerance
    ), f"нижний край срезан: {found}"
    assert found.area <= expected.area * max_area_factor, (
        f"область раздулась: {found.area} > {expected.area} × {max_area_factor}"
    )


# --------------------------------------------------------------------------
# 3.1 — дисперсия и яркость
# --------------------------------------------------------------------------


def test_mask_covers_demo_region_and_not_participant_tiles(
    slide_frames, expected_region
):
    """Маска «низкая дисперсия И высокая яркость» = область демонстрации (3.1)."""
    stats = pixel_stats([f.path for f in sample_frames(slide_frames.frames, count=100)])
    mask = static_bright_mask(stats).astype(bool)

    scale = stats.scale
    inside = np.zeros(mask.shape, dtype=bool)
    inside[
        round(expected_region.y / scale) : round(
            (expected_region.y + expected_region.height) / scale
        ),
        round(expected_region.x / scale) : round(
            (expected_region.x + expected_region.width) / scale
        ),
    ] = True

    assert mask[inside].mean() > 0.9, "маска не покрывает область демонстрации"
    assert mask[~inside].mean() < 0.1, "маска залезла на тайлы участников"


def test_participant_tiles_are_not_static_and_bright(slide_frames, expected_region):
    """Разведение признаков: тайлы подвижнее и темнее области демонстрации."""
    stats = pixel_stats([f.path for f in sample_frames(slide_frames.frames, count=100)])
    scale = stats.scale
    inside = np.zeros(stats.spread.shape, dtype=bool)
    inside[
        round(expected_region.y / scale) : round(
            (expected_region.y + expected_region.height) / scale
        ),
        round(expected_region.x / scale) : round(
            (expected_region.x + expected_region.width) / scale
        ),
    ] = True

    assert np.median(stats.brightness[inside]) > 150
    assert np.percentile(stats.brightness[~inside], 90) < 150
    assert np.percentile(stats.spread[inside], 90) < 8.0


# --------------------------------------------------------------------------
# 3.2 — прямоугольник области демонстрации и её отсутствие
# --------------------------------------------------------------------------


def test_region_found_on_clip_with_slide(slide_frames, expected_region):
    """Фрагмент с демонстрацией: границы найдены и совпадают с эталонными (3.2)."""
    region = find_region([f.path for f in sample_frames(slide_frames.frames, count=100)])
    assert_region_covers(region, expected_region)


def test_region_absent_on_clip_without_slide(no_slide_frames):
    """Фрагмент без демонстрации: область не найдена (3.2)."""
    region = find_region(
        [f.path for f in sample_frames(no_slide_frames.frames, count=100)]
    )
    assert region is None


def test_region_from_mask_rejects_thin_strip():
    """Длинная узкая светлая полоса не считается областью демонстрации."""
    from lecture_transcript.slide_extraction import region_from_mask

    mask = np.zeros((270, 480), dtype=np.uint8)
    mask[10:40, :] = 1  # полоса во всю ширину, но всего 30 px высотой
    assert (
        region_from_mask(mask, frame_width=1920, frame_height=1080) is None
    )


# --------------------------------------------------------------------------
# 3.3 — раскладка и гистерезис
# --------------------------------------------------------------------------


def test_hysteresis_fills_switch_artifact_but_keeps_real_absence():
    """Провал короче порога — артефакт переключения, длинный — выключение (3.3)."""
    flags = [True] * 30 + [False] * 4 + [True] * 30 + [False] * 60 + [True] * 30
    smoothed = smooth_presence(flags, step_s=1.0, min_absence_s=20.0)

    assert all(smoothed[:64]), "четырёхсекундный провал не заполнен"
    assert not any(smoothed[64:124]), "шестидесятисекундный провал заполнен зря"
    assert all(smoothed[124:])


def test_hysteresis_suppresses_single_frame_spike():
    """Одиночный ложный кадр не разрезает настоящее выключение."""
    flags = [True] * 10 + [False] * 15 + [True] + [False] * 15 + [True] * 10
    smoothed = smooth_presence(flags, step_s=1.0, min_absence_s=20.0)

    assert not any(smoothed[10:41]), "выключение развалилось на два коротких"


def test_layout_switch_clip_keeps_absence_as_its_own_interval(
    switch_frames, expected_region, clip_specs
):
    """Пропадание демонстрации при смене слайда — отдельный интервал без области (3.3).

    На этой платформе смена слайда — это настоящее снятие демонстрации:
    тайлы участников разворачиваются на весь кадр. Манифест отмечает его
    на 28..33-й секунде фрагмента. Раньше гистерезис 20 с это отсутствие
    заливал, и интервал раскладки говорил «демонстрация есть».
    """
    intervals = detect_layout(switch_frames.frames)
    lo, hi = clip_specs["clip_slide_switch"]["expect"]["switch_at_offset_s"]

    assert [i.region is None for i in intervals] == [False, True, False], [
        (i.start_s, i.end_s, i.region) for i in intervals
    ]
    absence = intervals[1]
    assert lo <= absence.start_s <= hi and lo <= absence.end_s <= hi, absence
    assert_region_covers(intervals[0].region, expected_region)
    # Демонстрация возвращается на то же место, что и до провала.
    assert intervals[2].region == intervals[0].region


def test_layout_no_slide_clip_has_no_region(no_slide_frames):
    """Фрагмент без демонстрации: ни один интервал не несёт области (3.3)."""
    intervals = detect_layout(no_slide_frames.frames)
    assert all(interval.region is None for interval in intervals)


def test_layout_intervals_cover_record_without_gaps(switch_frames):
    """Интервалы раскладки идут подряд, не пересекаясь и не оставляя дыр."""
    intervals = detect_layout(switch_frames.frames)
    for previous, current in zip(intervals, intervals[1:]):
        assert previous.end_s == current.start_s
    assert intervals[0].start_s == switch_frames.frames[0].timestamp_s


# --------------------------------------------------------------------------
# 3.4 — группировка по perceptual hash
# --------------------------------------------------------------------------


def test_marker_writing_stays_in_one_group(marker_frames, expected_region):
    """Дописывание маркером не рвёт логический слайд (3.4)."""
    region = find_region(
        [f.path for f in sample_frames(marker_frames.frames, count=100)]
    )
    groups = group_frames(marker_frames.frames, region)

    assert len(groups) == 1
    assert len(groups[0].frames) == len(marker_frames.frames)


def test_marker_distances_stay_below_threshold(marker_frames):
    """Числовое обоснование: расстояния между соседями меньше порога (3.4)."""
    from lecture_transcript.slide_extraction import hash_distances, hash_sequence

    region = find_region(
        [f.path for f in sample_frames(marker_frames.frames, count=100)]
    )
    distances = hash_distances(hash_sequence(marker_frames.frames, region))

    assert max(distances) < DEFAULT_HAMMING_THRESHOLD


def test_slide_switch_splits_groups(switch_frames, expected_region):
    """Смена содержимого закрывает слайд и открывает новый (3.4)."""
    slides = _slides_of(switch_frames, expected_region)
    assert len(slides) == 2


def _slides_of(artifact: FramesArtifact, region: Rect):
    from lecture_transcript.slide_extraction import analyze_layout, groups_from_layout

    analysis = analyze_layout(artifact.frames)
    return groups_from_layout(
        analysis, hamming_threshold=DEFAULT_HAMMING_THRESHOLD, step_s=1.0
    )


# --------------------------------------------------------------------------
# 3.5 — репрезентативный кадр
# --------------------------------------------------------------------------


def test_representative_is_last_frame_with_most_ink(marker_frames):
    """Репрезентативный кадр содержит дописанное, а не заготовку (3.5).

    Проверка измеримая: считается число тёмных пикселей внутри области —
    прямая мера того, сколько на слайде написано. У репрезентативного
    кадра оно максимально и заметно больше, чем у первого кадра группы.
    """
    region = find_region(
        [f.path for f in sample_frames(marker_frames.frames, count=100)]
    )
    group = group_frames(marker_frames.frames, region)[0]

    assert group.representative is group.frames[-1]

    ink = [ink_pixels(frame.path, region) for frame in group.frames]
    assert ink[-1] > ink[0] * 1.2, "на репрезентативном кадре не прибавилось написанного"
    assert ink[-1] >= max(ink) * 0.98
    # Монотонный рост: дописывание идёт, а не мигает.
    correlation = float(np.corrcoef(np.arange(len(ink)), ink)[0, 1])
    assert correlation > 0.9


def test_representative_drifts_away_from_first_frame(marker_frames):
    """Хэш репрезентативного кадра дальше всех от начала группы (3.5)."""
    region = find_region(
        [f.path for f in sample_frames(marker_frames.frames, count=100)]
    )
    group = group_frames(marker_frames.frames, region)[0]

    first = crop_hash(group.frames[0].path, region)
    distances = [int(crop_hash(f.path, region) - first) for f in group.frames]

    assert distances[0] == 0
    assert distances[-1] >= max(distances) * 0.8


# --------------------------------------------------------------------------
# 3.6 — сохранение кропов и список слайдов
# --------------------------------------------------------------------------


def test_slides_saved_with_stable_names(marker_frames, clip_marker, tmp_path):
    """PNG со стабильными именами, упорядоченными по времени (3.6)."""
    first = extract_slides(marker_frames, tmp_path / "a", source=clip_marker)
    second = extract_slides(marker_frames, tmp_path / "b", source=clip_marker)

    assert [s.image_path.name for s in first] == ["slide_001.png"]
    assert [s.image_path.name for s in first] == [s.image_path.name for s in second]
    assert [(s.start_s, s.end_s) for s in first] == [
        (s.start_s, s.end_s) for s in second
    ]
    for slide in first:
        assert slide.image_path.exists()
        with Image.open(slide.image_path) as image:
            assert image.format == "PNG"
            assert image.size == (slide.region.width, slide.region.height)


def test_slide_intervals_are_ordered_and_disjoint(switch_frames, clip_slide_switch, tmp_path):
    """Интервалы упорядочены по времени и не пересекаются (3.6)."""
    slides = extract_slides(switch_frames, tmp_path, source=clip_slide_switch)

    assert len(slides) == 2
    assert [s.index for s in slides] == [1, 2]
    for previous, current in zip(slides, slides[1:]):
        assert previous.end_s <= current.start_s
    validate_slides(slides)


def test_crop_contains_only_demo_region(marker_frames, clip_marker, tmp_path):
    """Изображение слайда содержит только область демонстрации (3.6)."""
    slide = extract_slides(marker_frames, tmp_path, source=clip_marker)[0]
    with Image.open(slide.image_path) as image:
        gray = np.asarray(image.convert("L"))
    # Внутри кропа нет тёмного поля видеосетки: светлых пикселей большинство.
    assert (gray >= 150).mean() > 0.6


def test_validate_slides_rejects_overlap(tmp_path):
    """Пересекающиеся интервалы — ошибка стадии, а не тихо принятый результат."""
    from lecture_transcript.contracts import Slide
    from lecture_transcript.slide_extraction import SlideExtractionError

    region = Rect(0, 0, 10, 10)
    slides = [
        Slide(1, 0.0, 10.0, region, 9.0, tmp_path / "slide_001.png"),
        Slide(2, 5.0, 15.0, region, 14.0, tmp_path / "slide_002.png"),
    ]
    with pytest.raises(SlideExtractionError):
        validate_slides(slides)


def test_no_demo_gives_empty_slide_list(no_slide_frames, clip_no_slide, tmp_path):
    """Запись без демонстрации: слайдов нет, стадия не падает."""
    assert extract_slides(no_slide_frames, tmp_path, source=clip_no_slide) == []


def test_empty_frames_artifact(tmp_path):
    """Нет кадров — нет слайдов."""
    empty = FramesArtifact(directory=tmp_path, fps=1.0, frames=())
    assert extract_slides(empty, tmp_path / "out") == []


def test_build_slides_falls_back_to_sample_frame(marker_frames, tmp_path):
    """Без исходника кроп режется из кадра выборки — слайд не теряется."""
    from lecture_transcript.slide_extraction import analyze_layout, groups_from_layout

    analysis = analyze_layout(marker_frames.frames)
    groups = groups_from_layout(analysis, hamming_threshold=DEFAULT_HAMMING_THRESHOLD)
    slides = build_slides(groups, tmp_path, source=None)

    assert len(slides) == 1
    assert slides[0].image_path.exists()


def test_frame_is_a_frame_type(marker_frames):
    """Кадры приходят типом контракта — стадии обмениваются только им."""
    assert all(isinstance(frame, Frame) for frame in marker_frames.frames)


# --------------------------------------------------------------------------
# Находки аудита: каждый тест ниже падал на коде до правки
# --------------------------------------------------------------------------


def _synthetic_stats(**overrides):
    """Карты статистик «кадр видеосетки с демонстрацией», сетка 480×270.

    Устройство повторяет эталонную запись: белая карточка шаблона (235)
    внутри рамки (122), вокруг — фон плиток участников (58), между
    плитками — чёрный промежуток, справа — живая вебка с большим разбросом.
    """
    from lecture_transcript.slide_extraction import PixelStats

    brightness = np.full((270, 480), 58.0, dtype=np.float32)
    spread = np.zeros((270, 480), dtype=np.float32)
    brightness[192:194, :] = 0.0  # промежуток под плиткой демонстрации
    brightness[:, 342] = 0.0  # промежуток справа от неё
    brightness[1:192, 2:342] = 122.0  # рамка шаблона = вся плитка демонстрации
    brightness[15:179, 16:328] = 235.0  # светлое поле (карточка)
    brightness[20:60, 360:420] = 120.0  # живая вебка
    spread[20:60, 360:420] = 25.0
    values = dict(
        brightness=brightness,
        spread=spread,
        mean=brightness,
        variance=spread**2,
        frame_width=1920,
        frame_height=1080,
        sample_size=100,
    )
    values.update(overrides)
    return PixelStats(**values)


def test_region_expands_over_template_border(caplog):
    """[P0-1] Область включает рамку шаблона, где дописывают ответ, но не тайлы.

    До правки область обрезалась по белой карточке (Rect(64,60,1248,656)),
    и ответ, написанный маркером под карточкой, в кроп не попадал.
    """
    from lecture_transcript.slide_extraction import region_from_stats

    region = region_from_stats(_synthetic_stats())

    assert region is not None
    # Плитка демонстрации в координатах кадра: x 8..1368, y 4..768.
    assert region.x <= 12 and region.y <= 8, region
    assert region.x + region.width >= 1364, region
    assert region.y + region.height >= 764, region
    # Не залезла ни в промежуток, ни в соседнюю плитку.
    assert region.x + region.width <= 1372, region
    assert region.y + region.height <= 772, region


def test_region_without_border_is_not_expanded():
    """[P0-1] Нет рамки — расширять не по чему, область совпадает с карточкой."""
    from lecture_transcript.slide_extraction import region_from_stats

    stats = _synthetic_stats()
    stats.brightness[1:192, 2:342] = 58.0
    stats.brightness[15:179, 16:328] = 235.0

    region = region_from_stats(stats)
    assert region is not None
    assert (region.x, region.y) == (64, 60)
    assert (region.width, region.height) == (1248, 656)


def _write_pattern(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    pixels = (rng.random((18, 32)) > 0.5).astype(np.uint8) * 255
    Image.fromarray(pixels).resize((320, 180), Image.NEAREST).save(path)
    return path


def _frames_with_gap(tmp_path: Path, after_seed: int) -> list[Frame]:
    """Кадры 0..9 с одним содержимым, провал 10..13, кадры 14..19."""
    before = _write_pattern(tmp_path / "a.png", 1)
    after = _write_pattern(tmp_path / f"b{after_seed}.png", after_seed)
    frames = [Frame(index=t, timestamp_s=float(t), path=before) for t in range(10)]
    frames += [
        Frame(index=t, timestamp_s=float(t), path=after) for t in range(14, 20)
    ]
    return frames


@pytest.mark.parametrize("after_seed", [1, 2], ids=["тот_же_слайд", "другой_слайд"])
def test_group_does_not_swallow_absence(tmp_path, after_seed):
    """[P0-2] Отрезок без демонстрации не принадлежит ни одному слайду.

    До правки группа закрывалась меткой следующего оставшегося кадра, и
    провал 10..14 целиком уходил внутрь интервала предыдущего слайда.
    """
    frames = _frames_with_gap(tmp_path, after_seed)
    groups = group_frames(frames, Rect(0, 0, 320, 180), step_s=1.0)

    assert [(g.start_s, g.end_s) for g in groups] == [(0.0, 10.0), (14.0, 20.0)]


def test_slide_intervals_leave_gap_on_switch_clip(
    switch_frames, clip_slide_switch, clip_specs, tmp_path
):
    """[P0-2] На реальном фрагменте провал при смене слайда остаётся разрывом."""
    lo, hi = clip_specs["clip_slide_switch"]["expect"]["switch_at_offset_s"]
    slides = extract_slides(switch_frames, tmp_path, source=clip_slide_switch)

    assert len(slides) == 2
    assert slides[0].end_s <= slides[1].start_s
    gap = (slides[0].end_s, slides[1].start_s)
    assert gap[1] - gap[0] >= 3.0, gap
    assert lo <= gap[0] and gap[1] <= hi, gap


def _config(**changes):
    from dataclasses import replace

    from lecture_transcript.config import load_config

    return replace(load_config().slide_extraction, **changes)


def test_config_fields_reach_the_place_of_use(switch_frames, tmp_path, monkeypatch):
    """[P0-3] Поля конфига доходят до мест использования, а не теряются.

    До правки `max_spread`, `min_brightness`, `min_absence_s` и `hash_size`
    из конфига не читались вовсе — использовались константы модулей.
    """
    from lecture_transcript.slide_extraction import pipeline

    seen: dict = {}
    real_analyze, real_groups = pipeline.analyze_layout, pipeline.groups_from_layout

    def spy_analyze(frames, **kwargs):
        seen.update(kwargs)
        return real_analyze(frames, **kwargs)

    def spy_groups(analysis, **kwargs):
        seen["hash_size"] = kwargs.get("hash_size")
        return real_groups(analysis, **kwargs)

    monkeypatch.setattr(pipeline, "analyze_layout", spy_analyze)
    monkeypatch.setattr(pipeline, "groups_from_layout", spy_groups)
    config = _config(
        max_spread=1.5, min_brightness=170.0, min_absence_s=7.0, hash_size=12
    )
    extract_slides(switch_frames, tmp_path, config=config)

    assert seen["max_spread"] == 1.5
    assert seen["min_brightness"] == 170.0
    assert seen["presence_level"] == 170.0
    assert seen["min_absence_s"] == 7.0
    assert seen["hash_size"] == 12


def _slides_with(frames, tmp_path, name, **changes):
    out = tmp_path / name
    return [
        (s.start_s, s.end_s, s.region)
        for s in extract_slides(frames, out, config=_config(**changes))
    ]


def test_config_min_brightness_changes_result(switch_frames, tmp_path):
    """[P0-3] Недостижимая яркость убивает детект — результат меняется."""
    assert _slides_with(switch_frames, tmp_path, "base")
    assert _slides_with(switch_frames, tmp_path, "b", min_brightness=250.0) == []


def test_config_max_spread_changes_result(switch_frames, tmp_path, monkeypatch):
    """[P0-3] Более жёсткий разброс меняет область, найденную разбором раскладки.

    Итоговая область слайда уточняется по кадрам у репрезентативного, где
    признак «статично» вырожден, поэтому эффект ручки виден на интервалах
    раскладки — там, где она и используется.
    """
    from lecture_transcript.slide_extraction import pipeline

    regions: list = []
    real = pipeline.analyze_layout

    def spy(frames, **kwargs):
        result = real(frames, **kwargs)
        regions.append([i.region for i in result.intervals if i.region is not None])
        return result

    monkeypatch.setattr(pipeline, "analyze_layout", spy)
    extract_slides(switch_frames, tmp_path / "a", config=_config())
    extract_slides(switch_frames, tmp_path / "b", config=_config(max_spread=1.0))

    assert regions[0] != regions[1], regions


def test_config_hash_size_changes_result(switch_frames, tmp_path):
    """[P0-3] Хэш 64×64 при том же пороге дробит слайд — результат меняется."""
    base = _slides_with(switch_frames, tmp_path, "base")
    fine = _slides_with(switch_frames, tmp_path, "h", hash_size=64)
    assert len(fine) > len(base)


def test_config_min_absence_s_changes_layout(switch_frames, tmp_path, monkeypatch):
    """[P0-3] Порог отсутствия из конфига меняет интервалы раскладки."""
    from lecture_transcript.slide_extraction import pipeline

    layouts: list = []
    real = pipeline.analyze_layout

    def spy(frames, **kwargs):
        result = real(frames, **kwargs)
        layouts.append([i.region is None for i in result.intervals])
        return result

    monkeypatch.setattr(pipeline, "analyze_layout", spy)
    extract_slides(switch_frames, tmp_path / "a", config=_config(min_absence_s=2.0))
    extract_slides(switch_frames, tmp_path / "b", config=_config(min_absence_s=10.0))

    assert layouts == [[False, True, False], [False]]


def test_manual_region_bypasses_detection(slide_frames, tmp_path, caplog):
    """[P1-5] Область, заданная вручную, работает там, где детект не сходится."""
    manual = Rect(8, 4, 1360, 764)
    cfg = _config(min_brightness=250.0)

    with caplog.at_level("WARNING"):
        detected = extract_slides(slide_frames, tmp_path / "auto", config=cfg)
    assert detected == []
    assert "region" in caplog.text, "нет подсказки задать область вручную"

    slides = extract_slides(
        slide_frames, tmp_path / "manual", region=manual, min_brightness=150.0
    )
    assert len(slides) == 1
    assert slides[0].region == manual


def test_dark_slides_give_explicit_diagnostics(caplog):
    """[P1-5] Неуверенный детект объясняет, какая именно проверка не сошлась."""
    from lecture_transcript.slide_extraction import region_from_stats

    stats = _synthetic_stats()
    stats.brightness[:] = 40.0  # тёмные слайды: светлого нет нигде

    with caplog.at_level("WARNING", logger="lecture_transcript.slide_extraction"):
        assert region_from_stats(stats) is None
    assert "Яркость" in caplog.text and "разброс" in caplog.text
    assert "150" in caplog.text, "в диагностике нет порога, с которым сравнивали"


def test_small_region_diagnostics_name_the_failed_check(caplog):
    """[P1-5] Слишком мелкий регион отбраковывается с объяснением."""
    from lecture_transcript.slide_extraction import region_from_mask

    mask = np.zeros((270, 480), dtype=np.uint8)
    mask[10:40, 10:50] = 1
    with caplog.at_level("WARNING", logger="lecture_transcript.slide_extraction"):
        assert region_from_mask(mask, frame_width=1920, frame_height=1080) is None
    assert "мелкий" in caplog.text


def test_single_frame_sample_is_not_trusted(caplog):
    """[P2-14] По одному кадру признак «статично» вырожден — область не ищется."""
    from lecture_transcript.slide_extraction import region_from_stats

    with caplog.at_level("WARNING", logger="lecture_transcript.slide_extraction"):
        assert region_from_stats(_synthetic_stats(sample_size=1)) is None
    assert "вырожден" in caplog.text


def _jpeg(path: Path, size: tuple[int, int], value: int = 200) -> Path:
    Image.new("L", size, value).save(path, format="JPEG")
    return path


def test_odd_aspect_and_missing_frames_are_skipped(tmp_path, caplog):
    """[P1-7] Кадр другого соотношения сторон и битый файл не роняют стадию."""
    paths = [_jpeg(tmp_path / f"f{i}.jpg", (1920, 1080)) for i in range(5)]
    paths.insert(2, _jpeg(tmp_path / "odd.jpg", (640, 480)))
    paths.insert(4, tmp_path / "missing.jpg")

    with caplog.at_level("WARNING", logger="lecture_transcript.slide_extraction"):
        stats = pixel_stats(paths)

    assert stats.sample_size == 5
    assert stats.shape == (270, 480)
    assert "не читается" in caplog.text
    assert "соотношения сторон" in caplog.text


def test_unreadable_sample_raises_pipeline_error(tmp_path):
    """[P1-7] Неустранимая ошибка — `PipelineError` контракта, а не ValueError."""
    from lecture_transcript.contracts import PipelineError

    first = _jpeg(tmp_path / "ok.jpg", (1920, 1080))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg")
    with pytest.raises(PipelineError):
        pixel_stats([broken, tmp_path / "missing.jpg"])
    with pytest.raises(PipelineError):
        pixel_stats([])
    assert pixel_stats([first, broken]).sample_size == 1


def test_slide_region_matches_png_when_frames_are_downscaled(
    marker_frames, clip_marker, tmp_path
):
    """[P2-9] `Slide.region` в той же системе координат, что и PNG."""
    from dataclasses import replace

    from lecture_transcript.slide_extraction import analyze_layout, groups_from_layout

    analysis = analyze_layout(marker_frames.frames)
    groups = groups_from_layout(analysis, hamming_threshold=DEFAULT_HAMMING_THRESHOLD)
    # Притворяемся, что кадры выборки извлечены в половинном разрешении.
    half = [
        replace(
            g,
            region=Rect(
                g.region.x // 2, g.region.y // 2, g.region.width // 2, g.region.height // 2
            ),
        )
        for g in groups
    ]
    slides = build_slides(half, tmp_path, source=clip_marker, frame_width=960)
    with Image.open(slides[0].image_path) as image:
        assert image.size == (slides[0].region.width, slides[0].region.height)


def test_ffmpeg_crop_is_clamped_to_frame(clip_marker, tmp_path, caplog):
    """[P2-11] Окно, вылезающее за кадр, клампится, а не сдвигается молча."""
    from lecture_transcript.slide_extraction.slides import _grab_png

    target = tmp_path / "crop.png"
    with caplog.at_level("WARNING", logger="lecture_transcript.slide_extraction"):
        assert _grab_png(clip_marker, 5.0, Rect(1800, 1000, 400, 400), target)
    with Image.open(target) as image:
        assert image.size == (120, 80)
    assert "выходит за кадр" in caplog.text


def test_stale_pngs_of_previous_run_are_removed(marker_frames, tmp_path):
    """[P2-12] Лишние кропы прошлого прогона не остаются в каталоге."""
    from lecture_transcript.slide_extraction import analyze_layout, groups_from_layout

    analysis = analyze_layout(marker_frames.frames)
    groups = groups_from_layout(analysis, hamming_threshold=DEFAULT_HAMMING_THRESHOLD)
    for number in (2, 3):
        (tmp_path / f"slide_{number:03d}.png").write_bytes(b"old")

    build_slides(groups[:1], tmp_path, source=None)

    assert sorted(p.name for p in tmp_path.glob("slide_*.png")) == ["slide_001.png"]


# --------------------------------------------------------------------------
# Находки аудита, итерация 2: каждый тест ниже падал на коде до правки
# --------------------------------------------------------------------------


def _grid_frame(path: Path, *, tile_w: int, tile_h: int) -> Path:
    """Кадр видеосетки: плитка демонстрации 16:9 и соседние камеры с «лицами».

    Как на эталоне: рамка шаблона 122 вокруг карточки 235, тёмный промежуток
    между плитками, у соседних плиток — светлое пятно камеры вплотную к
    промежутку.
    """
    x0, y0 = 8, 4
    img = np.full((1080, 1920), 58, dtype=np.uint8)
    img[y0 : y0 + tile_h, x0 : x0 + tile_w] = 122
    img[y0 + 56 : y0 + tile_h - 56, x0 + 56 : x0 + tile_w - 56] = 235
    img[y0 + 200 : y0 + 210, x0 + 100 : x0 + 700] = 20
    right, bottom = x0 + tile_w, y0 + tile_h
    img[: bottom + 8, right : right + 8] = 0
    img[bottom : bottom + 8, : right + 8] = 0
    img[40:400, right + 16 : right + 300] = 90
    img[100:340, right + 24 : right + 240] = 210
    img[bottom + 16 :, 300:700] = 90
    img[bottom + 24 : bottom + 250, 320:620] = 210
    Image.fromarray(img).save(path, quality=95)
    return path


def _grid_frames(tmp_path: Path, *, tile_w: int, tile_h: int, count: int = 5):
    return [
        Frame(
            index=i,
            timestamp_s=float(i),
            path=_grid_frame(tmp_path / f"g{tile_w}_{i}.jpg", tile_w=tile_w, tile_h=tile_h),
        )
        for i in range(count)
    ]


def test_group_region_follows_window_growth(tmp_path):
    """[A2-1] Окно выросло внутри интервала — кроп берётся по большому окну.

    До правки у группы оставалась область интервала 1272×716, и у слайда
    срезались правая часть и нижняя рамка, где лектор пишет ответы.
    """
    from lecture_transcript.contracts import LayoutInterval
    from lecture_transcript.slide_extraction import LayoutAnalysis, groups_from_layout

    frames = _grid_frames(tmp_path, tile_w=1360, tile_h=764)
    analysis = LayoutAnalysis(
        frames=tuple(frames),
        present=(True,) * len(frames),
        intervals=(LayoutInterval(0.0, 5.0, Rect(8, 4, 1272, 716)),),
    )
    [group] = groups_from_layout(analysis, hamming_threshold=43)

    region = group.region
    assert abs(region.x - 8) <= 8 and abs(region.y - 4) <= 8, region
    assert abs(region.x + region.width - 1368) <= 8, region
    assert abs(region.y + region.height - 768) <= 8, region


def test_group_region_does_not_capture_participant_cameras(tmp_path):
    """[A2-1] Окно сжалось внутри интервала — полоса с камерами в кроп не попадает.

    До правки у группы оставалась область интервала 1360×764, и в кроп
    слайда попадала полоса соседних плиток с камерами участников.
    """
    from lecture_transcript.contracts import LayoutInterval
    from lecture_transcript.slide_extraction import LayoutAnalysis, groups_from_layout

    frames = _grid_frames(tmp_path, tile_w=1272, tile_h=716)
    analysis = LayoutAnalysis(
        frames=tuple(frames),
        present=(True,) * len(frames),
        intervals=(LayoutInterval(0.0, 5.0, Rect(8, 4, 1360, 764)),),
    )
    [group] = groups_from_layout(analysis, hamming_threshold=43)

    region = group.region
    assert region.x + region.width <= 8 + 1272 + 4, region
    assert region.y + region.height <= 4 + 716 + 4, region
    assert region.x + region.width >= 8 + 1272 - 8, region
    assert region.y + region.height >= 4 + 716 - 8, region


def _copy_frames(artifact: FramesArtifact, dest: Path) -> FramesArtifact:
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in artifact.frames:
        target = dest / frame.path.name
        shutil.copyfile(frame.path, target)
        frames.append(Frame(index=frame.index, timestamp_s=frame.timestamp_s, path=target))
    return FramesArtifact(directory=dest, fps=artifact.fps, frames=tuple(frames))


def test_broken_and_missing_frames_mid_record_do_not_crash(
    switch_frames, tmp_path, caplog
):
    """[A2-2] Битый и отсутствующий кадр в середине — WARNING, а не ValueError."""
    copy = _copy_frames(switch_frames, tmp_path / "frames")
    copy.frames[30].path.write_bytes(b"not a jpeg")
    copy.frames[40].path.unlink()

    with caplog.at_level("WARNING", logger="lecture_transcript.slide_extraction"):
        slides = extract_slides(copy, tmp_path / "out")

    assert len(slides) >= 2
    assert "не читается" in caplog.text


def test_all_frames_unreadable_is_pipeline_error(switch_frames, tmp_path):
    """[A2-2] Не прочитан ни один кадр — `PipelineError` контракта."""
    from lecture_transcript.contracts import PipelineError

    copy = _copy_frames(switch_frames, tmp_path / "frames")
    for frame in copy.frames:
        frame.path.write_bytes(b"not a jpeg")
    with pytest.raises(PipelineError):
        extract_slides(copy, tmp_path / "out")


def test_odd_first_frame_does_not_spoil_region(switch_frames, tmp_path):
    """[A2-3] Кадр другого размера первым в выборке не портит область."""
    baseline = extract_slides(switch_frames, tmp_path / "base")
    copy = _copy_frames(switch_frames, tmp_path / "frames")
    Image.new("L", (640, 480), 200).save(copy.frames[0].path, format="JPEG")

    slides = extract_slides(copy, tmp_path / "out")

    assert [s.region for s in slides] == [s.region for s in baseline]


def test_single_dark_frame_does_not_split_slide(marker_frames, tmp_path):
    """[A2-4] Одиночный тёмный кадр не режет слайд; порог — `min_absence_s`."""
    copy = _copy_frames(marker_frames, tmp_path / "frames")
    path = copy.frames[30].path
    with Image.open(path) as image:
        dark = image.point(lambda v: int(v * 0.25))
    dark.save(path, format="JPEG")

    default = extract_slides(copy, tmp_path / "default", config=_config())
    strict = extract_slides(copy, tmp_path / "strict", config=_config(min_absence_s=0.5))

    assert [(s.start_s, s.end_s) for s in default] == [(0.0, 90.0)]
    assert len(strict) == 2


def test_slide_region_is_the_clamped_crop(marker_frames, clip_marker, tmp_path):
    """[A2-5] Область за краем кадра: `Slide.region` совпадает с PNG."""
    from lecture_transcript.slide_extraction import FrameGroup

    group = FrameGroup(
        frames=marker_frames.frames[:3],
        region=Rect(1800, 1000, 400, 400),
        start_s=0.0,
        end_s=3.0,
    )
    [slide] = build_slides([group], tmp_path, source=clip_marker)
    with Image.open(slide.image_path) as image:
        assert image.size == (slide.region.width, slide.region.height)
    assert (slide.region.width, slide.region.height) == (120, 80)


def test_config_region_is_used(slide_frames, tmp_path):
    """[A2-6] `config.region` при прямом вызове стадии задаёт область."""
    slides = extract_slides(
        slide_frames, tmp_path, config=_config(region=(100, 100, 800, 450))
    )
    assert slides
    assert {s.region for s in slides} == {Rect(100, 100, 800, 450)}
