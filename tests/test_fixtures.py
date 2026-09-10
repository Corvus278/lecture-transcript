"""Проверка тестовых фрагментов (задача 1.5).

Здесь проверяется не пайплайн, а сами фикстуры: что каждый заявленный
фрагмент существует, читается ffprobe, имеет ожидаемую длительность и —
главное — реально содержит то, что заявлено в манифесте:

* ``clip_slide``        — крупная светлая низкодисперсная область есть;
* ``clip_no_slide``     — такой области нет;
* ``clip_slide_switch`` — внутри фрагмента есть скачок perceptual hash;
* ``clip_marker``       — содержимое слайда меняется малыми шагами без скачка;
* ``clip_speech``       — есть речь и есть пауза.

Признаки — те же, что закладываются в пайплайн (design.md D3, D4), поэтому
эти тесты заодно проверяют, что признаки на эталоне вообще работают.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pytest
from PIL import Image

from conftest import CLIP_NAMES, REPO_ROOT, reference_path

pytestmark = pytest.mark.reference

# Ширина, до которой уменьшаются кадры при анализе: детали не нужны,
# а на 480 всё считается мгновенно.
ANALYSIS_WIDTH = 480

# Пороги признака «область демонстрации» (design.md D3).
DIFF_THRESHOLD = 3.0  # медиана |разности соседних кадров|, 0..255
BRIGHT_THRESHOLD = 140.0  # медианная яркость, 0..255
MIN_REGION_AREA_RATIO = 0.06  # доля площади кадра
MIN_REGION_SIDE_RATIO = 0.15  # доля стороны кадра
MIN_REGION_FILL = 0.55  # заполненность bbox компонентом

# Пороги perceptual hash при hash_size=16 (256 бит).
SWITCH_HAMMING = 40  # скачок такого размера — смена слайда
EDIT_HAMMING_MIN = 4  # мелкое изменение — дописывание маркером


# --------------------------------------------------------------------------
# Вспомогательное: чтение медиа
# --------------------------------------------------------------------------


def ffprobe(path: Path) -> dict:
    """Сводка ffprobe по файлу; падает, если файл не читается."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def read_frames(path: Path, fps: float = 1.0, start_s: float = 0.0, duration_s: float | None = None) -> np.ndarray:
    """Кадры видео как массив (N, H, W, 3) в BGR, уменьшенные до ANALYSIS_WIDTH.

    Читается через пайп ffmpeg, без временных файлов.
    """
    cmd = ["ffmpeg", "-v", "error"]
    if start_s:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += [
        "-an",
        "-vf",
        f"fps={fps},scale={ANALYSIS_WIDTH}:-2",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    height = ANALYSIS_WIDTH * 1080 // 1920  # эталон и фрагменты — 16:9 1920x1080
    frame_size = ANALYSIS_WIDTH * height * 3
    assert raw and len(raw) % frame_size == 0, f"неожиданный размер сырого видео из {path.name}"
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, ANALYSIS_WIDTH, 3)


def read_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    """Аудио файла как float32 моно в диапазоне [-1, 1]."""
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "-acodec", "pcm_s16le", "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# --------------------------------------------------------------------------
# Вспомогательное: признаки из design.md
# --------------------------------------------------------------------------


def detect_slide_region(frames: np.ndarray) -> tuple[int, int, int, int] | None:
    """Область демонстрации: крупнейший светлый низкодисперсный прямоугольник.

    Дисперсия считается робастно — как медиана модуля разности соседних кадров.
    Так одно переключение слайда внутри выборки не разрушает признак, а
    движение в тайлах участников по-прежнему даёт высокие значения.

    Возвращает (x, y, w, h) в координатах уменьшенного кадра либо None.
    """
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    if len(gray) < 3:
        raise ValueError("для оценки нужно минимум 3 кадра")
    diff = np.median(np.abs(np.diff(gray, axis=0)), axis=0)
    bright = np.median(gray, axis=0)

    mask = ((diff < DIFF_THRESHOLD) & (bright > BRIGHT_THRESHOLD)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if w < width * MIN_REGION_SIDE_RATIO or h < height * MIN_REGION_SIDE_RATIO:
            continue
        if area / (w * h) < MIN_REGION_FILL:
            continue
        if area < width * height * MIN_REGION_AREA_RATIO:
            continue
        if best is None or area > best[4]:
            best = (int(x), int(y), int(w), int(h), int(area))
    return None if best is None else best[:4]


def region_in_analysis_scale(region_1080p: dict) -> tuple[int, int, int, int]:
    """Пересчёт области из координат кадра 1920x1080 в координаты анализа."""
    k = ANALYSIS_WIDTH / 1920.0
    return (
        int(region_1080p["x"] * k),
        int(region_1080p["y"] * k),
        int(region_1080p["width"] * k),
        int(region_1080p["height"] * k),
    )


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Пересечение над объединением двух прямоугольников."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def crop_hashes(frames: np.ndarray, region: tuple[int, int, int, int]) -> list[imagehash.ImageHash]:
    """Perceptual hash кропа области демонстрации по каждому кадру."""
    x, y, w, h = region
    out = []
    for frame in frames:
        crop = frame[y : y + h, x : x + w]
        out.append(
            imagehash.phash(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), hash_size=16)
        )
    return out


def hash_deltas(hashes: list[imagehash.ImageHash]) -> list[int]:
    """Hamming distance между соседними кадрами."""
    return [hashes[i + 1] - hashes[i] for i in range(len(hashes) - 1)]


def speech_mask(samples: np.ndarray, sample_rate: int = 16000, hop_s: float = 0.05) -> np.ndarray:
    """Грубый VAD по энергии: маска «здесь речь» с шагом hop_s."""
    hop = int(hop_s * sample_rate)
    n = len(samples) // hop
    rms = np.sqrt((samples[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-8)
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 95)
    return db > floor + (peak - floor) * 0.35


def longest_pause_s(mask: np.ndarray, hop_s: float = 0.05) -> float:
    """Самая длинная пауза (непрерывная серия «не речь») в секундах."""
    longest = current = 0
    for value in mask:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest * hop_s


# --------------------------------------------------------------------------
# Манифест
# --------------------------------------------------------------------------


def test_manifest_describes_all_clips(fixtures_manifest: dict) -> None:
    """Манифест описывает ровно тот набор фрагментов, который ждут фикстуры."""
    names = [clip["name"] for clip in fixtures_manifest["clips"]]
    assert sorted(names) == sorted(CLIP_NAMES)
    assert len(names) == len(set(names)), "дубликаты имён фрагментов"


def test_manifest_reference_facts(fixtures_manifest: dict) -> None:
    """В манифесте зафиксированы факты об эталоне и способ его найти."""
    ref = fixtures_manifest["reference"]
    assert ref["env_var"] == "LECTURE_REFERENCE_MP4"
    assert ref["default_path"] == "~/Downloads/wr_20260909_1150.mp4"
    assert 5480 < ref["duration_s"] < 5484, "длительность эталона 1:31:22"
    assert ref["video"]["width"] == 1920 and ref["video"]["height"] == 1080
    assert ref["audio"]["channels"] == 1 and ref["audio"]["sample_rate"] == 48000


def test_manifest_timecodes_are_sane(fixtures_manifest: dict) -> None:
    """Найденные таймкоды непротиворечивы и лежат внутри записи."""
    duration = fixtures_manifest["reference"]["duration_s"]
    tc = fixtures_manifest["timecodes"]

    for key in ("no_demo_intervals_s", "slide_switches_s", "slide_groups_s", "marker_writing_intervals_s"):
        intervals = tc[key]
        assert intervals, f"{key} пуст"
        assert intervals == sorted(intervals), f"{key} не упорядочен по времени"
        for start, end in intervals:
            assert 0 <= start < end <= duration, f"{key}: интервал {start}-{end} вне записи"

    # Интервалы без демонстрации не пересекаются между собой.
    no_demo = tc["no_demo_intervals_s"]
    for (_, end), (start, _) in zip(no_demo, no_demo[1:]):
        assert end <= start, "интервалы без демонстрации пересекаются"

    counts = tc["counts"]
    assert counts["no_demo_intervals"] == len(no_demo)
    assert counts["slide_switches"] == len(tc["slide_switches_s"])
    assert counts["slide_groups"] == len(tc["slide_groups_s"])
    assert counts["marker_writing_intervals"] == len(tc["marker_writing_intervals_s"])
    assert counts["no_demo_intervals_longer_than_30s"] == sum(
        1 for start, end in no_demo if end - start > 30
    )

    # Интервалы фрагментов со слайдом не должны пересекаться
    # с интервалами «демонстрация выключена».
    by_name = {clip["name"]: clip for clip in fixtures_manifest["clips"]}
    for name in ("clip_slide", "clip_marker", "clip_speech"):
        clip = by_name[name]
        for start, end in no_demo:
            assert not (start < clip["end_s"] and end > clip["start_s"]), (
                f"{name} пересекается с интервалом без демонстрации {start}-{end}"
            )


# --------------------------------------------------------------------------
# Файлы фрагментов
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", CLIP_NAMES)
def test_clip_readable_and_has_expected_duration(name: str, request: pytest.FixtureRequest) -> None:
    """Фрагмент существует, читается ffprobe, длительность в ожидаемых пределах."""
    path: Path = request.getfixturevalue(name)
    spec = next(
        c for c in json.loads((REPO_ROOT / "tests" / "fixtures_manifest.json").read_text("utf-8"))["clips"]
        if c["name"] == name
    )
    probe = ffprobe(path)

    kinds = {s["codec_type"] for s in probe["streams"]}
    assert {"video", "audio"} <= kinds, f"{name}: во фрагменте нет видео или аудио"

    duration = float(probe["format"]["duration"])
    expected = spec["duration_s"]
    assert abs(duration - expected) < 1.5, f"{name}: длительность {duration:.2f} вместо {expected}"
    assert 30.0 <= expected <= 90.0, f"{name}: фрагмент должен быть 30–90 с"

    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1920, 1080)


# --------------------------------------------------------------------------
# Содержимое фрагментов
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_clip_slide_contains_slide_region(clip_slide: Path, clip_specs: dict) -> None:
    """В clip_slide есть крупная светлая низкодисперсная область на месте слайда."""
    frames = read_frames(clip_slide, fps=1.0)
    region = detect_slide_region(frames)
    assert region is not None, "область демонстрации не найдена, а должна быть"

    expected = region_in_analysis_scale(clip_specs["clip_slide"]["expect"]["region_1080p"])
    assert iou(region, expected) > 0.6, f"найдена область {region}, ожидалась ~{expected}"

    # Область действительно светлая — это слайд, а не статичная тёмная заглушка.
    x, y, w, h = region
    gray = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2GRAY)
    assert gray[y : y + h, x : x + w].mean() > BRIGHT_THRESHOLD


@pytest.mark.slow
def test_clip_slide_has_no_switch(clip_slide: Path, clip_specs: dict) -> None:
    """Внутри clip_slide слайд не меняется — фрагмент годится как «уверенный»."""
    frames = read_frames(clip_slide, fps=1.0)
    region = region_in_analysis_scale(clip_specs["clip_slide"]["expect"]["region_1080p"])
    deltas = hash_deltas(crop_hashes(frames, region))
    assert max(deltas) < SWITCH_HAMMING, f"неожиданная смена слайда, max hamming={max(deltas)}"


@pytest.mark.slow
def test_clip_no_slide_has_no_slide_region(clip_no_slide: Path) -> None:
    """В clip_no_slide светлой низкодисперсной области нет — демонстрация выключена."""
    frames = read_frames(clip_no_slide, fps=1.0)
    region = detect_slide_region(frames)
    assert region is None, f"область демонстрации найдена ({region}), а её быть не должно"

    # Дополнительно: кадр в целом тёмный, светлых пикселей мало.
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    assert (gray > 150).mean() < 0.10


@pytest.mark.slow
def test_clip_slide_switch_has_hash_jump_inside(clip_slide_switch: Path, clip_specs: dict) -> None:
    """В clip_slide_switch есть скачок phash, и он внутри клипа, а не на границе."""
    spec = clip_specs["clip_slide_switch"]
    frames = read_frames(clip_slide_switch, fps=1.0)
    region = region_in_analysis_scale(spec["expect"]["region_1080p"])
    deltas = hash_deltas(crop_hashes(frames, region))

    peak = int(np.argmax(deltas))
    assert deltas[peak] >= SWITCH_HAMMING, f"скачка phash нет, max hamming={max(deltas)}"

    duration = spec["duration_s"]
    offset = peak + 0.5  # шаг выборки 1 с, скачок между кадрами peak и peak+1
    assert 0.15 * duration < offset < 0.85 * duration, (
        f"скачок на {offset:.1f} с — слишком близко к границе фрагмента"
    )
    low, high = spec["expect"]["switch_at_offset_s"]
    assert low - 2.0 <= offset <= high + 2.0, (
        f"скачок на {offset:.1f} с, в манифесте заявлено {low}–{high} с"
    )


@pytest.mark.slow
def test_clip_marker_changes_gradually(clip_marker: Path, clip_specs: dict) -> None:
    """В clip_marker слайд один, но содержимое прирастает — это дописывание маркером."""
    spec = clip_specs["clip_marker"]
    frames = read_frames(clip_marker, fps=1.0)
    region = region_in_analysis_scale(spec["expect"]["region_1080p"])
    hashes = crop_hashes(frames, region)
    deltas = hash_deltas(hashes)

    assert max(deltas) < SWITCH_HAMMING, (
        f"внутри фрагмента смена слайда (max hamming={max(deltas)}), ожидалось дописывание"
    )
    edits = [d for d in deltas if d >= EDIT_HAMMING_MIN]
    assert len(edits) >= 5, f"содержимое почти не менялось, изменений={len(edits)}"

    # Накопленное изменение первый->последний кадр заметно больше шага:
    # это прирост содержимого, а не дрожание кодека.
    assert (hashes[-1] - hashes[0]) > max(deltas), "содержимое слайда не приросло за фрагмент"


@pytest.mark.slow
def test_clip_speech_has_speech_and_pause(clip_speech: Path, clip_specs: dict) -> None:
    """В clip_speech есть речь и есть различимая пауза."""
    expect = clip_specs["clip_speech"]["expect"]
    mask = speech_mask(read_audio(clip_speech))
    ratio = float(mask.mean())
    pause = longest_pause_s(mask)

    assert ratio >= expect["speech_ratio_min"], f"речи слишком мало: {ratio:.2f}"
    assert ratio < 0.98, "речь без пауз — фрагмент не годится для проверки VAD"
    assert pause >= expect["min_pause_s"], f"самая длинная пауза {pause:.2f} с — слишком коротко"


# --------------------------------------------------------------------------
# Сверка заявленных таймкодов с самой эталонной записью
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_reference_probes_match_manifest(reference_mp4: Path, fixtures_manifest: dict) -> None:
    """Опорные таймкоды 3.2 подтверждаются на самой записи, а не только в манифесте."""
    probes = fixtures_manifest["timecodes"]["task_3_2_probes"]

    for moment in probes["with_slide_s"]:
        frames = read_frames(reference_mp4, fps=1.0, start_s=moment - 5.0, duration_s=10.0)
        assert detect_slide_region(frames) is not None, f"на t={moment} слайд не найден"

    moment = probes["without_slide_s"]
    frames = read_frames(reference_mp4, fps=1.0, start_s=moment - 1.0, duration_s=8.0)
    assert detect_slide_region(frames) is None, f"на t={moment} слайд найден, а не должен"


@pytest.mark.slow
def test_reference_demo_starts_at_declared_moment(
    reference_mp4: Path, fixtures_manifest: dict
) -> None:
    """Демонстрация включается в заявленный момент: до него слайда нет, после — есть."""
    moment = fixtures_manifest["timecodes"]["demo_on_at_s"]
    before = read_frames(reference_mp4, fps=1.0, start_s=max(0.0, moment - 25.0), duration_s=20.0)
    after = read_frames(reference_mp4, fps=1.0, start_s=moment + 5.0, duration_s=20.0)
    assert detect_slide_region(before) is None, "до включения демонстрации слайд не должен находиться"
    assert detect_slide_region(after) is not None, "после включения демонстрации слайд должен находиться"


def test_reference_path_from_env_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Путь к эталону берётся из LECTURE_REFERENCE_MP4, а не зашит в код."""
    fake = tmp_path / "somewhere" / "other.mp4"
    monkeypatch.setenv("LECTURE_REFERENCE_MP4", str(fake))
    assert reference_path() == fake
    monkeypatch.delenv("LECTURE_REFERENCE_MP4")
    assert reference_path() == Path("~/Downloads/wr_20260909_1150.mp4").expanduser()
