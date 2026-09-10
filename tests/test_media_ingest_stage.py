"""Стадии `audio` и `frames` как точки входа оркестратора и их кэш (design D8).

Проверяется требование «Переиспользование промежуточных артефактов»:
повторный прогон с теми же параметрами обязан не запускать декодирование
заново, а смена параметров — извлекать заново **только свой** артефакт.

Ключевой критерий здесь — **факт запуска ffmpeg**, а не наличие файлов на
диске: артефакт остаётся на месте и при полном пересчёте, поэтому проверка
«файл существует» не поймала бы отсутствие кэша.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import types
from pathlib import Path
from typing import Any, Callable

import pytest

from lecture_transcript import cli
from lecture_transcript.cache import Cache, StageCache
from lecture_transcript.config import PipelineConfig, load_config
from lecture_transcript.contracts import MissingAudioTrackError
from lecture_transcript.media_ingest import _ffmpeg as ffmpeg_util
from lecture_transcript.media_ingest import (
    AUDIO_NAME,
    FRAMES_DIR_NAME,
    audio_artifact,
    frames_artifact,
    run_audio_stage,
    run_frames_stage,
)

_TOOLS = ("ffmpeg", "ffprobe")


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------


def _make(*args: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"не удалось подготовить фикстуру: {result.stderr[-500:]}")


@pytest.fixture(scope="session")
def stage_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Клип 6 с со звуком — быстрый вход для полного прогона стадии."""
    path = tmp_path_factory.mktemp("stage_input") / "lecture.mp4"
    _make(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    )
    return path


@pytest.fixture(scope="session")
def stage_two_tracks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Видео + две аудиодорожки — вход, где смена `audio_track` осмысленна."""
    path = tmp_path_factory.mktemp("stage_two") / "lecture_two.mp4"
    _make(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=1200:sample_rate=48000:duration=6",
        "-map", "0:v", "-map", "1:a", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    )
    return path


@pytest.fixture(scope="session")
def stage_audio_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("stage_audio") / "lecture_audio.m4a"
    _make(
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3",
        "-c:a", "aac", "-vn", str(path),
    )
    return path


@pytest.fixture(scope="session")
def stage_video_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("stage_video") / "lecture_video.mp4"
    _make(
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
    )
    return path


@pytest.fixture
def ffmpeg_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Журнал всех запусков ffmpeg/ffprobe из пакета media_ingest."""
    calls: list[list[str]] = []
    real_run = subprocess.run

    def counting(args, **kwargs):  # noqa: ANN001, ANN202 — обёртка subprocess.run
        calls.append([str(a) for a in args])
        return real_run(args, **kwargs)

    monkeypatch.setattr(ffmpeg_util, "subprocess", types.SimpleNamespace(run=counting))
    return calls


def _decodes(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c and Path(c[0]).name in _TOOLS]


def _run_cli(source: Path, out_dir: Path, *extra: str) -> int:
    """Прогон CLI по одной стадии `frames`.

    `--only-stage` здесь обязателен: эти тесты про кэш стадии кадров, а не
    про весь пайплайн, и на полном графе прогон честно возвращает 1 —
    остальные стадии ещё не реализованы.
    """
    return cli.main(
        [
            str(source),
            "--out-dir", str(out_dir),
            "--only-stage", "frames",
            "--hwaccel", "none",
            "--log-level", "WARNING",
            *extra,
        ]
    )


def _stage_dirs(out_dir: Path, stage: str = "frames") -> list[Path]:
    """Каталоги ячеек кэша стадии (по одному на набор параметров)."""
    root = out_dir / ".cache" / stage
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Имя -> (размер, mtime_ns): пересчёт кадров заметен даже при том же имени."""
    return {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(directory.iterdir())
    }


def _config(**overrides: Any) -> PipelineConfig:
    config = load_config()
    if not overrides:
        return config
    return dataclasses.replace(
        config, media_ingest=dataclasses.replace(config.media_ingest, **overrides)
    )


def _context(source: Path, build_dir: Path, config: PipelineConfig, stage: str):
    build_dir.mkdir(parents=True, exist_ok=True)
    return cli.StageContext(
        stage=stage,
        source=source,
        out_dir=build_dir.parent,
        build_dir=build_dir,
        config=config,
    )


def _run_cached(
    cell: StageCache,
    func: Callable[[Any], dict[str, Any]],
    source: Path,
    config: PipelineConfig,
) -> tuple[dict[str, Any], Path]:
    """Мини-оркестратор: то же, что делает `cli` вокруг одной ячейки кэша."""
    if cell.hit():
        return cell.load(), cell.path
    build_dir = cell.new_build_dir()
    payload = func(_context(source, build_dir, config, cell.name))
    cell.save(payload, build_dir=build_dir)
    return payload, cell.path


# ==========================================================================
# Независимость ячеек кэша: `audio` и `frames` — две корневые стадии
# ==========================================================================


def _cells(cache: Cache, *, audio_track: int | None, fps: float) -> tuple[StageCache, StageCache]:
    """Ячейки обеих стадий.

    Обе строятся как корневые (`previous=None`): ни одна не потребляет
    результат другой, поэтому связывать их в линейную цепочку нельзя —
    иначе ключ второй включит ключ первой и независимость пропадёт.
    """
    return (
        cache.stage("audio", {"audio_track": audio_track, "audio_sample_rate": 16000}),
        cache.stage("frames", {"frames_fps": fps, "hwaccel": "none"}),
    )


def test_changed_audio_track_does_not_reextract_frames(
    stage_two_tracks: Path, tmp_path: Path, ffmpeg_calls: list[list[str]]
) -> None:
    """Смена аудиодорожки пересчитывает только WAV; 5500 кадров не трогаются."""
    cache = Cache(tmp_path / ".cache", stage_two_tracks)
    audio_cell, frames_cell = _cells(cache, audio_track=1, fps=1.0)

    _run_cached(audio_cell, run_audio_stage, stage_two_tracks, _config(audio_track=1))
    frames_payload, frames_dir = _run_cached(
        frames_cell, run_frames_stage, stage_two_tracks, _config(frames_fps=1.0, hwaccel="none")
    )
    before = _snapshot(frames_dir / FRAMES_DIR_NAME)
    assert before

    # тот же файл, другая дорожка
    audio_cell2, frames_cell2 = _cells(cache, audio_track=2, fps=1.0)
    assert frames_cell2.key == frames_cell.key, "ключ кадров зависит от аудиопараметров"

    ffmpeg_calls.clear()
    payload2, frames_dir2 = _run_cached(
        frames_cell2, run_frames_stage, stage_two_tracks, _config(frames_fps=1.0, hwaccel="none")
    )
    assert _decodes(ffmpeg_calls) == [], "смена аудиодорожки заставила переизвлечь кадры"
    assert payload2 == frames_payload
    assert _snapshot(frames_dir2 / FRAMES_DIR_NAME) == before

    ffmpeg_calls.clear()
    audio_payload2, _ = _run_cached(
        audio_cell2, run_audio_stage, stage_two_tracks, _config(audio_track=2)
    )
    assert _decodes(ffmpeg_calls), "смена аудиодорожки обязана пересчитать WAV"
    assert audio_payload2["audio"]["source_track_index"] == 2


def test_changed_fps_does_not_reextract_audio(
    stage_clip: Path, tmp_path: Path, ffmpeg_calls: list[list[str]]
) -> None:
    """Смена частоты выборки кадров не трогает уже извлечённый WAV."""
    cache = Cache(tmp_path / ".cache", stage_clip)
    audio_cell, frames_cell = _cells(cache, audio_track=None, fps=1.0)

    audio_payload, audio_dir = _run_cached(
        audio_cell, run_audio_stage, stage_clip, _config()
    )
    _run_cached(frames_cell, run_frames_stage, stage_clip, _config(frames_fps=1.0, hwaccel="none"))
    wav_mtime = (audio_dir / AUDIO_NAME).stat().st_mtime_ns

    audio_cell2, frames_cell2 = _cells(cache, audio_track=None, fps=2.0)
    assert audio_cell2.key == audio_cell.key, "ключ аудио зависит от параметров кадров"
    assert frames_cell2.key != frames_cell.key

    ffmpeg_calls.clear()
    payload2, audio_dir2 = _run_cached(audio_cell2, run_audio_stage, stage_clip, _config())
    assert _decodes(ffmpeg_calls) == [], "смена fps заставила переизвлечь аудио"
    assert payload2 == audio_payload
    assert (audio_dir2 / AUDIO_NAME).stat().st_mtime_ns == wav_mtime

    ffmpeg_calls.clear()
    _run_cached(frames_cell2, run_frames_stage, stage_clip, _config(frames_fps=2.0, hwaccel="none"))
    assert _decodes(ffmpeg_calls), "смена fps обязана переизвлечь кадры"
    assert len(_stage_dirs(tmp_path, "frames")) == 2, "прежние кадры не сохранены"


def test_repeated_run_of_both_stages_does_not_decode(
    stage_clip: Path, tmp_path: Path, ffmpeg_calls: list[list[str]]
) -> None:
    """Повтор с теми же параметрами: ни одного запуска ffmpeg/ffprobe."""
    cache = Cache(tmp_path / ".cache", stage_clip)
    for _ in range(1):
        audio_cell, frames_cell = _cells(cache, audio_track=None, fps=1.0)
        _run_cached(audio_cell, run_audio_stage, stage_clip, _config())
        _run_cached(frames_cell, run_frames_stage, stage_clip, _config(hwaccel="none"))
    assert _decodes(ffmpeg_calls)

    ffmpeg_calls.clear()
    audio_cell, frames_cell = _cells(cache, audio_track=None, fps=1.0)
    assert audio_cell.hit() and frames_cell.hit()
    _run_cached(audio_cell, run_audio_stage, stage_clip, _config())
    _run_cached(frames_cell, run_frames_stage, stage_clip, _config(hwaccel="none"))
    assert _decodes(ffmpeg_calls) == []


# ==========================================================================
# Стадии не пересекаются по артефактам
# ==========================================================================


def test_audio_stage_writes_only_wav(stage_clip: Path, tmp_path: Path) -> None:
    ctx = _context(stage_clip, tmp_path / "build", _config(), "audio")
    payload = run_audio_stage(ctx)

    json.dumps(payload)  # нагрузка обязана быть JSON-совместимой для meta.json
    assert payload["audio"]["path"] == AUDIO_NAME
    assert not Path(payload["audio"]["path"]).is_absolute()
    assert "frames" not in payload
    assert sorted(p.name for p in ctx.build_dir.iterdir()) == [AUDIO_NAME]

    audio = audio_artifact(payload, ctx.build_dir)
    assert audio.sample_rate == 16000
    assert audio.duration_s == pytest.approx(payload["duration_s"], abs=0.25)


def test_frames_stage_writes_only_frames(stage_clip: Path, tmp_path: Path) -> None:
    ctx = _context(stage_clip, tmp_path / "build", _config(hwaccel="none", frames_fps=1.0), "frames")
    payload = run_frames_stage(ctx)

    json.dumps(payload)
    assert payload["frames"]["directory"] == FRAMES_DIR_NAME
    assert "audio" not in payload
    assert sorted(p.name for p in ctx.build_dir.iterdir()) == [FRAMES_DIR_NAME]
    assert not (ctx.build_dir / AUDIO_NAME).exists()

    frames = frames_artifact(payload, ctx.build_dir)
    assert frames is not None
    assert len(frames.frames) == payload["frames"]["count"]
    assert frames.frames[1].timestamp_s == pytest.approx(1.0)


def test_frames_stage_in_audio_mode_returns_no_frames(
    stage_audio_only: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Файл без видео: стадия кадров не падает, но предупреждает о неполноте."""
    ctx = _context(stage_audio_only, tmp_path / "build", _config(), "frames")

    with caplog.at_level(logging.WARNING, logger="lecture_transcript.media_ingest"):
        payload = run_frames_stage(ctx)

    assert payload["has_video"] is False
    assert payload["frames"] is None
    assert frames_artifact(payload, ctx.build_dir) is None
    assert not (ctx.build_dir / FRAMES_DIR_NAME).exists()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("неполн" in m for m in warnings), warnings


def test_audio_stage_works_without_video(stage_audio_only: Path, tmp_path: Path) -> None:
    ctx = _context(stage_audio_only, tmp_path / "build", _config(), "audio")
    payload = run_audio_stage(ctx)
    assert payload["has_video"] is False
    assert audio_artifact(payload, ctx.build_dir).path.is_file()


def test_audio_stage_fails_without_audio_track(
    stage_video_only: Path, tmp_path: Path
) -> None:
    """Видео без звука: падает только стадия `audio`, кадры извлекаются."""
    audio_ctx = _context(stage_video_only, tmp_path / "a", _config(), "audio")
    with pytest.raises(MissingAudioTrackError):
        run_audio_stage(audio_ctx)
    assert not (audio_ctx.build_dir / AUDIO_NAME).exists()

    frames_ctx = _context(stage_video_only, tmp_path / "f", _config(hwaccel="none"), "frames")
    payload = run_frames_stage(frames_ctx)
    assert payload["frames"]["count"] > 0


# ==========================================================================
# Стадия `frames` через оркестратор (как её зовёт cli сегодня)
# ==========================================================================


def test_second_run_reuses_artifacts_and_does_not_decode(
    stage_clip: Path, tmp_path: Path, ffmpeg_calls: list[list[str]]
) -> None:
    """Повторный прогон CLI с теми же параметрами не запускает ffmpeg вообще."""
    out_dir = tmp_path / "out"

    assert _run_cli(stage_clip, out_dir) == 0
    assert _decodes(ffmpeg_calls), "первый прогон обязан декодировать"

    stage_dirs = _stage_dirs(out_dir)
    assert len(stage_dirs) == 1, stage_dirs
    frames_dir = stage_dirs[0] / FRAMES_DIR_NAME
    before_frames = _snapshot(frames_dir)
    assert before_frames

    ffmpeg_calls.clear()
    assert _run_cli(stage_clip, out_dir) == 0

    assert _decodes(ffmpeg_calls) == [], (
        "повторный прогон декодирует заново: "
        f"{[c[0] for c in _decodes(ffmpeg_calls)]}"
    )
    assert _snapshot(frames_dir) == before_frames
    assert _stage_dirs(out_dir) == stage_dirs


def test_changed_fps_invalidates_frames(
    stage_clip: Path, tmp_path: Path, ffmpeg_calls: list[list[str]]
) -> None:
    """Другая частота выборки -> кадры извлекаются заново, в свою ячейку."""
    out_dir = tmp_path / "out"

    assert _run_cli(stage_clip, out_dir, "--frames-fps", "1") == 0
    first = _stage_dirs(out_dir)
    assert len(first) == 1

    ffmpeg_calls.clear()
    assert _run_cli(stage_clip, out_dir, "--frames-fps", "2") == 0

    assert _decodes(ffmpeg_calls), "смена fps не вызвала повторного извлечения"
    stage_dirs = _stage_dirs(out_dir)
    assert len(stage_dirs) == 2, "артефакт с прежними параметрами не сохранён"

    counts = sorted(len(list((d / FRAMES_DIR_NAME).iterdir())) for d in stage_dirs)
    assert counts[1] > counts[0], counts


def test_disabled_cache_forces_decoding(
    stage_clip: Path, tmp_path: Path, ffmpeg_calls: list[list[str]]
) -> None:
    """`--no-cache` — обратная проверка: без кэша ffmpeg зовётся каждый раз."""
    out_dir = tmp_path / "out"
    assert _run_cli(stage_clip, out_dir) == 0

    ffmpeg_calls.clear()
    assert _run_cli(stage_clip, out_dir, "--no-cache") == 0
    assert _decodes(ffmpeg_calls), "с выключенным кэшем стадия обязана считаться"
