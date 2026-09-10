"""Тесты точки входа CLI и конфигурации (задача 1.3)."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lecture_transcript import cli
from lecture_transcript.config import (
    PLACEHOLDER_DEFAULTS,
    STAGES,
    ConfigError,
    PipelineConfig,
    load_config,
)


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="нет ffmpeg/ffprobe в окружении",
)


def _help_command() -> list[str]:
    """Реальный запуск CLI: консольный скрипт, иначе python -c."""
    script = Path(sys.executable).with_name("lecture-transcript")
    if script.is_file():
        return [str(script), "--help"]
    return [
        sys.executable,
        "-c",
        "from lecture_transcript.cli import main; raise SystemExit(main())",
        "--help",
    ]


@pytest.fixture(scope="module")
def help_output() -> str:
    result = subprocess.run(
        _help_command(), capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_help_prints_backend_selection(help_output: str) -> None:
    """Проверка 1.3: --help печатает параметры выбора ASR- и OCR-бэкенда."""
    assert "--asr-backend" in help_output
    assert "--ocr-backend" in help_output
    assert "бэкенд распознавания речи" in help_output
    assert "бэкенд распознавания слайдов" in help_output


def test_help_lists_registered_backend_names(help_output: str) -> None:
    """Список допустимых значений печатается, даже если веса не скачаны.

    Раньше `--help` брал имена из `available_backend_names()` (готовые к
    работе) и на машине без torch/paddle печатал «реестр ещё не доступен»,
    хотя реестр знает пять имён.
    """
    for kind in ("ocr", "asr"):
        names = cli.known_backends(kind)
        if not names:
            pytest.skip(f"реестр {kind}-бэкендов недоступен в этом окружении")
        for name in names:
            assert name in help_output, name
    assert "реестр бэкендов ещё не доступен" not in help_output


def test_help_prints_pipeline_options(help_output: str) -> None:
    for option in (
        "--frames-fps",
        "--hwaccel",
        "--cache-dir",
        "--no-cache",
        "--config",
        "--from-stage",
        "--only-stage",
        "--log-level",
    ):
        assert option in help_output, option
    for stage in STAGES:
        assert stage in help_output, stage


def test_help_survives_absent_backend_registries(monkeypatch) -> None:
    """Реестров 4.1/5.1 может ещё не быть — CLI обязан работать без них."""
    monkeypatch.setattr(cli, "_registry", lambda kind: None)
    assert cli.available_backends("ocr") == []
    assert cli.available_backends("asr") == []
    text = cli.build_parser().format_help()
    assert "--asr-backend" in text and "--ocr-backend" in text


def test_registry_names_land_in_help(monkeypatch) -> None:
    class FakeRegistry:
        @staticmethod
        def list_backend_names():
            return ["gigaam-v2", "whisper-large-v3"]

        @staticmethod
        def available_backend_names():
            return ["gigaam-v2"]

    monkeypatch.setattr(
        cli, "_registry", lambda kind: FakeRegistry if kind == "asr" else None
    )
    text = cli.build_parser().format_help()
    assert "gigaam-v2" in text and "whisper-large-v3" in text
    # незарегистрированное имя не печатается как доступное
    assert "готовы сейчас: gigaam-v2" in text


# --------------------------------------------------------------------------
# Подключение стадий к пакетам
# --------------------------------------------------------------------------


def test_every_stage_has_an_entrypoint() -> None:
    """Каждая стадия графа подключена: точка входа пакета либо адаптер."""
    unwired = [stage for stage in STAGES if cli._stage_callable(stage) is None]
    assert not unwired, f"стадии без реализации: {unwired}"


def test_package_entrypoint_wins_over_adapter(monkeypatch) -> None:
    """Как только пакет объявит run_*_stage, адаптер уступает ему место."""
    import types

    fake = types.SimpleNamespace(run_slides_stage=lambda ctx: {"ok": True})
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name, *a, **kw: fake if name.endswith("slide_extraction") else __import__(name),
    )
    assert cli._stage_callable("slides") is fake.run_slides_stage


def test_stage_payload_codecs_roundtrip(tmp_path: Path) -> None:
    """Нагрузки стадий JSON-совместимы и разворачиваются обратно в контракты."""
    from lecture_transcript.contracts import (
        OcrFragment,
        Rect,
        Slide,
        SlideOcr,
        Transcription,
        Word,
    )

    base = tmp_path / "cell"
    (base / "slides").mkdir(parents=True)
    image = base / "slides" / "001.png"
    image.write_bytes(b"png")

    slide = Slide(
        index=1,
        start_s=1.0,
        end_s=9.5,
        region=Rect(10, 20, 300, 400),
        representative_timestamp_s=8.0,
        image_path=image,
    )
    payload = json.loads(json.dumps(cli._slide_json(slide, base)))
    assert cli._slide_obj(payload, base) == slide

    ocr = SlideOcr(
        slide_index=1,
        image_path=image,
        markdown="# слайд",
        fragments=(
            OcrFragment(
                text="$x^2$",
                kind="formula",
                confidence=0.42,
                bbox=Rect(1, 2, 3, 4),
                low_confidence=True,
            ),
            OcrFragment(text="Пример 2.", kind="text", confidence=0.99),
        ),
        unreliable=True,
        backend="hybrid",
    )
    payload = json.loads(json.dumps(cli._ocr_json(ocr, base)))
    assert cli._ocr_obj(payload, base) == ocr

    transcription = Transcription(
        words=(Word("привет", 0.0, 0.5, 0.9), Word("мир", 0.5, 1.0, None)),
        backend="gigaam-v2",
        has_punctuation=False,
        used_glossary=True,
        warnings=("бэкенд без глоссария",),
    )
    payload = json.loads(json.dumps(cli._transcription_json(transcription)))
    assert cli._transcription_obj(payload) == transcription


def test_stage_api_mismatch_is_reported_not_crashed(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Разъехавшийся API пакета — понятное сообщение и ненулевой код."""

    def broken(ctx: cli.StageContext):
        raise TypeError("extract_slides() got an unexpected keyword argument 'config'")

    monkeypatch.setattr(
        cli, "_stage_callable", lambda stage: broken if stage == "slides" else None
    )
    monkeypatch.setattr(cli, "STAGES", ("slides",))
    monkeypatch.setattr(cli, "STAGE_DEPENDENCIES", {"slides": ()})
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    code = cli.main([str(source), "--out-dir", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "API пакета не стыкуется" in err
    assert "run_slides_stage(ctx)" in err


# --------------------------------------------------------------------------
# Ручная область демонстрации (design, Risks)
# --------------------------------------------------------------------------


def test_slide_region_flag_parses_and_overrides_config() -> None:
    parser = cli.build_parser()
    config = load_config()
    assert config.slide_extraction.region is None

    args = parser.parse_args(["a.mp4", "--slide-region", "10, 20,1280,720"])
    applied = cli.apply_overrides(config, args)
    assert applied.slide_extraction.region == (10, 20, 1280, 720)

    # auto сбрасывает область, заданную в конфиге, обратно на автодетект
    preset = dataclasses.replace(
        config,
        slide_extraction=dataclasses.replace(
            config.slide_extraction, region=(1, 2, 3, 4)
        ),
    )
    args = parser.parse_args(["a.mp4", "--slide-region", "auto"])
    assert cli.apply_overrides(preset, args).slide_extraction.region is None

    # без флага область из конфига не трогается
    args = parser.parse_args(["a.mp4"])
    assert cli.apply_overrides(preset, args).slide_extraction.region == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "garbage",
    ["10,20,300", "a,b,c,d", "10,20,0,400", "-1,0,300,400", "1.5,2,3,4", ""],
)
def test_slide_region_flag_rejects_garbage(garbage: str, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["a.mp4", "--slide-region", garbage])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--slide-region" in err
    assert "X,Y,W,H" in err or "больше нуля" in err or "отрицательными" in err


def test_slide_region_in_yaml(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("slide_extraction:\n  region: [10, 20, 300, 400]\n", encoding="utf-8")
    assert load_config(good).slide_extraction.region == (10, 20, 300, 400)

    bad = tmp_path / "bad.yaml"
    bad.write_text("slide_extraction:\n  region: [10, 20, 300]\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(bad)
    assert "slide_extraction.region" in str(exc.value)


def test_slides_adapter_delivers_config_to_extract_slides(
    tmp_path: Path, monkeypatch
) -> None:
    """Пороги из конфига и ручная область реально доходят до стадии.

    Шпионим на уровне `analyze_layout` / `groups_from_layout` внутри
    `slide_extraction.pipeline`: так проверяется не то, что адаптер что-то
    передал, а то, что стадия получила именно значения из конфига.
    """
    from unittest import mock

    from PIL import Image

    from lecture_transcript.contracts import Rect
    from lecture_transcript.slide_extraction import pipeline

    frames_base = tmp_path / "frames_cell"
    (frames_base / "frames").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (64, 36), (200, 200, 200)).save(
            frames_base / "frames" / f"frame_{i + 1:06d}.jpg"
        )
    frames_payload = {
        "frames": {"directory": "frames", "fps": 1.0, "count": 3, "suffix": ".jpg"}
    }

    config = load_config()
    marked = dataclasses.replace(
        config,
        slide_extraction=dataclasses.replace(
            config.slide_extraction,
            max_spread=11.5,
            min_brightness=123.0,
            min_absence_s=7.0,
            hash_size=12,
            hamming_threshold=33,
            region=(5, 6, 40, 20),
        ),
    )

    captured: dict[str, dict] = {}

    class Stop(Exception):
        pass

    def fake_analyze(*args, **kwargs):
        captured["analyze"] = kwargs
        return mock.MagicMock()

    def fake_groups(*args, **kwargs):
        captured["groups"] = kwargs
        raise Stop

    monkeypatch.setattr(pipeline, "analyze_layout", fake_analyze)
    monkeypatch.setattr(pipeline, "groups_from_layout", fake_groups)

    build = tmp_path / "build"
    build.mkdir()
    ctx = cli.StageContext(
        stage="slides",
        source=tmp_path / "lecture.mp4",
        out_dir=tmp_path / "out",
        build_dir=build,
        config=marked,
        results={"frames": frames_payload},
        stage_dirs={"frames": frames_base},
    )
    with pytest.raises(Stop):
        cli._adapt_slides(ctx)

    analyze = captured["analyze"]
    assert analyze["max_spread"] == 11.5
    assert analyze["min_brightness"] == 123.0
    assert analyze["min_absence_s"] == 7.0
    assert analyze["region"] == Rect(5, 6, 40, 20)
    groups = captured["groups"]
    assert groups["hash_size"] == 12
    assert groups["hamming_threshold"] == 33


# --------------------------------------------------------------------------
# Аудит 2: результат merge на диске при попадании в кэш, осиротевшие PNG,
# --no-cache с частичной выборкой
# --------------------------------------------------------------------------


def _merge_pipeline(monkeypatch, holder: dict) -> None:
    """Все стадии застублены, `merge` — настоящий адаптер (как у аудитора)."""
    from PIL import Image

    from lecture_transcript.contracts import (
        OcrFragment,
        Rect,
        Slide,
        SlideOcr,
        Transcription,
        Word,
    )

    def slides_stub(ctx):
        images = ctx.build_dir / "slides"
        images.mkdir(parents=True, exist_ok=True)
        items = []
        for i in range(1, holder["count"] + 1):
            image = images / f"slide_{i:03d}.png"
            Image.new("RGB", (32, 18), (255, 255, 255)).save(image)
            slide = Slide(
                index=i,
                start_s=(i - 1) * 10.0,
                end_s=i * 10.0,
                region=Rect(0, 0, 32, 18),
                representative_timestamp_s=i * 10.0 - 1.0,
                image_path=image,
            )
            items.append(cli._slide_json(slide, ctx.build_dir))
        return {"slides": items}

    def ocr_stub(ctx):
        base = ctx.stage_dirs["slides"]
        items = []
        for data in ctx.results["slides"]["slides"]:
            slide = cli._slide_obj(data, base)
            ocr = SlideOcr(
                slide_index=slide.index,
                image_path=slide.image_path,
                fragments=(OcrFragment(f"Пример {slide.index}", "text", 0.99),),
                markdown=f"Пример {slide.index}",
                backend="fake",
            )
            items.append(cli._ocr_json(ocr, base))
        return {"ocr": items}

    def punctuation_stub(ctx):
        return cli._transcription_json(
            Transcription(
                words=(Word("Начнём.", 1.0, 1.5, 0.9), Word("Дальше.", 12.0, 12.5, 0.9)),
                backend="fake",
                has_punctuation=True,
            )
        )

    real = {"merge": cli._adapt_merge}
    stubs = {"slides": slides_stub, "ocr": ocr_stub, "punctuation": punctuation_stub}

    def dispatch(stage):
        if stage in real:
            return real[stage]
        return stubs.get(stage, lambda ctx: {})

    monkeypatch.setattr(cli, "_stage_callable", dispatch)
    monkeypatch.setattr(cli, "_STAGE_MODEL_HOLDERS", {})


def _pngs(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("*.png")) if directory.is_dir() else []


def test_merge_cache_hit_restores_transcript_and_slides(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Сценарий аудитора: удалить transcript.md, оставить .cache, перезапустить."""
    _merge_pipeline(monkeypatch, {"count": 2})
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    out_dir = tmp_path / "out"
    args = [str(source), "--out-dir", str(out_dir), "--log-level", "INFO"]

    assert cli.main(args) == 0
    transcript = out_dir / "transcript.md"
    assert transcript.is_file()
    first = transcript.read_text(encoding="utf-8")
    assert _pngs(out_dir / "slides") == ["slide_001.png", "slide_002.png"]

    transcript.unlink()
    shutil.rmtree(out_dir / "slides")
    capsys.readouterr()

    assert cli.main(args) == 0
    log = capsys.readouterr().err
    assert "стадия merge: попадание в кэш" in log, "merge должен был взяться из кэша"
    assert transcript.is_file(), "попадание в кэш не восстановило transcript.md"
    assert transcript.read_text(encoding="utf-8") == first
    assert _pngs(out_dir / "slides") == ["slide_001.png", "slide_002.png"]


def test_transcript_header_has_recording_duration(
    tmp_path: Path, monkeypatch
) -> None:
    """Длительность записи из нагрузки `audio` доходит до шапки transcript.md."""
    from lecture_transcript.transcript_assembly import format_timecode

    _merge_pipeline(monkeypatch, {"count": 1})
    staged = cli._stage_callable

    def with_duration(stage):
        if stage == "audio":
            return lambda ctx: {"duration_s": 5482.0, "audio": {"path": "audio.wav"}}
        return staged(stage)

    monkeypatch.setattr(cli, "_stage_callable", with_duration)
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    out_dir = tmp_path / "out"
    assert cli.main([str(source), "--out-dir", str(out_dir)]) == 0

    text = (out_dir / "transcript.md").read_text(encoding="utf-8")
    header = text.split("## ", 1)[0]
    assert "Длительность" in header, header
    assert format_timecode(5482.0, with_hours=True) in header, header


def test_short_offslide_threshold_reaches_assembly(tmp_path: Path, monkeypatch) -> None:
    """Порог присоединения коротких секций из конфига доходит до сборки,
    а не подменяется дефолтом пакета через getattr."""
    from lecture_transcript.transcript_assembly import consolidate

    _merge_pipeline(monkeypatch, {"count": 1})
    seen: list[float] = []
    original = consolidate.absorb_short_offslide_sections

    def spy(sections, threshold, *args, **kwargs):
        seen.append(threshold)
        return original(sections, threshold, *args, **kwargs)

    monkeypatch.setattr(consolidate, "absorb_short_offslide_sections", spy)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("transcript_assembly:\n  short_offslide_section_s: 7.5\n", encoding="utf-8")
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    code = cli.main([str(source), "--out-dir", str(tmp_path / "out"), "--config", str(cfg)])
    assert code == 0
    assert seen == [7.5], seen


def test_recording_duration_prefers_audio_then_frames() -> None:
    assert cli._recording_duration({"audio": {"duration_s": 10.0}, "frames": {"duration_s": 9.0}}) == 10.0
    assert cli._recording_duration({"audio": {}, "frames": {"duration_s": 9.0}}) == 9.0
    assert cli._recording_duration({"audio": {}, "frames": {}}) is None


def test_merge_hit_without_transcript_in_cell_is_a_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Повреждённая ячейка merge: «выполнено», а файла нет — код не 0."""
    _merge_pipeline(monkeypatch, {"count": 1})
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    out_dir = tmp_path / "out"
    args = [str(source), "--out-dir", str(out_dir)]
    assert cli.main(args) == 0

    (out_dir / "transcript.md").unlink()
    for cell in (out_dir / ".cache" / "merge").iterdir():
        (cell / "transcript.md").unlink()
    capsys.readouterr()

    assert cli.main(args) == 1
    assert "отмечена выполненной" in capsys.readouterr().err


def test_rerun_with_fewer_slides_removes_orphan_pngs(
    tmp_path: Path, monkeypatch
) -> None:
    """Меньше слайдов во втором прогоне — лишние slide_NNN.png убраны,
    посторонние файлы в каталоге не тронуты."""
    holder = {"count": 3}
    _merge_pipeline(monkeypatch, holder)
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    out_dir = tmp_path / "out"

    assert cli.main([str(source), "--out-dir", str(out_dir)]) == 0
    images = out_dir / "slides"
    assert _pngs(images) == ["slide_001.png", "slide_002.png", "slide_003.png"]
    (images / "мои_заметки.txt").write_text("не трогать", encoding="utf-8")
    (images / "схема.png").write_bytes(b"png")

    holder["count"] = 2
    # другая область демонстрации -> ключ slides сменился -> пересчёт вниз
    assert (
        cli.main([str(source), "--out-dir", str(out_dir), "--slide-region", "0,0,32,18"])
        == 0
    )
    assert _pngs(images) == ["slide_001.png", "slide_002.png", "схема.png"]
    assert (images / "мои_заметки.txt").read_text(encoding="utf-8") == "не трогать"


@pytest.mark.parametrize(
    "selection",
    [["--from-stage", "merge"], ["--from-stage", "frames"], ["--only-stage", "vad"]],
)
def test_no_cache_with_selection_needing_cache_is_rejected(
    tmp_path: Path, capsys, selection: list[str]
) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    with pytest.raises(SystemExit) as exc:
        cli.main([str(source), "--out-dir", str(tmp_path / "out"), *selection, "--no-cache"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--no-cache несовместим" in err
    assert "только из кэша" in err


def test_no_cache_with_self_contained_selection_is_allowed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Корневая стадия или полный прогон с --no-cache — осмысленно; лог про
    пропущенные стадии называет настоящую причину, а не «нет в кэше»."""
    parser = cli.build_parser()
    for selection in (["--from-stage", "audio"], ["--only-stage", "frames"], []):
        args = parser.parse_args(["a.mp4", *selection, "--no-cache"])
        assert cli._stages_needing_cache(cli.selected_stages(args)) == []

    monkeypatch.setattr(cli, "_stage_callable", lambda stage: lambda ctx: {})
    monkeypatch.setattr(cli, "_STAGE_MODEL_HOLDERS", {})
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    code = cli.main(
        [
            str(source), "--out-dir", str(tmp_path / "out"),
            "--only-stage", "frames", "--no-cache", "--log-level", "INFO",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "стадия audio: пропущена (чтение кэша выключено флагом --no-cache)" in err
    assert "нет в кэше" not in err


# --------------------------------------------------------------------------
# Аудио-режим: файл без видеодорожки (spec media-ingest)
# --------------------------------------------------------------------------


@requires_ffmpeg
def test_audio_only_file_produces_transcript_without_slides(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Файл без видео проходит весь пайплайн: код 0, transcript.md без секций
    слайдов, OCR не трогается вовсе, предупреждение о неполноте — один раз."""
    from lecture_transcript import slide_ocr
    from lecture_transcript import speech_transcription as speech
    from lecture_transcript.contracts import Availability, Transcription, Word
    from lecture_transcript.slide_ocr import registry as ocr_registry

    source = tmp_path / "lecture_audio.m4a"
    made = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=4",
            "-c:a", "aac", "-vn", str(source),
        ],
        capture_output=True, text=True, check=False,
    )
    if made.returncode != 0:
        pytest.skip(f"ffmpeg не собрал аудио-фикстуру: {made.stderr[-200:]}")

    class FakeAsr:
        name = "fake-asr"
        provides_punctuation = True
        supports_glossary = True

        def check_availability(self):
            return Availability(True)

        def transcribe(self, audio, intervals, glossary=None):
            return Transcription(
                words=(Word("Здравствуйте.", 0.5, 1.0, 0.9), Word("Начнём.", 2.0, 2.5, 0.9)),
                backend=self.name,
                has_punctuation=True,
                used_glossary=glossary is not None,
            )

        def unload(self):
            pass

    speech.register("fake-asr", FakeAsr)
    try:
        # машина без PaddleOCR: бэкенд OCR недоступен, а любое обращение
        # стадии к нему — ошибка теста
        def must_not_touch(*args, **kwargs):
            raise AssertionError("в аудио-режиме OCR не должен вызываться")

        monkeypatch.setattr(
            ocr_registry, "check", lambda name: Availability(False, "нет paddle")
        )
        monkeypatch.setattr(ocr_registry, "ensure_available", must_not_touch)
        monkeypatch.setattr(slide_ocr, "recognize_slides", must_not_touch)
        monkeypatch.setattr(slide_ocr, "build_glossary", must_not_touch)

        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "speech_transcription:\n  vad_backend: energy\n", encoding="utf-8"
        )
        out_dir = tmp_path / "out"
        code = cli.main(
            [
                str(source), "--out-dir", str(out_dir),
                "--config", str(config_file), "--asr-backend", "fake-asr",
                "--hwaccel", "none", "--log-level", "INFO",
            ]
        )
    finally:
        speech.unregister("fake-asr")

    log = capsys.readouterr().err
    assert code == 0, log[-2000:]
    transcript = out_dir / "transcript.md"
    assert transcript.is_file()
    text = transcript.read_text(encoding="utf-8")
    assert "## Речь вне слайдов" in text
    assert "## Слайд " not in text
    assert "Здравствуйте." in text
    assert not list((out_dir / "slides").glob("slide_*.png")) if (out_dir / "slides").exists() else True
    assert log.count("аудио-режим: во входном файле нет видеодорожки") == 1


def test_broken_registry_does_not_break_help(monkeypatch) -> None:
    """Битая установка реестра (OSError от CUDA-библиотек) не ломает --help."""
    import importlib

    def explode(name: str, *a, **kw):
        if "registry" in name:
            raise OSError("libcudart.so: cannot open shared object file")
        return importlib.__import__(name)

    monkeypatch.setattr(cli.importlib, "import_module", explode)
    assert cli.known_backends("asr") == []
    assert cli.available_backends("ocr") == []
    assert "--asr-backend" in cli.build_parser().format_help()


def test_cli_overrides_config() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "lecture.mp4",
            "--frames-fps",
            "2.5",
            "--hwaccel",
            "cuda",
            "--asr-backend",
            "whisper-large-v3",
            "--ocr-backend",
            "qwen2.5-vl",
            "--cache-identity",
            "content",
        ]
    )
    config = cli.apply_overrides(load_config(), args)
    assert config.media_ingest.frames_fps == 2.5
    assert config.media_ingest.hwaccel == "cuda"
    assert config.speech_transcription.asr_backend == "whisper-large-v3"
    assert config.slide_ocr.backend == "qwen2.5-vl"
    assert config.cache.identity == "content"


def test_stage_selection() -> None:
    parser = cli.build_parser()
    assert cli.selected_stages(parser.parse_args(["a.mp4"])) == STAGES
    assert cli.selected_stages(parser.parse_args(["a.mp4", "--from-stage", "asr"])) == (
        "asr",
        "punctuation",
        "merge",
    )
    assert cli.selected_stages(parser.parse_args(["a.mp4"]))[0] == "audio"
    assert cli.selected_stages(
        parser.parse_args(["a.mp4", "--only-stage", "ocr"])
    ) == ("ocr",)


def test_unknown_stage_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["a.mp4", "--from-stage", "нет-такой"])
    assert exc.value.code == 2


@pytest.mark.parametrize("kind", ["asr", "ocr"])
def test_unknown_backend_rejected_before_the_run(
    tmp_path: Path, capsys, monkeypatch, kind: str
) -> None:
    """Опечатка в имени бэкенда отвергается ДО стадий (задачи 4.1, 5.1).

    Проверка идёт по зарегистрированным именам: на машине без весов список
    «доступных» пуст, и валидация по нему пропускала любое имя.
    """
    if not cli.known_backends(kind):
        pytest.skip(f"реестр {kind}-бэкендов недоступен в этом окружении")

    def explode(stage: str):
        raise AssertionError(f"стадия {stage} запущена, хотя бэкенд неизвестен")

    monkeypatch.setattr(cli, "_stage_callable", explode)
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    code = cli.main(
        [
            str(source), "--out-dir", str(tmp_path / "out"),
            f"--{kind}-backend", "НЕТ-ТАКОГО", "--log-level", "WARNING",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "неизвестный" in err and "зарегистрированы" in err


def test_known_backend_name_passes_validation(tmp_path: Path) -> None:
    """Зарегистрированное имя (и псевдоним ASR) валидацию проходит."""
    if not cli.known_backends("asr"):
        pytest.skip("реестр ASR-бэкендов недоступен в этом окружении")
    cli._validate_backend("asr", cli.known_backends("asr")[0])
    cli._validate_backend("ocr", cli.known_backends("ocr")[0])


def test_missing_input_reports_error(tmp_path: Path, capsys) -> None:
    code = cli.main([str(tmp_path / "нет.mp4"), "--out-dir", str(tmp_path / "out")])
    assert code == 2
    assert "не найден" in capsys.readouterr().err


def test_run_with_unimplemented_stages_returns_nonzero(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Каркас проходит цепочку целиком, но неполный прогон — не успех.

    Стадии подменены на «не реализована» намеренно: тест про каркас
    оркестрации, а не про чужие пакеты, которые пишутся параллельно.
    """
    monkeypatch.setattr(cli, "_stage_callable", lambda stage: None)
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    out_dir = tmp_path / "out"
    code = cli.main(
        [str(source), "--out-dir", str(out_dir), "--log-level", "DEBUG"]
    )
    assert code == 1, "прогон без результата обязан вернуть ненулевой код"
    err = capsys.readouterr().err
    assert "не реализована" in err
    assert "не выполнены стадии" in err
    assert out_dir.is_dir()


def test_run_without_any_result_returns_nonzero(tmp_path: Path, capsys) -> None:
    """`--from-stage merge` на пустом кэше: transcript.md нет — код не 0."""
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    out_dir = tmp_path / "out"
    code = cli.main(
        [
            str(source), "--out-dir", str(out_dir),
            "--from-stage", "merge", "--log-level", "INFO",
        ]
    )
    assert code == 1
    assert not (out_dir / "transcript.md").exists()
    assert "не выполнены стадии" in capsys.readouterr().err


def test_stage_without_prerequisites_is_reported(tmp_path: Path, capsys) -> None:
    """Стадия, чьи предшественницы не посчитаны, не запускается вслепую."""
    calls: list[str] = []

    def fake_merge(ctx: cli.StageContext) -> dict:
        calls.append(ctx.stage)
        return {}

    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    import unittest.mock as mock

    with mock.patch.object(
        cli, "_stage_callable", lambda stage: fake_merge if stage == "merge" else None
    ):
        code = cli.main(
            [
                str(source), "--out-dir", str(tmp_path / "out"),
                "--from-stage", "merge", "--log-level", "INFO",
            ]
        )
    assert code == 1
    assert calls == [], "стадия запущена без результатов предшественниц"
    assert "нет результата предшественниц" in capsys.readouterr().err


def test_unreadable_media_stops_the_run(tmp_path: Path, capsys) -> None:
    """Не-медиа на входе отвергается реализованной стадией frames, код 2."""
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    code = cli.main(
        [str(source), "--out-dir", str(tmp_path / "out"), "--log-level", "WARNING"]
    )
    assert code == 2
    assert capsys.readouterr().err.strip()


def test_from_and_only_stage_are_exclusive(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(source), "--from-stage", "asr", "--only-stage", "merge"])
    assert exc.value.code == 2


def test_models_are_unloaded_after_every_model_stage(
    tmp_path: Path, monkeypatch
) -> None:
    """D7: модель каждой модельной стадии выгружается ДО старта следующей
    стадии — не только ocr/asr и не только в конце прогона."""
    events: list[str] = []

    class FakeModel:
        def __init__(self, stage: str) -> None:
            self.stage = stage

        def unload(self) -> None:
            events.append(f"unload:{self.stage}")

    model_stages = ("ocr", "asr", "punctuation")
    assert set(model_stages) <= set(cli._STAGE_MODEL_HOLDERS), (
        f"держатели моделей объявлены только для {sorted(cli._STAGE_MODEL_HOLDERS)}"
    )
    monkeypatch.setattr(
        cli,
        "_STAGE_MODEL_HOLDERS",
        {stage: (lambda cfg, s=stage: FakeModel(s)) for stage in model_stages},
    )

    def stage_stub(stage: str):
        def run(ctx: cli.StageContext) -> dict:
            events.append(f"run:{stage}")
            if stage == "merge":
                # настоящий результат, иначе страховка merge даст код 1
                (ctx.build_dir / "transcript.md").write_text("# т\n", encoding="utf-8")
                return {"transcript": "transcript.md"}
            return {"ok": stage}

        return run

    monkeypatch.setattr(cli, "_stage_callable", stage_stub)

    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    out_dir = tmp_path / "out"
    assert cli.main([str(source), "--out-dir", str(out_dir)]) == 0
    assert (out_dir / "transcript.md").is_file()

    unloads = [e for e in events if e.startswith("unload:")]
    assert sorted(unloads) == sorted(f"unload:{s}" for s in model_stages), events
    order = {event: i for i, event in enumerate(events)}
    for stage in model_stages:
        following = cli.STAGES[cli.STAGES.index(stage) + 1]
        assert order[f"unload:{stage}"] < order[f"run:{following}"], (
            f"модель {stage} выгружена после старта {following}: {events}"
        )


# --------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------


def test_default_config_loads_from_package() -> None:
    config = load_config()
    assert config.media_ingest.frames_fps == 1.0
    assert config.media_ingest.audio_sample_rate == 16000
    assert config.speech_transcription.punctuation == "auto"
    assert config.transcript_assembly.output_filename == "transcript.md"


def test_dataclass_defaults_match_yaml() -> None:
    """Дефолты dataclass и default_config.yaml не должны расходиться.

    YAML побеждает при load_config, поэтому расхождение молча даёт неверные
    значения любому коду, который собирает конфиг прямо из dataclass.
    """
    from_code = PipelineConfig().to_dict()
    from_yaml = load_config().to_dict()
    assert set(from_code) == set(from_yaml), "разошёлся состав секций"
    drift = {
        f"{section}.{key}": (value, from_yaml[section].get(key))
        for section, values in from_code.items()
        for key, value in values.items()
        if key not in from_yaml[section] or from_yaml[section][key] != value
    }
    missing = {
        f"{section}.{key}"
        for section, values in from_yaml.items()
        for key in values
        if key not in from_code[section]
    }
    assert not drift, f"dataclass != yaml: {drift}"
    assert not missing, f"поля есть в yaml, но нет в dataclass: {missing}"


def test_placeholders_are_declared_and_present() -> None:
    """Значения из 3.7 и 7.6 помечены как подбираемые на эталонной записи."""
    config = load_config()
    data = config.to_dict()
    for dotted in PLACEHOLDER_DEFAULTS:
        section, key = dotted.split(".")
        assert key in data[section], dotted
    assert "эталонной записи" in " ".join(PLACEHOLDER_DEFAULTS.values())


def test_user_config_overlay(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "slide_extraction:\n  hamming_threshold: 12\n"
        "speech_transcription:\n  asr_backend: whisper-large-v3\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.slide_extraction.hamming_threshold == 12
    assert config.speech_transcription.asr_backend == "whisper-large-v3"
    # остальное осталось дефолтным
    assert config.media_ingest.frames_fps == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "нет_такой_секции:\n  a: 1\n",
        "slide_extraction:\n  нет_такого_ключа: 1\n",
        "media_ingest:\n  frames_fps: не-число\n",
        "media_ingest: 42\n",
    ],
)
def test_bad_config_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


# Поле конфига -> где живёт дефолт той же величины в пакете стадии.
# Третий источник правды: YAML не должен расходиться с кодом, который эти
# значения потребляет. Путь после ":" разворачивается по атрибутам,
# "Name()" означает вызов (дефолтный экземпляр dataclass).
STAGE_DEFAULT_CONTRACT: dict[str, tuple[str, str]] = {
    "media_ingest.audio_sample_rate": (
        "lecture_transcript.media_ingest", "TARGET_SAMPLE_RATE"),
    "slide_extraction.variance_sample_frames": (
        "lecture_transcript.slide_extraction.layout", "DEFAULT_SAMPLE_FRAMES"),
    "slide_extraction.variance_window_s": (
        "lecture_transcript.slide_extraction.layout", "DEFAULT_WINDOW_S"),
    "slide_extraction.min_region_area_ratio": (
        "lecture_transcript.slide_extraction.region", "DEFAULT_MIN_AREA_RATIO"),
    "slide_extraction.min_layout_interval_s": (
        "lecture_transcript.slide_extraction.layout", "DEFAULT_MIN_LAYOUT_INTERVAL_S"),
    "slide_extraction.min_absence_s": (
        "lecture_transcript.slide_extraction.layout", "DEFAULT_MIN_ABSENCE_S"),
    "slide_extraction.max_spread": (
        "lecture_transcript.slide_extraction.variance", "DEFAULT_MAX_SPREAD"),
    "slide_extraction.min_brightness": (
        "lecture_transcript.slide_extraction.variance", "DEFAULT_MIN_BRIGHTNESS"),
    "slide_extraction.hash_size": (
        "lecture_transcript.slide_extraction.dedup", "DEFAULT_HASH_SIZE"),
    "slide_extraction.hamming_threshold": (
        "lecture_transcript.slide_extraction.dedup", "DEFAULT_HAMMING_THRESHOLD"),
    "slide_ocr.backend": ("lecture_transcript.slide_ocr", "DEFAULT_BACKEND"),
    "slide_ocr.confidence_threshold": (
        "lecture_transcript.slide_ocr", "DEFAULT_ASSEMBLE.low_confidence_threshold"),
    "slide_ocr.unreliable_fragment_ratio": (
        "lecture_transcript.slide_ocr", "DEFAULT_ASSEMBLE.unreliable_ratio"),
    "slide_ocr.glossary_min_term_length": (
        "lecture_transcript.slide_ocr", "DEFAULT_GLOSSARY.min_term_length"),
    "slide_ocr.glossary_max_terms": (
        "lecture_transcript.slide_ocr", "DEFAULT_GLOSSARY.max_terms"),
    "speech_transcription.asr_backend": (
        "lecture_transcript.speech_transcription", "DEFAULT_BACKEND_NAME"),
    "speech_transcription.vad_threshold": (
        "lecture_transcript.speech_transcription", "VadConfig().silero_threshold"),
    "speech_transcription.min_silence_s": (
        "lecture_transcript.speech_transcription", "VadConfig().min_silence_s"),
    "speech_transcription.min_speech_s": (
        "lecture_transcript.speech_transcription", "VadConfig().min_speech_s"),
    "speech_transcription.speech_pad_s": (
        "lecture_transcript.speech_transcription", "VadConfig().speech_pad_s"),
    "transcript_assembly.section_boundary_tolerance_s": (
        "lecture_transcript.transcript_assembly", "BoundaryPolicy().tolerance_s"),
    "transcript_assembly.short_offslide_section_s": (
        "lecture_transcript.transcript_assembly",
        "ConsolidationPolicy().short_offslide_section_s"),
    "transcript_assembly.paragraph_pause_s": (
        "lecture_transcript.transcript_assembly", "ParagraphPolicy().pause_s"),
    "transcript_assembly.timecode_every_s": (
        "lecture_transcript.transcript_assembly", "RenderPolicy().timecode_every_s"),
    "transcript_assembly.filler_cleanup": (
        "lecture_transcript.transcript_assembly", "CleanupPolicy().enabled"),
}


def _resolve_attr(module_name: str, path: str):
    """Развернуть 'DEFAULT_ASSEMBLE.unreliable_ratio' или 'VadConfig().field'."""
    import importlib

    obj = importlib.import_module(module_name)
    for part in path.split("."):
        call = part.endswith("()")
        name = part[:-2] if call else part
        if not hasattr(obj, name):
            raise AttributeError(
                f"{module_name}:{path} — нет атрибута {name!r} "
                "(переименован или удалён в пакете стадии)"
            )
        obj = getattr(obj, name)
        if call:
            obj = obj()
    return obj


@pytest.mark.parametrize("config_path", sorted(STAGE_DEFAULT_CONTRACT))
def test_yaml_matches_stage_defaults(config_path: str) -> None:
    """default_config.yaml не должен расходиться с дефолтами кода стадии.

    Третий источник правды после dataclass и YAML: значения из YAML станут
    эффективными после стыковки и молча перекроют подобранные автором стадии.
    """
    module_name, attr_path = STAGE_DEFAULT_CONTRACT[config_path]
    section, field = config_path.split(".")
    ours = load_config().to_dict()[section][field]
    try:
        theirs = _resolve_attr(module_name, attr_path)
    except ImportError:
        pytest.skip(f"пакет {module_name} недоступен в этом окружении")
    assert ours == theirs, (
        f"дефолт разошёлся:\n"
        f"  default_config.yaml  {section}.{field} = {ours!r}\n"
        f"  {module_name}:{attr_path} = {theirs!r}\n"
        "выровняй YAML по коду стадии (значения подбираются там) "
        "или обнови таблицу STAGE_DEFAULT_CONTRACT, если поле переименовали"
    )


def test_stage_defaults_contract_covers_shared_fields() -> None:
    """Каждая запись таблицы указывает на существующее поле конфига."""
    config = load_config().to_dict()
    for config_path in STAGE_DEFAULT_CONTRACT:
        section, field = config_path.split(".")
        assert section in config and field in config[section], config_path


def test_stage_params_cover_all_stages() -> None:
    config = load_config()
    for stage in STAGES:
        params = config.stage_params(stage)
        assert isinstance(params, dict) and params, stage
    with pytest.raises(ConfigError):
        config.stage_params("нет-такой")


# --------------------------------------------------------------------------
# Сквозная стыковка со стадией frames (media_ingest)
# --------------------------------------------------------------------------



@pytest.fixture(scope="module")
def tiny_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Двухсекундный клип со звуком — минимальный реальный вход для frames."""
    path = tmp_path_factory.mktemp("core_cli_input") / "lecture.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg не собрал фикстуру: {result.stderr[-200:]}")
    return path


@requires_ffmpeg
def test_frames_stage_runs_and_second_run_hits_cache(
    tiny_mp4: Path, tmp_path: Path, capsys
) -> None:
    """Стадия отработала -> повторный прогон взял её из кэша, не пересчитывая."""
    out_dir = tmp_path / "out"
    args = [
        str(tiny_mp4), "--out-dir", str(out_dir),
        "--only-stage", "frames", "--hwaccel", "none", "--log-level", "INFO",
    ]
    assert cli.main(args) == 0
    first = capsys.readouterr().err
    assert "стадия frames: готово за" in first

    cells = sorted((out_dir / ".cache" / "frames").iterdir())
    assert len(cells) == 1, cells
    before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(cells[0].rglob("*"))
    }
    # WAV теперь в своей ячейке `audio`: стадии независимы
    assert not any(name.startswith("audio") for name in before)
    assert "frames" in before

    assert cli.main(args) == 0
    second = capsys.readouterr().err
    assert "попадание в кэш" in second
    assert "стадия frames: готово за" not in second
    after = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(cells[0].rglob("*"))
    }
    assert after == before, "артефакты стадии были пересчитаны"


@requires_ffmpeg
def test_next_stage_gets_cell_dir_for_relative_payload(
    tiny_mp4: Path, tmp_path: Path, monkeypatch
) -> None:
    """Следующая стадия разворачивает пути frames через ctx.stage_dirs."""
    from lecture_transcript.media_ingest import audio_artifact, frames_artifact

    seen: dict[str, object] = {}
    original = cli._stage_callable

    def fake_slides(ctx: cli.StageContext) -> dict:
        audio = audio_artifact(ctx.results["audio"], ctx.stage_dirs["audio"])
        frames = frames_artifact(ctx.results["frames"], ctx.stage_dirs["frames"])
        seen["audio_exists"] = audio.path.is_file()
        seen["frames_count"] = len(frames.frames)
        seen["base"] = ctx.stage_dirs["frames"]
        seen["audio_base"] = ctx.stage_dirs["audio"]
        return {"slides": 0}

    def dispatch(stage: str):
        if stage in ("audio", "frames"):
            return original(stage)
        if stage == "slides":
            return fake_slides
        return None

    monkeypatch.setattr(cli, "_stage_callable", dispatch)
    # ограничиваем граф двумя ветвями, чтобы тест не зависел от стадий,
    # которые пишут параллельно другие агенты
    monkeypatch.setattr(cli, "STAGES", ("audio", "frames", "slides"))

    out_dir = tmp_path / "out"
    args = [
        str(tiny_mp4), "--out-dir", str(out_dir),
        "--hwaccel", "none", "--log-level", "WARNING",
    ]
    assert cli.main(args) == 0
    assert seen["audio_exists"] is True
    assert seen["frames_count"] > 0
    assert Path(seen["audio_base"]).parent.name == "audio"
    # свежесчитанная стадия отдаёт каталог сборки, он же становится ячейкой
    assert Path(seen["base"]).is_dir()

    # повторный прогон: frames взят из кэша, база — каталог ячейки.
    # Ячейку slides сносим, иначе она тоже попадёт в кэш и стадия не вызовется.
    shutil.rmtree(out_dir / ".cache" / "slides")
    seen.clear()
    assert cli.main(args) == 0
    assert seen["audio_exists"] is True
    assert Path(seen["base"]).parent.name == "frames"
    assert Path(seen["base"]).parent.parent == out_dir / ".cache"
