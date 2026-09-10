"""Тесты слоя кэша промежуточных артефактов (задача 1.4, design D8)."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from lecture_transcript.cache import Cache, CacheError, file_identity
from lecture_transcript.config import STAGE_DEPENDENCIES, STAGES, load_config

# Упрощённая цепочка «как в пайплайне»: параметры у каждой стадии свои.
CHAIN_PARAMS: dict[str, dict] = {
    "frames": {"frames_fps": 1.0, "hwaccel": "none"},
    "slides": {"hamming_threshold": 8},
    "ocr": {"backend": "paddle+pix2tex", "confidence_threshold": 0.6},
    "glossary": {"glossary_max_terms": 300},
    "vad": {"vad_backend": "silero"},
    "asr": {"asr_backend": "gigaam-v2"},
    "punctuation": {"punctuation": "auto"},
    "merge": {"section_boundary_tolerance_s": 2.0},
}


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    path = tmp_path / "lecture.mp4"
    path.write_bytes(b"fake-mp4-payload" * 1024)
    return path


def build_chain(tmp_path: Path, source: Path, params: dict[str, dict], **kwargs):
    cache = Cache(tmp_path / "cache", source, **kwargs)
    return cache, cache.chain((stage, params[stage]) for stage in params)


def test_chain_keys_are_stable(tmp_path: Path, source: Path) -> None:
    _, first = build_chain(tmp_path, source, CHAIN_PARAMS)
    _, second = build_chain(tmp_path, source, CHAIN_PARAMS)
    assert [cell.key for cell in first.values()] == [
        cell.key for cell in second.values()
    ]


@pytest.mark.parametrize("changed", list(CHAIN_PARAMS))
def test_param_change_invalidates_own_stage_and_later_only(
    tmp_path: Path, source: Path, changed: str
) -> None:
    """Ключевая проверка 1.4: инвалидация ровно вниз по цепочке."""
    _, before = build_chain(tmp_path, source, CHAIN_PARAMS)

    modified = {stage: dict(params) for stage, params in CHAIN_PARAMS.items()}
    modified[changed]["__probe__"] = "изменённое значение параметра"
    _, after = build_chain(tmp_path, source, modified)

    order = list(CHAIN_PARAMS)
    index = order.index(changed)
    for stage in order[:index]:
        assert before[stage].key == after[stage].key, (
            f"стадия {stage} до изменённой не должна инвалидироваться"
        )
    for stage in order[index:]:
        assert before[stage].key != after[stage].key, (
            f"стадия {stage} обязана инвалидироваться"
        )


def test_saved_stages_before_change_stay_hits(tmp_path: Path, source: Path) -> None:
    """То же самое, но на реальных артефактах на диске, а не только на ключах."""
    _, cells = build_chain(tmp_path, source, CHAIN_PARAMS)
    for stage, cell in cells.items():
        build = cell.new_build_dir()
        (build / "artifact.txt").write_text(stage, encoding="utf-8")
        cell.save({"stage": stage}, build_dir=build)
        assert cell.hit()

    modified = {stage: dict(params) for stage, params in CHAIN_PARAMS.items()}
    modified["ocr"]["confidence_threshold"] = 0.9
    _, after = build_chain(tmp_path, source, modified)

    order = list(CHAIN_PARAMS)
    index = order.index("ocr")
    for stage in order[:index]:
        assert after[stage].hit(), f"{stage} должна остаться в кэше"
        assert after[stage].load() == {"stage": stage}
    for stage in order[index:]:
        assert not after[stage].hit(), f"{stage} обязана пересчитаться"


def test_hit_and_payload_roundtrip(tmp_path: Path, source: Path) -> None:
    cache = Cache(tmp_path / "cache", source)
    cell = cache.stage("frames", CHAIN_PARAMS["frames"])
    assert not cell.hit()
    with pytest.raises(CacheError):
        cell.load()

    build = cell.new_build_dir()
    (build / "0001.png").write_bytes(b"png")
    cell.save({"frames": ["0001.png"], "fps": 1.0}, build_dir=build)

    again = Cache(tmp_path / "cache", source).stage("frames", CHAIN_PARAMS["frames"])
    assert again.hit()
    assert again.load() == {"frames": ["0001.png"], "fps": 1.0}
    assert (again.path / "0001.png").read_bytes() == b"png"
    meta = json.loads((again.path / "meta.json").read_text(encoding="utf-8"))
    assert meta["stage"] == "frames" and meta["key"] == again.key


def test_mtime_change_invalidates_whole_chain(tmp_path: Path, source: Path) -> None:
    _, before = build_chain(tmp_path, source, CHAIN_PARAMS)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    _, after = build_chain(tmp_path, source, CHAIN_PARAMS)
    for stage in CHAIN_PARAMS:
        assert before[stage].key != after[stage].key


def test_size_change_invalidates_whole_chain(tmp_path: Path, source: Path) -> None:
    _, before = build_chain(tmp_path, source, CHAIN_PARAMS)
    with source.open("ab") as handle:
        handle.write(b"tail")
    _, after = build_chain(tmp_path, source, CHAIN_PARAMS)
    for stage in CHAIN_PARAMS:
        assert before[stage].key != after[stage].key


def test_content_identity_survives_touch_but_not_edit(
    tmp_path: Path, source: Path
) -> None:
    cache = Cache(tmp_path / "cache", source, identity="content")
    cell = cache.stage("frames", CHAIN_PARAMS["frames"])
    cell.save({"ok": True})

    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    touched = Cache(tmp_path / "cache", source, identity="content").stage(
        "frames", CHAIN_PARAMS["frames"]
    )
    assert touched.hit(), "содержимое не менялось — должен быть хит"

    data = source.read_bytes()
    source.write_bytes(data[:-1] + b"X")
    edited = Cache(tmp_path / "cache", source, identity="content").stage(
        "frames", CHAIN_PARAMS["frames"]
    )
    assert not edited.hit(), "содержимое изменилось — должен быть промах"


def test_sample_identity_reads_only_chunks(tmp_path: Path) -> None:
    """Идентичность по выборке не тянет весь файл, но ловит правку в начале."""
    big = tmp_path / "big.mp4"
    big.write_bytes(b"\x00" * (5 << 20))
    identity = file_identity(big, "sample")
    assert identity["mode"] == "sample" and identity["size"] == 5 << 20

    with big.open("r+b") as handle:
        handle.write(b"changed")
    assert file_identity(big, "sample") != identity


def test_interrupted_build_is_not_a_hit(tmp_path: Path, source: Path) -> None:
    """Прерванная запись не оставляет ложного попадания (атомарность)."""
    cache = Cache(tmp_path / "cache", source)
    cell = cache.stage("frames", CHAIN_PARAMS["frames"])
    build = cell.new_build_dir()
    (build / "partial.png").write_bytes(b"half")
    # save() не вызван — имитация прерывания прогона.
    assert not cell.hit()
    assert not cell.path.exists()

    cell.save({"ok": True}, build_dir=cell.new_build_dir())
    assert cell.hit()


def test_broken_meta_is_a_miss(tmp_path: Path, source: Path) -> None:
    cache = Cache(tmp_path / "cache", source)
    cell = cache.stage("frames", CHAIN_PARAMS["frames"])
    cell.save({"ok": True})
    (cell.path / "meta.json").write_text("{битый json", encoding="utf-8")
    assert not cell.hit()


def test_save_replaces_previous_artifact(tmp_path: Path, source: Path) -> None:
    cache = Cache(tmp_path / "cache", source)
    cell = cache.stage("slides", CHAIN_PARAMS["slides"])
    first = cell.new_build_dir()
    (first / "old.txt").write_text("старое", encoding="utf-8")
    cell.save({"v": 1}, build_dir=first)

    second = cell.new_build_dir()
    (second / "new.txt").write_text("новое", encoding="utf-8")
    cell.save({"v": 2}, build_dir=second)

    assert cell.load() == {"v": 2}
    assert (cell.path / "new.txt").is_file()
    assert not (cell.path / "old.txt").exists()


def test_disabled_cache_never_hits(tmp_path: Path, source: Path) -> None:
    cache = Cache(tmp_path / "cache", source, enabled=False)
    cell = cache.stage("frames", CHAIN_PARAMS["frames"])
    cell.save({"ok": True})
    assert not cell.hit()
    # но артефакт материализован — стадиям есть куда писать
    assert (cell.path / "meta.json").is_file()


def test_missing_source_and_unknown_identity_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(CacheError):
        Cache(tmp_path / "cache", tmp_path / "нет.mp4")
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    with pytest.raises(CacheError):
        Cache(tmp_path / "cache", real, identity="магия")


def test_duplicate_stage_in_chain_rejected(tmp_path: Path, source: Path) -> None:
    cache = Cache(tmp_path / "cache", source)
    with pytest.raises(CacheError):
        cache.chain([("frames", {}), ("frames", {})])


# --------------------------------------------------------------------------
# Реальный граф стадий (design D2 + D8)
# --------------------------------------------------------------------------


def real_cells(tmp_path: Path, source: Path, config) -> dict:
    cache = Cache(tmp_path / "cache", source)
    return cache.chain(
        ((stage, config.stage_params(stage)) for stage in STAGES),
        dependencies=STAGE_DEPENDENCIES,
    )


def descendants(stage: str) -> set[str]:
    """Стадия и всё, что от неё зависит, — по графу зависимостей."""
    affected = {stage}
    changed = True
    while changed:
        changed = False
        for name, parents in STAGE_DEPENDENCIES.items():
            if name not in affected and affected.intersection(parents):
                affected.add(name)
                changed = True
    return affected


def other_value(value):
    """Заведомо другое значение того же типа."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "-другое"
    if isinstance(value, tuple):
        return value + ("другое",)
    if value is None:
        return 7
    raise AssertionError(f"не знаю, как изменить {value!r}")


def test_pipeline_config_drives_the_real_graph(tmp_path: Path, source: Path) -> None:
    """Граф из настоящей конфигурации совпадает по составу со STAGES."""
    cells = real_cells(tmp_path, source, load_config())
    assert tuple(cells) == STAGES
    assert len({cell.key for cell in cells.values()}) == len(STAGES)
    assert cells["audio"].parent_keys == () and cells["frames"].parent_keys == ()


# Поле конфига -> стадия, которую его правка ОБЯЗАНА инвалидировать.
# Это связь «параметр -> стадия», без неё проверка 1.4 проверяет только
# механику кэша, но не то, что параметр вообще доходит до своей стадии.
FIELD_STAGE: dict[tuple[str, str], str] = {
    ("media_ingest", "audio_track"): "audio",
    ("media_ingest", "audio_sample_rate"): "audio",
    ("media_ingest", "frames_fps"): "frames",
    ("media_ingest", "hwaccel"): "frames",
    ("slide_extraction", "variance_sample_frames"): "slides",
    ("slide_extraction", "variance_window_s"): "slides",
    ("slide_extraction", "min_region_area_ratio"): "slides",
    ("slide_extraction", "min_layout_interval_s"): "slides",
    ("slide_extraction", "min_absence_s"): "slides",
    ("slide_extraction", "max_spread"): "slides",
    ("slide_extraction", "min_brightness"): "slides",
    ("slide_extraction", "hash_size"): "slides",
    ("slide_extraction", "hamming_threshold"): "slides",
    ("slide_extraction", "region"): "slides",
    ("slide_ocr", "backend"): "ocr",
    ("slide_ocr", "confidence_threshold"): "ocr",
    ("slide_ocr", "unreliable_fragment_ratio"): "ocr",
    ("slide_ocr", "glossary_min_term_length"): "glossary",
    ("slide_ocr", "glossary_max_terms"): "glossary",
    ("slide_ocr", "glossary_stopwords"): "glossary",
    ("speech_transcription", "vad_backend"): "vad",
    ("speech_transcription", "vad_threshold"): "vad",
    ("speech_transcription", "min_silence_s"): "vad",
    ("speech_transcription", "min_speech_s"): "vad",
    ("speech_transcription", "speech_pad_s"): "vad",
    ("speech_transcription", "asr_backend"): "asr",
    ("speech_transcription", "use_glossary"): "asr",
    ("speech_transcription", "punctuation"): "punctuation",
    ("transcript_assembly", "section_boundary_tolerance_s"): "merge",
    ("transcript_assembly", "short_offslide_section_s"): "merge",
    ("transcript_assembly", "paragraph_pause_s"): "merge",
    ("transcript_assembly", "timecode_every_s"): "merge",
    ("transcript_assembly", "filler_cleanup"): "merge",
    ("transcript_assembly", "output_filename"): "merge",
}


# Поля, у которых «другое значение» должно быть осмысленным, а не 7.
FIELD_VALUE_OVERRIDES: dict[tuple[str, str], object] = {
    ("slide_extraction", "region"): (10, 20, 300, 400),
}


def test_field_stage_table_covers_every_config_field() -> None:
    """Новое поле конфига обязано получить свою стадию в таблице выше."""
    config = load_config()
    declared = {
        (section, field)
        for section, values in config.to_dict().items()
        if section != "cache"  # секция кэша не параметр стадии
        for field in values
    }
    assert declared == set(FIELD_STAGE), (
        f"не в таблице: {sorted(declared - set(FIELD_STAGE))}; "
        f"лишние: {sorted(set(FIELD_STAGE) - declared)}"
    )


@pytest.mark.parametrize(("section", "field"), sorted(FIELD_STAGE))
def test_config_field_invalidates_its_stage_and_descendants(
    tmp_path: Path, source: Path, section: str, field: str
) -> None:
    """Правка поля инвалидирует свою стадию и её потомков — и никого больше."""
    config = load_config()
    before = real_cells(tmp_path, source, config)

    current = getattr(getattr(config, section), field)
    changed_value = FIELD_VALUE_OVERRIDES.get((section, field), None)
    if changed_value is None:
        changed_value = other_value(current)
    modified = dataclasses.replace(
        config,
        **{
            section: dataclasses.replace(
                getattr(config, section), **{field: changed_value}
            )
        },
    )
    after = real_cells(tmp_path, source, modified)

    changed = {stage for stage in STAGES if before[stage].key != after[stage].key}
    assert changed == descendants(FIELD_STAGE[(section, field)]), (
        f"{section}.{field}: инвалидировались {sorted(changed)}"
    )


def test_cache_section_is_not_a_stage_parameter(tmp_path: Path, source: Path) -> None:
    """Каталог кэша и режим идентичности не влияют на ключи стадий."""
    config = load_config()
    before = real_cells(tmp_path, source, config)
    modified = dataclasses.replace(
        config, cache=dataclasses.replace(config.cache, directory="иначе")
    )
    after = real_cells(tmp_path, source, modified)
    assert all(before[s].key == after[s].key for s in STAGES)


def test_audio_and_frames_are_independent(tmp_path: Path, source: Path) -> None:
    """Смена аудиодорожки не трогает кадры и ветку слайдов (и наоборот)."""
    config = load_config()
    before = real_cells(tmp_path, source, config)

    other_audio = dataclasses.replace(
        config,
        media_ingest=dataclasses.replace(config.media_ingest, audio_track=3),
    )
    after = real_cells(tmp_path, source, other_audio)
    for stage in ("frames", "slides", "ocr", "glossary"):
        assert before[stage].key == after[stage].key, stage
    for stage in ("audio", "vad", "asr", "punctuation", "merge"):
        assert before[stage].key != after[stage].key, stage

    other_fps = dataclasses.replace(
        config,
        media_ingest=dataclasses.replace(config.media_ingest, frames_fps=4.0),
    )
    after = real_cells(tmp_path, source, other_fps)
    # ASR потребляет глоссарий, поэтому он тоже потомок ветки кадров;
    # чистая аудио-ветка (audio, vad) обязана уцелеть.
    for stage in ("audio", "vad"):
        assert before[stage].key == after[stage].key, stage
    for stage in descendants("frames"):
        assert before[stage].key != after[stage].key, stage


def test_graph_requires_topological_order(tmp_path: Path, source: Path) -> None:
    cache = Cache(tmp_path / "cache", source)
    with pytest.raises(CacheError):
        cache.chain(
            [("slides", {}), ("frames", {})], dependencies={"slides": ("frames",)}
        )


def test_non_json_params_rejected(tmp_path: Path, source: Path) -> None:
    """Множество в параметрах даёт свой ключ в каждом процессе — это ошибка."""
    cache = Cache(tmp_path / "cache", source)
    with pytest.raises(CacheError):
        cache.stage("frames", {"backends": {"a", "b"}})
    with pytest.raises(CacheError):
        cache.stage("frames", {"path": Path("/tmp/x")})


def test_stat_identity_notices_edit_with_restored_mtime(tmp_path: Path) -> None:
    """Правка содержимого «в обход» mtime не должна давать ложное попадание."""
    path = tmp_path / "lecture.mp4"
    path.write_bytes(b"A" * 4096)
    before = file_identity(path, "stat")
    stat = path.stat()

    path.write_bytes(b"B" * 4096)  # тот же размер
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # тот же mtime
    assert file_identity(path, "stat") != before


def test_manual_region_invalidates_slides_and_descendants_only(
    tmp_path: Path, source: Path
) -> None:
    """Ручная область демонстрации входит в ключ `slides`: смена области
    пересчитывает слайды и всё после них, но не audio/frames/vad."""
    config = load_config()
    auto = real_cells(tmp_path, source, config)

    def with_region(region):
        return dataclasses.replace(
            config,
            slide_extraction=dataclasses.replace(
                config.slide_extraction, region=region
            ),
        )

    manual = real_cells(tmp_path, source, with_region((0, 0, 1280, 720)))
    other = real_cells(tmp_path, source, with_region((0, 0, 1280, 700)))

    assert auto["slides"].params["region"] is None
    assert manual["slides"].params["region"] == [0, 0, 1280, 720]
    for cells in (manual, other):
        changed = {s for s in STAGES if auto[s].key != cells[s].key}
        assert changed == descendants("slides"), sorted(changed)
    assert manual["slides"].key != other["slides"].key
    assert manual["frames"].key == auto["frames"].key
