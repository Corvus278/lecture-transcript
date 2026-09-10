"""Отпечаток кода стадии в ключе кэша (design D8).

Правка исходников стадии обязана инвалидировать её и её потомков по графу,
но не предков: иначе правка шаблона вывода молча отдаёт прежний
transcript.md из кэша.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lecture_transcript import cli
from lecture_transcript.config import STAGE_DEPENDENCIES, STAGES


def descendants(stage: str) -> set[str]:
    affected = {stage}
    changed = True
    while changed:
        changed = False
        for name, parents in STAGE_DEPENDENCIES.items():
            if name not in affected and affected.intersection(parents):
                affected.add(name)
                changed = True
    return affected


# --------------------------------------------------------------------------
# Отпечаток каталога
# --------------------------------------------------------------------------


def make_package(tmp_path: Path) -> Path:
    root = tmp_path / "pkg"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("X = 1\n", encoding="utf-8")
    (root / "sub" / "render.py").write_text("TEMPLATE = '## {}'\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "stale.py").write_text("мусор кэша", encoding="utf-8")
    return root


def test_fingerprint_is_stable_and_follows_content(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    first = cli._package_fingerprint(root)
    assert cli._package_fingerprint(root) == first, "неизменённый код -> тот же отпечаток"

    (root / "sub" / "render.py").write_text("TEMPLATE = '### {}'\n", encoding="utf-8")
    edited = cli._package_fingerprint(root)
    assert edited != first, "правка шаблона не изменила отпечаток"

    (root / "sub" / "render.py").rename(root / "sub" / "renderer.py")
    assert cli._package_fingerprint(root) != edited, "переименование — тоже правка"


def test_fingerprint_ignores_pycache_and_non_python(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    first = cli._package_fingerprint(root)
    (root / "__pycache__" / "stale.py").write_text("другой мусор", encoding="utf-8")
    (root / "notes.txt").write_text("не код", encoding="utf-8")
    assert cli._package_fingerprint(root) == first


def test_fingerprint_does_not_depend_on_traversal_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_package(tmp_path)
    (root / "a.py").write_text("A = 1\n", encoding="utf-8")
    (root / "z.py").write_text("Z = 1\n", encoding="utf-8")
    natural = cli._package_fingerprint(root)

    original = Path.rglob
    monkeypatch.setattr(
        Path, "rglob", lambda self, pattern: list(reversed(sorted(original(self, pattern))))
    )
    assert cli._package_fingerprint(root) == natural


# --------------------------------------------------------------------------
# Отпечаток в ключе стадии
# --------------------------------------------------------------------------


def recording_pipeline(monkeypatch: pytest.MonkeyPatch, runs: list[str]) -> None:
    """Все стадии — заглушки, журналирующие фактическое выполнение."""

    def stub(stage: str):
        def run(ctx: cli.StageContext) -> dict:
            runs.append(stage)
            if stage == "merge":
                (ctx.build_dir / "transcript.md").write_text("# т\n", encoding="utf-8")
                return {"transcript": "transcript.md"}
            return {"stage": stage}

        return run

    monkeypatch.setattr(cli, "_stage_callable", stub)
    monkeypatch.setattr(cli, "_STAGE_MODEL_HOLDERS", {})


@pytest.fixture()
def run_args(tmp_path: Path) -> list[str]:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"fake" * 256)
    return [str(source), "--out-dir", str(tmp_path / "out"), "--log-level", "WARNING"]


def test_unchanged_code_hits_cache(monkeypatch, run_args) -> None:
    runs: list[str] = []
    recording_pipeline(monkeypatch, runs)
    assert cli.main(run_args) == 0
    assert runs == list(STAGES)
    runs.clear()
    assert cli.main(run_args) == 0
    assert runs == [], f"без правок кода стадии пересчитаны: {runs}"


@pytest.mark.parametrize(
    ("package", "first_stage"),
    [
        ("transcript_assembly", "merge"),  # правка шаблона вывода — только сборка
        ("slide_ocr", "ocr"),
        ("media_ingest", "audio"),
    ],
)
def test_code_edit_invalidates_stage_and_descendants_only(
    monkeypatch, run_args, package: str, first_stage: str
) -> None:
    runs: list[str] = []
    recording_pipeline(monkeypatch, runs)
    assert cli.main(run_args) == 0

    original = cli._package_fingerprint
    monkeypatch.setattr(
        cli,
        "_package_fingerprint",
        lambda directory: "правка" if Path(directory).name == package else original(directory),
    )
    runs.clear()
    assert cli.main(run_args) == 0

    expected: set[str] = set()
    for stage in STAGES:
        if cli._STAGE_MODULES.get(stage, "").endswith(f".{package}"):
            expected |= descendants(stage)
    assert first_stage in expected
    assert set(runs) == expected, f"пересчитаны {sorted(runs)}, ожидались {sorted(expected)}"


def test_fingerprint_computed_once_per_run(monkeypatch, run_args) -> None:
    runs: list[str] = []
    recording_pipeline(monkeypatch, runs)
    calls: dict[Path, int] = {}
    original = cli._package_fingerprint

    def counting(directory: Path) -> str:
        calls[Path(directory)] = calls.get(Path(directory), 0) + 1
        return original(directory)

    monkeypatch.setattr(cli, "_package_fingerprint", counting)
    assert cli.main(run_args) == 0
    assert calls, "отпечаток кода не считался"
    assert max(calls.values()) == 1, f"каталоги хэшируются повторно: {calls}"


def test_adapter_stage_depends_on_cli_source_package_stage_does_not(monkeypatch) -> None:
    adapter_stages = [s for s in STAGES if cli._stage_callable(s) is cli._STAGE_ADAPTERS.get(s)]
    package_stages = [
        s
        for s in STAGES
        if cli._stage_callable(s) is not None
        and cli._stage_callable(s) is not cli._STAGE_ADAPTERS.get(s)
    ]
    if not adapter_stages or not package_stages:
        pytest.skip("нужны и адаптерная стадия, и стадия с точкой входа пакета")

    stages = [adapter_stages[0], package_stages[0]]
    before = cli.stage_code_fingerprints(stages)
    cli_path = Path(cli.__file__).resolve()
    original = cli._package_fingerprint
    monkeypatch.setattr(
        cli,
        "_package_fingerprint",
        lambda directory: "правка-cli" if Path(directory) == cli_path else original(directory),
    )
    after = cli.stage_code_fingerprints(stages)
    assert after[adapter_stages[0]] != before[adapter_stages[0]]
    assert after[package_stages[0]] == before[package_stages[0]]
