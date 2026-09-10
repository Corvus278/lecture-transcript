"""Общие фикстуры и настройки тестов пайплайна.

Эталонная запись `wr_20260909_1150.mp4` (~830 МБ) в репозиторий не кладётся.
Путь к ней берётся из переменной окружения ``LECTURE_REFERENCE_MP4``,
по умолчанию ``~/Downloads/wr_20260909_1150.mp4``.

Короткие фрагменты для быстрых тестов лежат в ``tests/fixtures/`` и
генерируются скриптом ``tools/make_fixtures.py``. Каталог в .gitignore,
поэтому на чистой машине фрагментов нет. По умолчанию тесты, которым нужен
отсутствующий фрагмент, скипаются с подсказкой; чтобы разрешить автоматическую
нарезку, выставь ``LECTURE_FIXTURES_AUTOGEN=1`` — тогда недостающий фрагмент
будет вырезан из эталона на лету (по одному фрагменту, а не вся запись).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
MANIFEST_PATH = TESTS_DIR / "fixtures_manifest.json"
MAKE_FIXTURES = REPO_ROOT / "tools" / "make_fixtures.py"

ENV_REFERENCE = "LECTURE_REFERENCE_MP4"
DEFAULT_REFERENCE = "~/Downloads/wr_20260909_1150.mp4"
ENV_AUTOGEN = "LECTURE_FIXTURES_AUTOGEN"

CLIP_NAMES = (
    "clip_slide",
    "clip_no_slide",
    "clip_slide_switch",
    "clip_marker",
    "clip_speech",
)


# --------------------------------------------------------------------------
# Маркеры окружения
# --------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Скипать тесты, требующие того, чего в этом окружении нет.

    ``gpu`` и ``models`` скипаются всегда, когда CUDA недоступна: рабочая
    машина разработки — macOS без CUDA. ``reference`` НЕ скипается, если
    эталонная запись на месте — она и есть основной источник правды.
    """
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="нет CUDA GPU в этом окружении")
    skip_models = pytest.mark.skip(
        reason="нет локально скачанных весов моделей / CUDA в этом окружении"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        elif "models" in item.keywords:
            item.add_marker(skip_models)


def _cuda_available() -> bool:
    """Есть ли в окружении рабочая CUDA GPU."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover — сломанная установка torch
        return False


# --------------------------------------------------------------------------
# Эталонная запись и манифест
# --------------------------------------------------------------------------


def reference_path() -> Path:
    """Путь к эталонной записи по env-переменной (без проверки существования)."""
    return Path(os.environ.get(ENV_REFERENCE) or DEFAULT_REFERENCE).expanduser()


@pytest.fixture(scope="session")
def reference_mp4() -> Path:
    """Эталонная запись wr_20260909_1150.mp4; скип, если её нет на машине."""
    path = reference_path()
    if not path.exists():
        pytest.skip(
            f"нет эталонной записи {path}; "
            f"задай путь через переменную окружения {ENV_REFERENCE}"
        )
    return path


@pytest.fixture(scope="session")
def fixtures_manifest() -> dict:
    """Разобранный tests/fixtures_manifest.json."""
    if not MANIFEST_PATH.exists():
        pytest.skip(
            f"нет {MANIFEST_PATH.relative_to(REPO_ROOT)}; "
            "запусти python tools/make_fixtures.py"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def clip_specs(fixtures_manifest: dict) -> dict[str, dict]:
    """Описания фрагментов из манифеста, по имени."""
    return {clip["name"]: clip for clip in fixtures_manifest["clips"]}


# --------------------------------------------------------------------------
# Фрагменты
# --------------------------------------------------------------------------


def _generate_clip(name: str) -> None:
    """Нарезать один фрагмент скриптом tools/make_fixtures.py."""
    subprocess.run(
        [sys.executable, str(MAKE_FIXTURES), "--only", name],
        check=True,
        cwd=str(REPO_ROOT),
    )


def resolve_clip(name: str) -> Path:
    """Путь к фрагменту; автогенерация по env либо скип с подсказкой."""
    if not MANIFEST_PATH.exists():
        pytest.skip(
            f"нет {MANIFEST_PATH.relative_to(REPO_ROOT)}; "
            "запусти python tools/make_fixtures.py"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    spec = next((c for c in manifest["clips"] if c["name"] == name), None)
    if spec is None:
        pytest.fail(f"фрагмент {name} не описан в манифесте")

    path = REPO_ROOT / spec["file"]
    if path.exists():
        return path

    if os.environ.get(ENV_AUTOGEN) == "1":
        if not reference_path().exists():
            pytest.skip(
                f"нет фрагмента {path.name} и нет эталонной записи "
                f"{reference_path()} для его нарезки"
            )
        _generate_clip(name)
        if path.exists():
            return path

    pytest.skip(
        f"нет фрагмента {path.name}: запусти python tools/make_fixtures.py "
        f"(или выставь {ENV_AUTOGEN}=1 для автогенерации)"
    )
    raise AssertionError("недостижимо")  # pragma: no cover


def _clip_fixture(name: str):
    @pytest.fixture(scope="session", name=name)
    def _fixture() -> Path:
        return resolve_clip(name)

    return _fixture


clip_slide = _clip_fixture("clip_slide")
clip_no_slide = _clip_fixture("clip_no_slide")
clip_slide_switch = _clip_fixture("clip_slide_switch")
clip_marker = _clip_fixture("clip_marker")
clip_speech = _clip_fixture("clip_speech")
