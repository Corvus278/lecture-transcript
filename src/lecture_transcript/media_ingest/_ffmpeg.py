"""Внутренние утилиты для вызова ffmpeg/ffprobe и атомарной записи артефактов.

Модуль не входит в публичный API стадии — им пользуются probe/audio/frames.

Две сквозные идеи:

1. *Атомарность.* Любой выходной артефакт сначала пишется во временный путь
   рядом с целевым (тот же каталог -> то же устройство -> `os.replace`
   атомарен), и только полностью готовый результат переименовывается в
   целевой путь. При любой ошибке временный путь удаляется, целевой не
   создаётся. Это выполняет требование «не создаёт частичных выходных
   артефактов».
2. *Аппаратное ускорение — параметр, а не хардкод.* Целевая платформа —
   WSL2 + RTX 4060 (nvdec), разработка идёт на macOS (videotoolbox/CPU).
   Поэтому способ декодирования задаётся снаружи, а доступные способы
   определяются у самого ffmpeg.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path

from ..contracts import PipelineError

logger = logging.getLogger(__name__)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: Значение hwaccel по умолчанию. Берётся из окружения, чтобы целевая машина
#: могла включить nvdec (`LECTURE_HWACCEL=cuda`), не трогая код.
#: ``auto`` — выбрать лучший из доступных у ffmpeg, ``none`` — только CPU.
DEFAULT_HWACCEL = os.environ.get("LECTURE_HWACCEL", "auto")

#: Порядок предпочтения для ``hwaccel="auto"``: nvdec целевой платформы,
#: затем videotoolbox macOS, затем CPU.
_AUTO_PREFERENCE = ("cuda", "videotoolbox")


class MediaProcessingError(PipelineError):
    """ffmpeg/ffprobe завершился с ошибкой при обработке корректного файла."""


class HwaccelUnavailableError(PipelineError):
    """Явно запрошенное аппаратное ускорение недоступно в этой сборке ffmpeg."""


# --------------------------------------------------------------------------
# Запуск внешних утилит
# --------------------------------------------------------------------------


def run(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Запустить ffmpeg/ffprobe и вернуть результат без выброса исключения.

    Решение об ошибке принимает вызывающий код: для probe нечитаемый файл —
    это `UnreadableMediaError`, а для извлечения — `MediaProcessingError`.

    `cwd` — рабочий каталог процесса. Нужен muxer'у ``image2``: он
    разворачивает printf-шаблон по всему пути, поэтому шаблон кадров
    задаётся относительным, а каталог — через `cwd` (иначе ``%`` в имени
    любого родительского каталога ломает вывод).
    """
    logger.debug("запуск: %s", " ".join(args))
    try:
        return subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
    except FileNotFoundError as exc:  # ffmpeg не установлен
        raise MediaProcessingError(f"не найден исполняемый файл {args[0]!r}") from exc


def stderr_tail(result: subprocess.CompletedProcess[str], *, lines: int = 5) -> str:
    """Последние строки stderr — для внятного текста ошибки."""
    text = (result.stderr or "").strip()
    if not text:
        return f"код возврата {result.returncode}"
    return "\n".join(text.splitlines()[-lines:])


# --------------------------------------------------------------------------
# Аппаратное ускорение
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def available_hwaccels() -> tuple[str, ...]:
    """Список hwaccel, о которых сообщает установленный ffmpeg."""
    result = run([FFMPEG, "-hide_banner", "-hwaccels"])
    if result.returncode != 0:
        return ()
    names: list[str] = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("hardware acceleration"):
            continue
        names.append(line)
    return tuple(names)


def resolve_hwaccel(requested: str) -> str:
    """Привести запрошенный hwaccel к фактически используемому.

    Поведение выбрано осознанно и различается для явного и автоматического
    режима:

    * явное значение (``cuda``, ``videotoolbox``) недоступно -> немедленная
      `HwaccelUnavailableError`. Молчаливый откат на CPU здесь вреден:
      на 91-минутной записи он превращает две минуты декодирования в
      десятки, и пользователь узнаёт об этом только по времени прогона;
    * ``auto`` -> берётся первый доступный из `_AUTO_PREFERENCE`, иначе CPU.
      Здесь откат — это и есть смысл режима, поэтому он тихий (лог INFO).

    Возвращает ``"none"``, если декодировать нужно на CPU.
    """
    requested = (requested or "none").lower()
    if requested in ("none", "cpu", ""):
        return "none"
    if requested == "auto":
        for name in _AUTO_PREFERENCE:
            if name in available_hwaccels():
                logger.info("hwaccel=auto: выбран %s", name)
                return name
        logger.info("hwaccel=auto: аппаратное ускорение недоступно, декодирование на CPU")
        return "none"
    if requested not in available_hwaccels():
        raise HwaccelUnavailableError(
            f"аппаратное ускорение {requested!r} недоступно в этой сборке ffmpeg; "
            f"доступны: {', '.join(available_hwaccels()) or '—'}. "
            f"Укажите hwaccel='auto' или 'none' для декодирования на CPU."
        )
    return requested


def hwaccel_args(hwaccel: str) -> list[str]:
    """Аргументы ffmpeg до ``-i`` для выбранного способа декодирования."""
    if hwaccel == "none":
        return []
    # Намеренно без -hwaccel_output_format: кадры возвращаются в системную
    # память, поэтому обычные CPU-фильтры (fps, scale) работают как есть.
    return ["-hwaccel", hwaccel]


# --------------------------------------------------------------------------
# Атомарная запись
# --------------------------------------------------------------------------


def _tmp_sibling(target: Path, suffix: str) -> Path:
    return target.parent / f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}{suffix}"


@contextlib.contextmanager
def atomic_file(target: Path) -> Iterator[Path]:
    """Отдать временный путь; по успешному выходу переименовать в `target`.

    При исключении временный файл удаляется, `target` не создаётся и не
    портится.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(target, target.suffix)
    try:
        yield tmp
        if not tmp.exists():
            raise MediaProcessingError(f"ffmpeg не создал выходной файл {tmp}")
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


@contextlib.contextmanager
def atomic_dir(target: Path) -> Iterator[Path]:
    """То же для каталога артефактов (кадры).

    Если `target` уже существует, он заменяется целиком: наполовину
    заполненный каталог кадров хуже, чем его отсутствие.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(target, "")
    tmp.mkdir(parents=True)
    try:
        yield tmp
        stale: Path | None = None
        if target.exists() or target.is_symlink():
            stale = _tmp_sibling(target, ".old")
            os.replace(target, stale)
        # Удаление старого каталога вынесено ЗА подстановку нового: окно,
        # в котором `target` отсутствует, сжимается со «времени rmtree по
        # 5500 файлам» до промежутка между двумя переименованиями.
        os.replace(tmp, target)
        if stale is not None:
            _remove_path(stale)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _remove_path(path: Path) -> None:
    """Убрать путь любого типа: каталог — рекурсивно, файл/ссылку — unlink.

    `shutil.rmtree(..., ignore_errors=True)` на обычном файле тихо не делает
    ничего, и переименованный старый артефакт остаётся в каталоге навсегда.
    """
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()
    except OSError as exc:  # чужой мусор не повод ронять успешный прогон
        logger.warning("не удалось удалить временный путь %s: %s", path, exc)
