"""Кэш промежуточных артефактов по стадиям (design D8).

Идея простая и без магии:

    ключ стадии = sha256( идентичность входного файла
                        + параметры стадии
                        + ключи стадий-предшественниц )

Стадии образуют не цепочку, а граф (design D2: ветка слайдов и ветка аудио
независимы и сходятся только на merge). Смена параметра стадии меняет её ключ
и, через `parents`, ключи всех её потомков — но не трогает ни предков, ни
соседнюю ветку. Их артефакты остаются валидными.

Раскладка на диске::

    <cache_dir>/<stage>/<key>/meta.json   # параметры + JSON-полезная нагрузка
    <cache_dir>/<stage>/<key>/...         # любые файлы стадии (PNG, WAV, ...)
    <cache_dir>/.tmp/<stage>-<key>-<uid>/ # незавершённая сборка

Запись атомарна: стадия пишет во временный каталог, `meta.json` кладётся туда же
последним, и только потом каталог переименовывается на место. Прерывание на
любом шаге оставляет мусор в `.tmp`, но не даёт ложного попадания в кэш.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import PipelineError

__all__ = [
    "CacheError",
    "IdentityMode",
    "file_identity",
    "StageCache",
    "Cache",
]

# Размер чанка для потокового/выборочного хэширования входного файла.
_CHUNK_SIZE = 1 << 20  # 1 МиБ
_KEY_LENGTH = 32  # hex-символов от sha256
_META_NAME = "meta.json"
_TMP_DIR = ".tmp"

IdentityMode = str  # "stat" | "sample" | "content"
_IDENTITY_MODES = ("stat", "sample", "content")


class CacheError(PipelineError):
    """Ошибка слоя кэша: битые метаданные, промах при обязательном чтении."""


# --------------------------------------------------------------------------
# Идентичность входного файла
# --------------------------------------------------------------------------


def file_identity(path: Path, mode: IdentityMode = "stat") -> dict[str, Any]:
    """Описать входной файл так, чтобы его подмена ломала ключ кэша.

    Режимы:

    * ``stat``    — путь + размер + mtime + ctime + inode. Дёшево, годится
      по умолчанию; правку содержимого «в обход» метаданных не ловит.
    * ``sample``  — размер + sha256 первого, среднего и последнего чанка.
      Компромисс для больших файлов: копия под другим именем даёт попадание,
      а правка содержимого — промах, при этом читается 3 МиБ, а не 830.
    * ``content`` — потоковый sha256 всего файла. Честно, но на 830 МБ это
      секунды дискового чтения, поэтому включается явно.
    """
    if mode not in _IDENTITY_MODES:
        raise CacheError(
            f"неизвестный режим идентичности {mode!r}; "
            f"доступны: {', '.join(_IDENTITY_MODES)}"
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise CacheError(f"входной файл не найден: {resolved}")
    stat = resolved.stat()

    if mode == "stat":
        # ctime_ns и inode добавлены к mtime намеренно: правка содержимого без
        # изменения размера с восстановленным mtime (`touch -r`, rsync -a)
        # иначе давала бы ложное попадание. Полную гарантию даёт только
        # режим `content`.
        return {
            "mode": "stat",
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino,
        }

    if mode == "sample":
        return {
            "mode": "sample",
            "size": stat.st_size,
            "sha256": _sample_digest(resolved, stat.st_size),
        }

    return {
        "mode": "content",
        "size": stat.st_size,
        "sha256": _stream_digest(resolved),
    }


def _stream_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_digest(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    offsets = sorted({0, max(size // 2 - _CHUNK_SIZE // 2, 0), max(size - _CHUNK_SIZE, 0)})
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(handle.read(_CHUNK_SIZE))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Стадия
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageCache:
    """Ячейка кэша одной стадии.

    Типичное использование::

        cell = cache.stage("frames", params)
        if cell.hit():
            payload = cell.load()
        else:
            build = cell.new_build_dir()
            ...  # стадия пишет файлы в build
            cell.save(payload, build_dir=build)
    """

    name: str
    key: str
    root: Path
    params: Mapping[str, Any]
    identity: Mapping[str, Any]
    parent_keys: tuple[str, ...] = ()
    enabled: bool = True

    @property
    def path(self) -> Path:
        """Каталог артефактов стадии. Существует только после `save`."""
        return self.root / self.name / self.key

    @property
    def meta_path(self) -> Path:
        return self.path / _META_NAME

    # -- чтение --------------------------------------------------------

    def hit(self) -> bool:
        """Есть ли готовый артефакт. Отключённый кэш всегда промахивается."""
        if not self.enabled:
            return False
        return self._read_meta() is not None

    def load(self) -> Any:
        """Полезная нагрузка стадии из meta.json. Промах — ошибка."""
        meta = self._read_meta()
        if meta is None:
            raise CacheError(f"промах кэша стадии {self.name!r} (ключ {self.key})")
        return meta.get("payload")

    def meta(self) -> dict[str, Any] | None:
        """Полные метаданные попадания либо None."""
        return self._read_meta()

    def _read_meta(self) -> dict[str, Any] | None:
        meta_path = self.meta_path
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(meta, dict) or meta.get("key") != self.key:
            return None
        return meta

    # -- запись --------------------------------------------------------

    def new_build_dir(self) -> Path:
        """Свежий временный каталог, куда стадия пишет свои файлы."""
        build_dir = self.root / _TMP_DIR / f"{self.name}-{self.key}-{uuid.uuid4().hex}"
        build_dir.mkdir(parents=True, exist_ok=False)
        return build_dir

    def save(self, payload: Any = None, *, build_dir: Path | None = None) -> Path:
        """Атомарно зафиксировать результат стадии и вернуть его каталог.

        `payload` — JSON-совместимые данные (пути храните строками
        относительно каталога стадии, чтобы кэш оставался переносимым).
        `build_dir` — каталог из `new_build_dir`, если стадия писала файлы;
        без него создаётся пустой.
        """
        build = build_dir if build_dir is not None else self.new_build_dir()
        build = Path(build)
        if not build.is_dir():
            raise CacheError(f"каталог сборки не найден: {build}")

        meta = {
            "stage": self.name,
            "key": self.key,
            "parent_keys": list(self.parent_keys),
            "params": dict(self.params),
            "identity": dict(self.identity),
            "created_at": time.time(),
            "payload": payload,
        }
        # meta.json пишется последним и внутри временного каталога —
        # точка фиксации ровно одна: переименование ниже.
        (build / _META_NAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            stale = target.with_name(f"{target.name}.stale-{uuid.uuid4().hex}")
            os.replace(target, stale)
            shutil.rmtree(stale, ignore_errors=True)
        os.replace(build, target)
        return target

    def discard(self) -> None:
        """Убрать артефакт стадии с диска (принудительный пересчёт)."""
        shutil.rmtree(self.path, ignore_errors=True)


# --------------------------------------------------------------------------
# Кэш прогона
# --------------------------------------------------------------------------


class Cache:
    """Кэш одного прогона: фиксированный вход + цепочка стадий."""

    def __init__(
        self,
        directory: str | Path,
        source: str | Path,
        *,
        enabled: bool = True,
        identity: IdentityMode = "stat",
    ) -> None:
        self.root = Path(directory).expanduser()
        self.source = Path(source).expanduser()
        self.enabled = enabled
        self.identity_mode = identity
        self.identity = file_identity(self.source, identity)

    def stage(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        parents: Sequence[StageCache] = (),
    ) -> StageCache:
        """Ячейка стадии.

        `parents` — ячейки стадий, чей результат эта стадия потребляет.
        Пустой список — корневая стадия: она зависит только от входного файла
        и своих параметров (например, `audio` и `frames` — две независимые
        корневые стадии, поэтому смена аудиодорожки не трогает кадры).
        """
        params = dict(params or {})
        parent_keys = tuple(sorted(cell.key for cell in parents))
        key = _hash_key(
            {
                "stage": name,
                "params": params,
                "identity": dict(self.identity),
                "parents": list(parent_keys),
            }
        )
        return StageCache(
            name=name,
            key=key,
            root=self.root,
            params=params,
            identity=self.identity,
            parent_keys=parent_keys,
            enabled=self.enabled,
        )

    def chain(
        self,
        stages: Iterable[tuple[str, Mapping[str, Any]]],
        dependencies: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, StageCache]:
        """Построить ключи всех стадий разом, в порядке их следования.

        `dependencies` — граф зависимостей `стадия -> стадии-предшественницы`
        (design D2). Без него стадии связываются линейно: каждая зависит от
        предыдущей. Стадии перечисляются в топологическом порядке: все
        предшественницы обязаны быть построены раньше.

        Возвращает словарь `имя стадии -> ячейка`; порядок вставки сохранён.
        """
        cells: dict[str, StageCache] = {}
        previous: StageCache | None = None
        for name, params in stages:
            if name in cells:
                raise CacheError(f"стадия {name!r} встречается в графе дважды")
            if dependencies is None:
                parents = () if previous is None else (previous,)
            else:
                parent_names = tuple(dependencies.get(name, ()))
                unknown = [p for p in parent_names if p not in cells]
                if unknown:
                    raise CacheError(
                        f"стадия {name!r} зависит от {', '.join(unknown)}, "
                        "но они не построены раньше неё"
                    )
                parents = tuple(cells[p] for p in parent_names)
            cell = self.stage(name, params, parents=parents)
            cells[name] = cell
            previous = cell
        return cells

    def clear(self) -> None:
        """Снести весь каталог кэша."""
        shutil.rmtree(self.root, ignore_errors=True)


def _hash_key(payload: Mapping[str, Any]) -> str:
    """Детерминированный ключ по каноническому JSON.

    Без `default=str`: подстановка `str(obj)` дала бы `set`-у и любому объекту
    свой ключ в каждом процессе (порядок множества зависит от PYTHONHASHSEED,
    repr объекта — от адреса в памяти), то есть кэш молча перестал бы
    попадать между запусками. Несериализуемый параметр — ошибка, а не сюрприз.
    """
    try:
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except TypeError as exc:
        raise CacheError(
            "параметры стадии должны быть JSON-совместимы "
            f"(списки вместо множеств, строки вместо объектов): {exc}"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_KEY_LENGTH]
