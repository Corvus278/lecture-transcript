"""Реестр ASR-бэкендов с проверкой доступности до начала прогона.

Требование спеки «Недоступный бэкенд»: система сообщает о недоступности
до начала обработки, а не в середине прогона. Поэтому pipeline обязан
получать бэкенд через :func:`ensure_available` — эта функция либо отдаёт
готовый к работе бэкенд, либо бросает ``BackendUnavailableError`` с
причиной, ещё до чтения аудио и запуска VAD.

Проверка доступности дешёвая: бэкенды смотрят наличие пакетов через
``importlib.util.find_spec`` и не импортируют тяжёлые библиотеки.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..contracts import AsrBackend, Availability, BackendUnavailableError
from .gigaam_backend import GIGAAM_BACKEND_NAME, GigaAmBackend
from .whisper_backend import WHISPER_BACKEND_NAME, WhisperBackend

__all__ = [
    "register",
    "unregister",
    "get_backend",
    "check_backend",
    "ensure_available",
    "list_backend_names",
    "available_backend_names",
    "resolve_name",
    "reset_registry",
    "DEFAULT_BACKEND_NAME",
]

#: Бэкенд по умолчанию — GigaAM-v2 для русских лекций (D6).
DEFAULT_BACKEND_NAME = GIGAAM_BACKEND_NAME


@dataclass
class _Entry:
    name: str
    factory: Callable[[], AsrBackend]
    aliases: tuple[str, ...] = ()
    instance: AsrBackend | None = field(default=None, repr=False)


_REGISTRY: dict[str, _Entry] = {}
_ALIASES: dict[str, str] = {}


def register(
    name: str,
    factory: Callable[[], AsrBackend],
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> None:
    """Зарегистрировать бэкенд под каноническим именем.

    ``factory`` вызывается лениво, при первом обращении к бэкенду: сама
    регистрация не должна ничего грузить.
    """
    if not replace and name in _REGISTRY:
        raise ValueError(f"ASR-бэкенд {name!r} уже зарегистрирован")
    _REGISTRY[name] = _Entry(name=name, factory=factory, aliases=aliases)
    for alias in aliases:
        _ALIASES[alias] = name


def unregister(name: str) -> None:
    """Убрать бэкенд из реестра (нужно тестам и переопределению конфигом)."""
    entry = _REGISTRY.pop(name, None)
    if entry is None:
        return
    for alias in entry.aliases:
        _ALIASES.pop(alias, None)


def list_backend_names() -> list[str]:
    """Все зарегистрированные канонические имена, отсортированные."""
    return sorted(_REGISTRY)


def resolve_name(name: str) -> str:
    """Развернуть псевдоним в каноническое имя; неизвестное имя — ошибка."""
    if name in _REGISTRY:
        return name
    canonical = _ALIASES.get(name)
    if canonical is not None:
        return canonical
    known = ", ".join(list_backend_names()) or "(реестр пуст)"
    raise BackendUnavailableError(
        f"ASR-бэкенд {name!r} не зарегистрирован; доступные имена: {known}"
    )


def get_backend(name: str) -> AsrBackend:
    """Вернуть экземпляр бэкенда по имени, без проверки доступности.

    Экземпляр кэшируется: модель грузится один раз, а ``unload()`` вызывается
    на том же объекте.
    """
    entry = _REGISTRY[resolve_name(name)]
    if entry.instance is None:
        entry.instance = entry.factory()
    return entry.instance


def check_backend(name: str) -> Availability:
    """Проверить доступность бэкенда. Неизвестное имя — тоже недоступность."""
    try:
        backend = get_backend(name)
    except BackendUnavailableError as exc:
        return Availability(False, str(exc))
    try:
        return backend.check_availability()
    except Exception as exc:  # noqa: BLE001 — сломанная проверка = недоступность
        return Availability(False, f"проверка доступности {name!r} упала: {exc!r}")


def ensure_available(name: str) -> AsrBackend:
    """Отдать готовый бэкенд либо бросить ``BackendUnavailableError`` с причиной.

    Единственная точка входа для pipeline: вызывается до чтения аудио, чтобы
    недоступный бэкенд отвергался до начала прогона.
    """
    availability = check_backend(name)
    if not availability.available:
        raise BackendUnavailableError(
            f"ASR-бэкенд {name!r} недоступен: {availability.reason}"
        )
    return get_backend(name)


def available_backend_names() -> list[str]:
    """Имена бэкендов, готовых к работе в текущем окружении (для CLI)."""
    return [name for name in list_backend_names() if check_backend(name).available]


def reset_registry() -> None:
    """Вернуть реестр к штатному составу. Нужно тестам."""
    _REGISTRY.clear()
    _ALIASES.clear()
    _register_builtin()


def _register_builtin() -> None:
    register(GIGAAM_BACKEND_NAME, GigaAmBackend, aliases=("gigaam", "gigaam_v2"))
    register(WHISPER_BACKEND_NAME, WhisperBackend, aliases=("whisper", "faster-whisper"))


_register_builtin()
