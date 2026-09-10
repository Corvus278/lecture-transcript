"""Реестр OCR-бэкендов слайдов (задача 4.1).

Бэкенд регистрируется фабрикой — объект создаётся лениво и кэшируется,
чтобы `unload()` работал по тому же экземпляру, что и `recognize()`.
Конструктор бэкенда обязан быть дешёвым: тяжёлые импорты живут внутри
`check_availability()` и первого вызова `recognize()`.

Публичный API:
    register(factory, name=None)   — зарегистрировать бэкенд
    list_backend_names()           — все зарегистрированные имена
    available_backend_names()      — имена, доступные в этом окружении
    get_backend(name)              — экземпляр бэкенда (без проверки доступности)
    ensure_available(name)         — проверка ДО начала прогона, иначе исключение
"""

from __future__ import annotations

from typing import Callable

from ..contracts import Availability, BackendUnavailableError, OcrBackend

BackendFactory = Callable[[], OcrBackend]

_FACTORIES: dict[str, BackendFactory] = {}
_INSTANCES: dict[str, OcrBackend] = {}


def register(factory: BackendFactory, name: str | None = None) -> str:
    """Зарегистрировать фабрику бэкенда.

    Если имя не задано явно, оно берётся из атрибута `name` созданного
    экземпляра (экземпляр при этом кэшируется). Повторная регистрация
    того же имени заменяет фабрику и сбрасывает кэш экземпляра.
    """
    if name is None:
        instance = factory()
        name = getattr(instance, "name", "")
        if not name:
            raise ValueError(
                "Фабрика бэкенда не задала имя: укажите register(factory, name=...) "
                "или атрибут `name` у бэкенда"
            )
        _INSTANCES[name] = instance
    else:
        _INSTANCES.pop(name, None)
    _FACTORIES[name] = factory
    return name


def unregister(name: str) -> None:
    """Убрать бэкенд из реестра (используется в тестах)."""
    _FACTORIES.pop(name, None)
    _INSTANCES.pop(name, None)


def list_backend_names() -> list[str]:
    """Все зарегистрированные имена бэкендов, по алфавиту."""
    return sorted(_FACTORIES)


def get_backend(name: str) -> OcrBackend:
    """Экземпляр бэкенда по имени.

    Доступность НЕ проверяется — для этого есть `ensure_available()`.
    Неизвестное имя падает сразу, перечисляя зарегистрированные.
    """
    if name not in _FACTORIES:
        known = ", ".join(list_backend_names()) or "нет ни одного"
        raise BackendUnavailableError(
            f"Неизвестный OCR-бэкенд слайдов: {name!r}. Доступные имена: {known}"
        )
    instance = _INSTANCES.get(name)
    if instance is None:
        instance = _FACTORIES[name]()
        _INSTANCES[name] = instance
    return instance


def check(name: str) -> Availability:
    """Проверить доступность бэкенда, не поднимая исключения."""
    backend = get_backend(name)
    try:
        return backend.check_availability()
    except Exception as exc:  # проверка доступности не имеет права падать
        return Availability(False, f"Проверка доступности бэкенда {name!r} упала: {exc}")


def ensure_available(name: str) -> OcrBackend:
    """Проверка ДО начала прогона: вернуть бэкенд либо упасть с причиной."""
    backend = get_backend(name)
    availability = check(name)
    if not availability.available:
        reason = availability.reason or "причина не указана"
        ready = ", ".join(available_backend_names()) or "нет ни одного"
        raise BackendUnavailableError(
            f"OCR-бэкенд слайдов {name!r} недоступен: {reason}. "
            f"Доступны сейчас: {ready}"
        )
    return backend


def available_backend_names() -> list[str]:
    """Имена бэкендов, готовых к работе в этом окружении."""
    return [name for name in list_backend_names() if check(name).available]
