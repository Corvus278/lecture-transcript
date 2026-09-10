"""Офлайн-режим: запрет сетевых обращений библиотек моделей.

Требование «Полностью локальная обработка»: ни один фрагмент аудио не
покидает машину. Сам код стадии в сеть не ходит; риск создают библиотеки
моделей, которые по умолчанию норовят сходить в Hugging Face Hub за весами
или за проверкой версии.

Поэтому перед прогоном выставляются переменные окружения, переводящие эти
библиотеки в режим «только локальные файлы». Отсутствие весов на диске в
этом режиме даёт честную ошибку, а не тихую загрузку из сети.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["OFFLINE_ENV", "enable_offline_mode", "offline_mode", "is_offline"]

#: Переменные, переводящие библиотеки моделей в офлайн.
OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}


def enable_offline_mode(env: dict[str, str] | None = None) -> dict[str, str | None]:
    """Выставить офлайн-переменные окружения.

    Возвращает предыдущие значения (``None`` — переменной не было), чтобы
    вызывающая сторона могла их вернуть.
    """
    values = env if env is not None else OFFLINE_ENV
    previous: dict[str, str | None] = {}
    for key, value in values.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def is_offline() -> bool:
    """Включён ли офлайн-режим для библиотек Hugging Face."""
    return os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0", "false", "False")


@contextmanager
def offline_mode(env: dict[str, str] | None = None) -> Iterator[None]:
    """Контекст с офлайн-переменными; на выходе окружение восстанавливается."""
    previous = enable_offline_mode(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
