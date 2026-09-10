"""Гарантия локальной обработки слайдов (задача 4.9).

Два независимых средства:

* `enforce_offline()` — выставляет переменные окружения, запрещающие
  библиотекам ходить в сеть за весами и телеметрией. Вызывается в начале
  этапа, ДО ленивых импортов тяжёлых библиотек: hub-клиенты читают эти
  переменные на импорте.
* `block_network()` — жёсткий контекстный менеджер: любая попытка открыть
  сокет падает с `NetworkAccessDenied`. Используется в тестах и может быть
  включён в прогоне как страховка.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Iterator, MutableMapping

from ..contracts import PipelineError

#: Переменные окружения офлайн-режима: HuggingFace (pix2tex, Qwen2.5-VL)
#: и Paddle (PaddleOCR/PaddleX тянут веса и проверяют обновления).
OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "PADDLE_PDX_DISABLE_DEV_MODE": "1",
    "PADDLE_DISABLE_UPDATE_CHECK": "1",
    "PADDLEHUB_DISABLE_TELEMETRY": "1",
    "NO_ALBUMENTATIONS_UPDATE": "1",
}


class NetworkAccessDenied(PipelineError):
    """Попытка сетевого вызова на этапе, который обязан быть локальным."""


def enforce_offline(env: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Включить офлайн-режим для библиотек моделей.

    Возвращает выставленные переменные. Уже заданные пользователем значения
    не перетираются, если они непустые.
    """
    target = os.environ if env is None else env
    for key, value in OFFLINE_ENV.items():
        if not target.get(key):
            target[key] = value
    return dict(OFFLINE_ENV)


@contextmanager
def block_network() -> Iterator[None]:
    """Запретить любые сетевые вызовы внутри блока."""
    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def _denied(*args, **kwargs):
        raise NetworkAccessDenied(
            "Сетевой вызов на этапе распознавания слайдов запрещён: "
            "обработка обязана быть полностью локальной"
        )

    socket.socket = _denied  # type: ignore[assignment]
    socket.create_connection = _denied  # type: ignore[assignment]
    socket.getaddrinfo = _denied  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
