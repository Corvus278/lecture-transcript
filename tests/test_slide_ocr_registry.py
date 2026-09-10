"""Реестр OCR-бэкендов и проверка доступности до прогона (задача 4.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from lecture_transcript.contracts import (
    Availability,
    BackendUnavailableError,
    OcrBackend,
    OcrFragment,
)
from lecture_transcript import slide_ocr
from lecture_transcript.slide_ocr import registry


class _FakeBackend:
    """Фейковый бэкенд за контрактом OcrBackend."""

    def __init__(self, name: str = "fake", available: bool = True, reason: str = "") -> None:
        self.name = name
        self._available = available
        self._reason = reason
        self.unloaded = False

    def check_availability(self) -> Availability:
        return Availability(self._available, self._reason)

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        return (OcrFragment(text="строка", kind="text", confidence=0.9),)

    def unload(self) -> None:
        self.unloaded = True


@pytest.fixture
def temp_registry():
    """Регистрирует фейки и убирает их после теста."""
    registered: list[str] = []

    def _register(backend: _FakeBackend) -> str:
        name = registry.register(lambda: backend, name=backend.name)
        registered.append(name)
        return name

    yield _register
    for name in registered:
        registry.unregister(name)


def test_фейковый_бэкенд_удовлетворяет_контракту():
    assert isinstance(_FakeBackend(), OcrBackend)


def test_несуществующий_бэкенд_падает_сразу_и_внятно():
    with pytest.raises(BackendUnavailableError) as info:
        registry.get_backend("не-существует")
    message = str(info.value)
    assert "не-существует" in message
    assert "hybrid" in message and "vlm" in message  # перечислены доступные имена


def test_ensure_available_тоже_падает_на_несуществующем_имени():
    with pytest.raises(BackendUnavailableError, match="Неизвестный OCR-бэкенд"):
        registry.ensure_available("нет-такого")


def test_недоступный_бэкенд_отвергается_до_прогона(temp_registry):
    backend = _FakeBackend("fake-unavailable", available=False, reason="веса не скачаны")
    temp_registry(backend)

    with pytest.raises(BackendUnavailableError) as info:
        registry.ensure_available("fake-unavailable")
    assert "веса не скачаны" in str(info.value)
    assert "fake-unavailable" not in registry.available_backend_names()


def test_доступный_бэкенд_проходит_проверку(temp_registry):
    backend = _FakeBackend("fake-ok")
    temp_registry(backend)

    assert registry.ensure_available("fake-ok") is backend
    assert "fake-ok" in registry.available_backend_names()
    assert "fake-ok" in registry.list_backend_names()


def test_падение_check_availability_не_ломает_проверку(temp_registry):
    class _Broken(_FakeBackend):
        def check_availability(self) -> Availability:
            raise RuntimeError("импорт упал")

    temp_registry(_Broken("fake-broken"))
    assert registry.check("fake-broken").available is False
    with pytest.raises(BackendUnavailableError, match="импорт упал"):
        registry.ensure_available("fake-broken")


def test_экземпляр_бэкенда_кэшируется(temp_registry):
    temp_registry(_FakeBackend("fake-cached"))
    assert registry.get_backend("fake-cached") is registry.get_backend("fake-cached")


def test_штатные_бэкенды_зарегистрированы_и_доступность_честная():
    names = slide_ocr.list_backend_names()
    assert {"hybrid", "paddle", "vlm"} <= set(names)
    # На машине без paddleocr/pix2tex/torch бэкенды обязаны честно сказать,
    # почему они недоступны, а не падать импортом.
    for name in ("hybrid", "paddle", "vlm"):
        availability = registry.check(name)
        assert availability.available or availability.reason
