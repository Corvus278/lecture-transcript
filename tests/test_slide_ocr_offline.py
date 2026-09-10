"""Отсутствие сетевых вызовов на этапе распознавания слайдов (задача 4.9)."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Sequence

import pytest

from lecture_transcript.contracts import Availability, OcrFragment, Rect, Slide
from lecture_transcript.slide_ocr import registry
from lecture_transcript.slide_ocr.offline import (
    OFFLINE_ENV,
    NetworkAccessDenied,
    block_network,
    enforce_offline,
)
from lecture_transcript.slide_ocr.assemble import recognize_slides


class _OfflineFakeBackend:
    """Фейковый бэкенд без сетевых вызовов."""

    name = "offline-fake"

    def __init__(self) -> None:
        self.unloaded = False

    def check_availability(self) -> Availability:
        return Availability(True)

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        return (
            OcrFragment(
                text="Как называются эти числа",
                kind="text",
                confidence=0.96,
                bbox=Rect(10, 10, 400, 40),
            ),
            OcrFragment(
                text=r"$\sqrt{x^2-x} = -5$",
                kind="formula",
                confidence=0.85,
                bbox=Rect(10, 80, 400, 60),
            ),
        )

    def unload(self) -> None:
        self.unloaded = True


@pytest.fixture
def offline_backend():
    backend = _OfflineFakeBackend()
    registry.register(lambda: backend, name=backend.name)
    yield backend
    registry.unregister(backend.name)


def _slides(tmp_path: Path) -> list[Slide]:
    return [
        Slide(
            index=index,
            start_s=float(index),
            end_s=float(index) + 5.0,
            region=Rect(0, 0, 800, 600),
            representative_timestamp_s=float(index) + 2.0,
            image_path=tmp_path / f"{index:03d}.png",
        )
        for index in (1, 2)
    ]


def test_офлайн_переменные_выставляются():
    env: dict[str, str] = {}
    enforce_offline(env)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert set(OFFLINE_ENV) <= set(env)


def test_заданные_пользователем_значения_не_перетираются():
    env = {"HF_HUB_OFFLINE": "0"}
    enforce_offline(env)
    assert env["HF_HUB_OFFLINE"] == "0"


def test_блокировка_сети_действительно_запрещает_сокеты():
    with block_network():
        with pytest.raises(NetworkAccessDenied):
            socket.socket()
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("example.com", 80))
        with pytest.raises(NetworkAccessDenied):
            socket.getaddrinfo("example.com", 80)
    # После выхода из блока сокеты снова работоспособны.
    socket.socket().close()


def test_этап_ocr_проходит_с_заблокированной_сетью(tmp_path: Path, offline_backend):
    slides = _slides(tmp_path)
    with block_network():
        results = recognize_slides(slides, backend_name=offline_backend.name)

    assert len(results) == 2
    assert [result.slide_index for result in results] == [1, 2]
    for result in results:
        assert result.backend == "offline-fake"
        assert f"![слайд {result.slide_index}]" in result.markdown
        assert r"$\sqrt{x^2-x} = -5$" in result.markdown
    # Модель выгружена по завершении этапа (design D7).
    assert offline_backend.unloaded is True


def test_глоссарий_собирается_с_заблокированной_сетью(tmp_path: Path, offline_backend):
    from lecture_transcript.slide_ocr import build_glossary

    with block_network():
        results = recognize_slides(_slides(tmp_path), backend_name=offline_backend.name)
        glossary = build_glossary(results)

    assert glossary.terms
