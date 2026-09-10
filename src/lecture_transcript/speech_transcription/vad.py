"""Детекция речи: аудио -> речевые интервалы с абсолютными таймкодами.

Два бэкенда VAD, выбор всегда явный (никакой молчаливой подмены):

- ``"silero"`` — silero-vad (ленивый импорт torch), основной путь по D7;
- ``"energy"`` — детерминированный энергетический фолбэк на numpy,
  без внешних весов; работает всегда и покрывается тестами;
- ``"auto"`` — сначала silero, при его недоступности — energy,
  и об этом выдаётся предупреждение в ``VadResult.warnings``.

Длительная тишина отсекается: пауза короче ``min_silence_s`` не разрывает
речь (близкие интервалы сливаются), пауза длиннее — разрывает, и участок
тишины в распознавание не попадает. Таймкоды интервалов всегда абсолютные,
от начала записи.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..contracts import AudioArtifact, Availability, BackendUnavailableError, SpeechInterval
from .audio_io import read_wav_mono

logger = logging.getLogger(__name__)

VadBackendName = Literal["silero", "energy", "auto"]

__all__ = [
    "VadConfig",
    "VadResult",
    "VadBackendName",
    "check_vad_backend",
    "detect_speech",
    "run_vad",
    "intervals_from_mask",
    "merge_and_pad",
]


@dataclass(frozen=True)
class VadConfig:
    """Параметры детекции речи.

    min_silence_s — минимальная длительность паузы, которая считается
        «длительной тишиной» и разрывает речь; всё, что короче, остаётся
        внутри интервала (короткие паузы не теряют слова).
    min_speech_s — интервалы короче отбрасываются как щелчки/шум.
    speech_pad_s — паддинг на границах, чтобы не срезать атаку и хвост слова.
    frame_s — шаг анализа для энергетического бэкенда.
    energy_floor_dbfs — абсолютный порог тишины.
    energy_relative_db — порог относительно пика записи; итоговый порог —
        максимум из абсолютного и относительного (устойчив к общей громкости).
    """

    min_silence_s: float = 1.0
    min_speech_s: float = 0.20
    speech_pad_s: float = 0.15
    frame_s: float = 0.03
    energy_floor_dbfs: float = -45.0
    energy_relative_db: float = -30.0
    silero_threshold: float = 0.5


@dataclass(frozen=True)
class VadResult:
    """Результат детекции: интервалы, фактический бэкенд и предупреждения."""

    intervals: tuple[SpeechInterval, ...]
    backend: str
    warnings: tuple[str, ...] = ()

    @property
    def speech_duration_s(self) -> float:
        return sum(iv.end_s - iv.start_s for iv in self.intervals)


# --------------------------------------------------------------------------
# Доступность
# --------------------------------------------------------------------------


def check_vad_backend(name: VadBackendName) -> Availability:
    """Проверить доступность бэкенда VAD без загрузки весов."""
    if name == "energy":
        return Availability(True)
    if name == "auto":
        return Availability(True)
    if name != "silero":
        return Availability(False, f"неизвестный бэкенд VAD {name!r}; доступны: silero, energy, auto")
    # Проверяются ровно те пакеты, которые импортирует _silero_intervals.
    # Имена только верхнего уровня: find_spec на подмодуле ("torch.hub")
    # импортирует родителя, то есть тянет torch в память на «дешёвой» проверке.
    if importlib.util.find_spec("torch") is None:
        return Availability(
            False,
            "не установлен torch — silero-vad не запустится; "
            "поставьте `pip install torch silero-vad` либо выберите vad_backend='energy'",
        )
    if importlib.util.find_spec("silero_vad") is None:
        return Availability(
            False,
            "не установлен пакет silero-vad; поставьте `pip install silero-vad` "
            "либо выберите vad_backend='energy'",
        )
    return Availability(True)


# --------------------------------------------------------------------------
# Публичный вход
# --------------------------------------------------------------------------


def run_vad(
    audio: AudioArtifact,
    config: VadConfig | None = None,
    backend: VadBackendName = "auto",
) -> VadResult:
    """Найти речевые интервалы в аудио выбранным бэкендом."""
    cfg = config or VadConfig()
    warnings: list[str] = []

    if backend == "auto":
        silero = check_vad_backend("silero")
        if silero.available:
            chosen: VadBackendName = "silero"
        else:
            chosen = "energy"
            message = (
                "silero-vad недоступен, используется энергетический VAD "
                f"(причина: {silero.reason})"
            )
            warnings.append(message)
            logger.warning(message)
    else:
        chosen = backend
        availability = check_vad_backend(chosen)
        if not availability.available:
            raise BackendUnavailableError(
                f"бэкенд VAD {chosen!r} недоступен: {availability.reason}"
            )

    samples, sample_rate = read_wav_mono(audio.path)
    if chosen == "silero":
        intervals = _silero_intervals(samples, sample_rate, cfg)
    else:
        intervals = _energy_intervals(samples, sample_rate, cfg)

    duration_s = audio.duration_s or (samples.shape[0] / sample_rate if sample_rate else 0.0)
    intervals = merge_and_pad(intervals, cfg, duration_s)
    return VadResult(intervals=tuple(intervals), backend=chosen, warnings=tuple(warnings))


def detect_speech(
    audio: AudioArtifact,
    config: VadConfig | None = None,
    backend: VadBackendName = "auto",
) -> list[SpeechInterval]:
    """Тонкая обёртка над :func:`run_vad`, когда предупреждения не нужны."""
    return list(run_vad(audio, config, backend).intervals)


# --------------------------------------------------------------------------
# Энергетический бэкенд (детерминированный, без внешних весов)
# --------------------------------------------------------------------------


def _energy_intervals(
    samples: np.ndarray, sample_rate: int, config: VadConfig
) -> list[SpeechInterval]:
    """Разметить речь по кадровой RMS-энергии."""
    frame_len = max(1, int(round(config.frame_s * sample_rate)))
    if samples.size < frame_len:
        return []
    frame_count = samples.shape[0] // frame_len
    if frame_count == 0:
        return []
    frames = samples[: frame_count * frame_len].reshape(frame_count, frame_len)
    # einsum суммирует квадраты сразу в float64 и возвращает только один
    # float на кадр. Прежний путь (astype(float64) + square) держал в памяти
    # две копии всей записи по 8 байт на сэмпл — на 91-минутной лекции это
    # 1.4 ГБ поверх самих сэмплов.
    mean_square = np.einsum("ij,ij->i", frames, frames, dtype=np.float64) / frame_len
    rms = np.sqrt(mean_square)
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))

    peak = float(dbfs.max())
    threshold = max(config.energy_floor_dbfs, peak + config.energy_relative_db)
    mask = dbfs >= threshold
    return intervals_from_mask(mask, config.frame_s)


def intervals_from_mask(mask: np.ndarray, frame_s: float) -> list[SpeechInterval]:
    """Собрать интервалы из булевой маски кадров. Таймкоды — от начала записи."""
    intervals: list[SpeechInterval] = []
    start_idx: int | None = None
    for idx, is_speech in enumerate(bool(v) for v in mask):
        if is_speech and start_idx is None:
            start_idx = idx
        elif not is_speech and start_idx is not None:
            intervals.append(SpeechInterval(start_idx * frame_s, idx * frame_s))
            start_idx = None
    if start_idx is not None:
        intervals.append(SpeechInterval(start_idx * frame_s, len(mask) * frame_s))
    return intervals


# --------------------------------------------------------------------------
# Слияние, отсечение тишины, паддинг
# --------------------------------------------------------------------------


def merge_and_pad(
    intervals: list[SpeechInterval], config: VadConfig, duration_s: float
) -> list[SpeechInterval]:
    """Слить близкие интервалы, отбросить слишком короткие, добавить паддинг.

    Порядок важен: сначала слияние по ``min_silence_s`` (короткая пауза не
    разрывает фразу), потом отбраковка коротышей, потом паддинг и повторное
    слияние наложившихся после паддинга интервалов.
    """
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda iv: iv.start_s)
    merged: list[SpeechInterval] = [ordered[0]]
    for interval in ordered[1:]:
        last = merged[-1]
        if interval.start_s - last.end_s < config.min_silence_s:
            merged[-1] = replace(last, end_s=max(last.end_s, interval.end_s))
        else:
            merged.append(interval)

    kept = [iv for iv in merged if iv.end_s - iv.start_s >= config.min_speech_s]
    if not kept:
        return []

    limit = duration_s if duration_s > 0 else max(iv.end_s for iv in kept)
    padded: list[SpeechInterval] = []
    for interval in kept:
        start = max(0.0, interval.start_s - config.speech_pad_s)
        end = min(limit, interval.end_s + config.speech_pad_s)
        if padded and start <= padded[-1].end_s:
            padded[-1] = replace(padded[-1], end_s=max(padded[-1].end_s, end))
        else:
            padded.append(SpeechInterval(start, end))
    return padded


# --------------------------------------------------------------------------
# silero-vad (ленивый импорт)
# --------------------------------------------------------------------------


def _silero_intervals(
    samples: np.ndarray, sample_rate: int, config: VadConfig
) -> list[SpeechInterval]:
    """Разметить речь через silero-vad. Модель грузится только здесь."""
    availability = check_vad_backend("silero")
    if not availability.available:
        raise BackendUnavailableError(f"silero-vad недоступен: {availability.reason}")

    import torch  # noqa: PLC0415 — ленивый импорт тяжёлой зависимости
    from silero_vad import get_speech_timestamps, load_silero_vad  # noqa: PLC0415

    model = load_silero_vad()
    tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    stamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sample_rate,
        threshold=config.silero_threshold,
        min_speech_duration_ms=int(config.min_speech_s * 1000),
        min_silence_duration_ms=int(config.min_silence_s * 1000),
        return_seconds=False,
    )
    return [
        SpeechInterval(float(stamp["start"]) / sample_rate, float(stamp["end"]) / sample_rate)
        for stamp in stamps
    ]
