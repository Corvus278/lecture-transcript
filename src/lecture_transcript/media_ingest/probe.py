"""Определение состава дорожек входного файла через ffprobe.

Единственная точка, где пайплайн узнаёт, что есть во входном файле:
длительность, аудиодорожки (с языком и названием) и видеопоток.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..contracts import AudioTrackInfo, MediaInfo, UnreadableMediaError, VideoStreamInfo
from ._ffmpeg import FFPROBE, run, stderr_tail

logger = logging.getLogger(__name__)

__all__ = ["probe"]


def _parse_fraction(value: str | None) -> float:
    """Разобрать дробь вида ``30000/1001``; при бессмыслице вернуть 0.0."""
    if not value:
        return 0.0
    try:
        if "/" in value:
            num, _, den = value.partition("/")
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _tag(stream: dict[str, Any], name: str) -> str | None:
    tags = stream.get("tags") or {}
    if not isinstance(tags, dict):
        return None
    for key, value in tags.items():
        if key.lower() == name and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _duration(payload: dict[str, Any], streams: list[dict[str, Any]]) -> float:
    """Длительность записи: сначала контейнер, затем максимум по потокам."""
    fmt = payload.get("format") or {}
    value = _to_float(fmt.get("duration"))
    if value:
        return value
    candidates = [d for s in streams if (d := _to_float(s.get("duration")))]
    return max(candidates) if candidates else 0.0


def _audio_track(stream: dict[str, Any]) -> AudioTrackInfo:
    return AudioTrackInfo(
        index=int(stream.get("index", 0)),
        codec=str(stream.get("codec_name") or ""),
        sample_rate=int(_to_float(stream.get("sample_rate")) or 0),
        channels=int(stream.get("channels") or 0),
        language=_tag(stream, "language"),
        title=_tag(stream, "title"),
    )


def _video_stream(stream: dict[str, Any]) -> VideoStreamInfo:
    fps = _parse_fraction(stream.get("avg_frame_rate")) or _parse_fraction(
        stream.get("r_frame_rate")
    )
    return VideoStreamInfo(
        index=int(stream.get("index", 0)),
        codec=str(stream.get("codec_name") or ""),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=fps,
    )


def probe(path: Path | str) -> MediaInfo:
    """Определить состав дорожек и длительность медиафайла.

    Нечитаемый файл, отсутствующий файл или файл без медиапотоков ->
    `UnreadableMediaError` с текстом ffprobe в качестве причины.
    Отсутствие аудио или видео ошибкой здесь не является: это нормальные
    состояния, о которых должен знать вызывающий (`has_audio`/`has_video`).
    """
    path = Path(path)
    if not path.exists():
        raise UnreadableMediaError(f"файл не найден: {path}")
    if path.is_dir():
        raise UnreadableMediaError(f"это каталог, а не медиафайл: {path}")

    result = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise UnreadableMediaError(f"ffprobe не смог прочитать {path}: {stderr_tail(result)}")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise UnreadableMediaError(f"ffprobe вернул неразбираемый ответ для {path}") from exc

    streams = [s for s in (payload.get("streams") or []) if isinstance(s, dict)]
    if not streams:
        raise UnreadableMediaError(f"во входном файле нет медиапотоков: {path}")

    audio_tracks = tuple(
        _audio_track(s) for s in streams if s.get("codec_type") == "audio"
    )
    video_streams = [
        s
        for s in streams
        # обложки/превью приходят как видеопоток с disposition attached_pic
        if s.get("codec_type") == "video"
        and not (s.get("disposition") or {}).get("attached_pic")
    ]
    video = _video_stream(video_streams[0]) if video_streams else None

    info = MediaInfo(
        path=path,
        duration_s=_duration(payload, streams),
        audio_tracks=audio_tracks,
        video=video,
    )
    logger.info(
        "probe %s: %.3f с, аудиодорожек %d, видео %s",
        path.name,
        info.duration_s,
        len(info.audio_tracks),
        f"{video.width}x{video.height}@{video.fps:.3f}" if video else "нет",
    )
    if video is None:
        # Спека требует не просто продолжить в аудио-режиме, но и предупредить
        # о неполноте результата: без кадров не будет ни слайдов, ни OCR.
        logger.warning(
            "во входном файле нет видеодорожки: %s — прогон продолжится в "
            "аудио-режиме, но результат будет неполным: кадры, слайды и OCR "
            "пропускаются, в транскрипт попадёт только речь",
            path,
        )
    return info
