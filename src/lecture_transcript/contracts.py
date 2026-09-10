"""Единые контракты данных пайплайна.

Это единственный источник правды по типам, которыми обмениваются стадии.
Модули стадий (media_ingest, slide_extraction, slide_ocr, speech_transcription,
transcript_assembly) не импортируют друг друга — только этот модуль.

Все временные метки — секунды с начала записи (float), абсолютные.
Все пути — pathlib.Path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, Sequence, runtime_checkable

# --------------------------------------------------------------------------
# Ошибки
# --------------------------------------------------------------------------


class PipelineError(Exception):
    """Базовая ошибка пайплайна."""


class UnreadableMediaError(PipelineError):
    """Входной файл не читается / не является медиа."""


class MissingAudioTrackError(PipelineError):
    """Во входном файле нет аудиодорожки — работать не с чем."""


class MissingVideoTrackError(PipelineError):
    """Во входном файле нет видеодорожки.

    Не ошибка сама по себе: пайплайн допускает аудио-режим. Ошибкой становится
    попытка запросить кадры у файла без видео.
    """


class BackendUnavailableError(PipelineError):
    """Запрошенный бэкенд не зарегистрирован или его зависимости не установлены."""


# --------------------------------------------------------------------------
# media-ingest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioTrackInfo:
    index: int  # индекс потока в контейнере (ffprobe stream index)
    codec: str
    sample_rate: int
    channels: int
    language: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class VideoStreamInfo:
    index: int
    codec: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_s: float
    audio_tracks: tuple[AudioTrackInfo, ...] = ()
    video: VideoStreamInfo | None = None

    @property
    def has_audio(self) -> bool:
        return len(self.audio_tracks) > 0

    @property
    def has_video(self) -> bool:
        return self.video is not None


@dataclass(frozen=True)
class AudioArtifact:
    """Нормализованное аудио: WAV PCM s16, 16 кГц, моно."""

    path: Path
    sample_rate: int
    duration_s: float
    source_track_index: int


@dataclass(frozen=True)
class Frame:
    index: int  # порядковый номер, начиная с 0
    timestamp_s: float  # абсолютная метка от начала записи
    path: Path


@dataclass(frozen=True)
class FramesArtifact:
    directory: Path
    fps: float
    frames: tuple[Frame, ...]


# --------------------------------------------------------------------------
# slide-extraction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    """Прямоугольник в пикселях кадра, left-top origin."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class LayoutInterval:
    """Интервал стабильной раскладки кадра.

    region is None — демонстрация в этом интервале выключена.
    """

    start_s: float
    end_s: float
    region: Rect | None


@dataclass(frozen=True)
class Slide:
    """Логический слайд: группа последовательных кадров с одним содержимым."""

    index: int  # 1-based, по возрастанию времени
    start_s: float
    end_s: float
    region: Rect
    representative_timestamp_s: float
    image_path: Path  # PNG-кроп репрезентативного кадра


# --------------------------------------------------------------------------
# slide-ocr
# --------------------------------------------------------------------------

FragmentKind = Literal["text", "formula"]


@dataclass(frozen=True)
class OcrFragment:
    """Одна распознанная строка слайда.

    text — уже готовый Markdown-фрагмент: для kind="formula" это `$...$`
    с LaTeX внутри, для kind="text" — обычный текст.
    """

    text: str
    kind: FragmentKind
    confidence: float  # 0.0..1.0
    bbox: Rect | None = None
    low_confidence: bool = False


@dataclass(frozen=True)
class SlideOcr:
    slide_index: int
    image_path: Path
    fragments: tuple[OcrFragment, ...]
    markdown: str  # собранный вид слайда, всегда содержит ссылку на image_path
    unreliable: bool = False
    backend: str = ""


@dataclass(frozen=True)
class Glossary:
    """Термины лекции, собранные по всем слайдам, для подачи в ASR."""

    terms: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.terms)


@dataclass(frozen=True)
class Availability:
    """Результат проверки доступности бэкенда до начала прогона."""

    available: bool
    reason: str = ""


@runtime_checkable
class OcrBackend(Protocol):
    """Контракт OCR-бэкенда: изображение -> Markdown-фрагменты с confidence."""

    name: str

    def check_availability(self) -> Availability: ...

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]: ...

    def unload(self) -> None:
        """Выгрузить модель и освободить VRAM (D7)."""


# --------------------------------------------------------------------------
# speech-transcription
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeechInterval:
    """Интервал речи по данным VAD, абсолютные таймкоды."""

    start_s: float
    end_s: float


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float
    confidence: float | None = None


@dataclass(frozen=True)
class Transcription:
    words: tuple[Word, ...]
    backend: str
    has_punctuation: bool
    used_glossary: bool = False
    warnings: tuple[str, ...] = ()


@runtime_checkable
class AsrBackend(Protocol):
    """Контракт ASR-бэкенда: аудио + опциональный глоссарий -> слова с таймкодами."""

    name: str
    provides_punctuation: bool  # если False — нужна отдельная стадия пунктуации (D6)
    supports_glossary: bool

    def check_availability(self) -> Availability: ...

    def transcribe(
        self,
        audio: AudioArtifact,
        intervals: Sequence[SpeechInterval],
        glossary: Glossary | None = None,
    ) -> Transcription: ...

    def unload(self) -> None:
        """Выгрузить модель и освободить VRAM (D7)."""


# --------------------------------------------------------------------------
# transcript-assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptSection:
    """Секция итогового транскрипта: речь, привязанная к слайду либо вне слайдов."""

    start_s: float
    end_s: float
    slide: Slide | None
    slide_ocr: SlideOcr | None
    words: tuple[Word, ...]
    paragraphs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Transcript:
    sections: tuple[TranscriptSection, ...]
    markdown: str
    source_path: Path
    metadata: dict[str, str] = field(default_factory=dict)
