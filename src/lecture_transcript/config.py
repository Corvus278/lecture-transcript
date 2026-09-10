"""Конфигурация пайплайна: дефолты из YAML + пользовательский оверлей.

Дефолты лежат рядом в `default_config.yaml` и грузятся через importlib.resources,
чтобы работать и из установленного пакета, и из исходников.

Конфигурация разбита по стадиям. Стадии перечислены в `STAGES` в порядке
исполнения (design D2) — тот же порядок используется цепочкой кэша (design D8)
и оркестрацией CLI.

Значения, помеченные в `PLACEHOLDER_DEFAULTS`, — заведомо временные: они
подбираются на эталонной записи (задача 7.6).

Дефолты dataclass обязаны совпадать с `default_config.yaml`: YAML побеждает
при `load_config`, но конфиг, собранный прямо из dataclass, обязан давать то
же самое. Сверяется тестом `test_dataclass_defaults_match_yaml`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import PipelineError

__all__ = [
    "STAGES",
    "STAGE_DEPENDENCIES",
    "PLACEHOLDER_DEFAULTS",
    "ConfigError",
    "MediaIngestConfig",
    "SlideExtractionConfig",
    "SlideOcrConfig",
    "SpeechTranscriptionConfig",
    "TranscriptAssemblyConfig",
    "CacheConfig",
    "PipelineConfig",
    "default_config_text",
    "load_config",
    "parse_region",
]

# Стадии в топологическом порядке исполнения. Граф — из design D2:
#
#   audio  -> vad -----------------> asr -> punctuation --+
#   frames -> slides -> ocr -> глоссарий ^                 |
#                          \------------------------------+-> merge
#
# `audio` и `frames` — две КОРНЕВЫЕ стадии: ни одна не потребляет результат
# другой, обе зависят только от входного файла. Поэтому смена аудиодорожки не
# инвалидирует 5500 кадров, а смена fps не заставляет переизвлекать WAV.
# `audio` идёт первой: файл без аудиодорожки обязан отпасть сразу, а не через
# четыре минуты извлечения кадров.
STAGES: tuple[str, ...] = (
    "audio",
    "frames",
    "slides",
    "ocr",
    "glossary",
    "vad",
    "asr",
    "punctuation",
    "merge",
)

#: Стадия -> стадии, чей результат она потребляет. Ключ кэша стадии включает
#: ключи предшественниц, поэтому правка параметра инвалидирует стадию и всех
#: её потомков — и только их (design D8).
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "audio": (),
    "frames": (),
    "slides": ("frames",),
    "ocr": ("slides",),
    "glossary": ("ocr",),
    "vad": ("audio",),
    "asr": ("audio", "vad", "glossary"),
    "punctuation": ("asr",),
    "merge": ("slides", "ocr", "punctuation"),
}

# Поля с временными значениями: подбираются на эталонной записи.
PLACEHOLDER_DEFAULTS: dict[str, str] = {
    "slide_ocr.confidence_threshold": (
        "подбирается на эталонной записи (задача 7.6)"
    ),
    "transcript_assembly.section_boundary_tolerance_s": (
        "подбирается на эталонной записи (задача 7.6)"
    ),
}

_DEFAULT_CONFIG_NAME = "default_config.yaml"


class ConfigError(PipelineError):
    """Конфигурация не читается, содержит неизвестные ключи или неверные типы."""


# --------------------------------------------------------------------------
# Секции
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaIngestConfig:
    """Стадия `frames`: аудио в WAV 16 кГц моно + кадры видео."""

    frames_fps: float = 1.0
    hwaccel: str = "auto"
    audio_track: int | None = None
    audio_sample_rate: int = 16000


@dataclass(frozen=True)
class SlideExtractionConfig:
    """Стадия `slides`: область демонстрации и дедупликация (design D3, D4)."""

    variance_sample_frames: int = 100
    variance_window_s: float = 300.0
    min_region_area_ratio: float = 0.05
    min_layout_interval_s: float = 20.0
    min_absence_s: float = 2.0
    max_spread: float = 8.0
    min_brightness: float = 150.0
    # Размер perceptual hash: hash_size=16 -> 256 бит. Живёт рядом с порогом,
    # потому что порог измеряется именно в битах этого хэша.
    hash_size: int = 16
    hamming_threshold: int = 43  # подобран на эталонной записи (задача 3.7)
    # Область демонстрации вручную (x, y, width, height) в пикселях кадра;
    # None — автодетект по дисперсии (design D3). Нужна, когда детект
    # ошибается (design, Risks).
    region: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class SlideOcrConfig:
    """Стадии `ocr` и `glossary` (design D5)."""

    backend: str = "hybrid"
    confidence_threshold: float = 0.6  # ПЛЕЙСХОЛДЕР
    unreliable_fragment_ratio: float = 0.5
    glossary_min_term_length: int = 4
    glossary_max_terms: int = 500
    glossary_stopwords: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpeechTranscriptionConfig:
    """Стадии `vad`, `asr`, `punctuation` (design D6)."""

    asr_backend: str = "gigaam-v2"
    vad_backend: str = "silero"
    vad_threshold: float = 0.5
    min_silence_s: float = 1.0
    min_speech_s: float = 0.2
    speech_pad_s: float = 0.15
    punctuation: str = "auto"  # auto | always | never
    use_glossary: bool = True


@dataclass(frozen=True)
class TranscriptAssemblyConfig:
    """Стадия `merge`: слияние по времени и генерация transcript.md."""

    section_boundary_tolerance_s: float = 2.0  # ПЛЕЙСХОЛДЕР
    # «Речь вне слайдов» короче порога присоединяется к соседнему слайду;
    # 0 выключает. Обоснование — в default_config.yaml (DECISIONS 8.3.2).
    short_offslide_section_s: float = 15.0
    paragraph_pause_s: float = 2.5
    timecode_every_s: float = 120.0
    filler_cleanup: bool = True
    output_filename: str = "transcript.md"


@dataclass(frozen=True)
class CacheConfig:
    """Кэш промежуточных артефактов (design D8)."""

    directory: str = ".cache"
    identity: str = "stat"  # stat | sample | content


@dataclass(frozen=True)
class PipelineConfig:
    """Полная конфигурация прогона."""

    media_ingest: MediaIngestConfig = dataclasses.field(
        default_factory=MediaIngestConfig
    )
    slide_extraction: SlideExtractionConfig = dataclasses.field(
        default_factory=SlideExtractionConfig
    )
    slide_ocr: SlideOcrConfig = dataclasses.field(default_factory=SlideOcrConfig)
    speech_transcription: SpeechTranscriptionConfig = dataclasses.field(
        default_factory=SpeechTranscriptionConfig
    )
    transcript_assembly: TranscriptAssemblyConfig = dataclasses.field(
        default_factory=TranscriptAssemblyConfig
    )
    cache: CacheConfig = dataclasses.field(default_factory=CacheConfig)

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Плоский словарь секций — для логов и meta.json кэша."""
        return {
            section.name: _section_to_dict(getattr(self, section.name))
            for section in fields(self)
        }

    def stage_params(self, stage: str) -> dict[str, Any]:
        """Параметры, от которых зависит результат стадии.

        Именно этот словарь идёт в ключ кэша (design D8), поэтому в него
        входят только те поля, смена которых обязана инвалидировать стадию.
        """
        if stage not in STAGES:
            raise ConfigError(
                f"неизвестная стадия {stage!r}; доступны: {', '.join(STAGES)}"
            )

        mi = self.media_ingest
        se = self.slide_extraction
        so = self.slide_ocr
        st = self.speech_transcription
        ta = self.transcript_assembly

        if stage == "audio":
            # Только параметры WAV: стадия пишет audio.wav и ничего больше.
            return {
                "audio_track": mi.audio_track,
                "audio_sample_rate": mi.audio_sample_rate,
            }
        if stage == "frames":
            # Только параметры кадров: WAV извлекает соседняя корневая стадия.
            return {
                "frames_fps": mi.frames_fps,
                "hwaccel": mi.hwaccel,
            }
        if stage == "slides":
            return _section_to_dict(se)
        if stage == "ocr":
            return {
                "backend": so.backend,
                "confidence_threshold": so.confidence_threshold,
                "unreliable_fragment_ratio": so.unreliable_fragment_ratio,
            }
        if stage == "glossary":
            return {
                "glossary_min_term_length": so.glossary_min_term_length,
                "glossary_max_terms": so.glossary_max_terms,
                "glossary_stopwords": list(so.glossary_stopwords),
            }
        if stage == "vad":
            return {
                "vad_backend": st.vad_backend,
                "vad_threshold": st.vad_threshold,
                "min_silence_s": st.min_silence_s,
                "min_speech_s": st.min_speech_s,
                "speech_pad_s": st.speech_pad_s,
            }
        if stage == "asr":
            return {
                "asr_backend": st.asr_backend,
                "use_glossary": st.use_glossary,
            }
        if stage == "punctuation":
            return {"punctuation": st.punctuation}
        # stage == "merge"
        return _section_to_dict(ta)


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------


def parse_region(value: Any) -> tuple[int, int, int, int]:
    """Разобрать область демонстрации: "X,Y,W,H" или список из четырёх целых.

    x, y — неотрицательные, width, height — положительные. На мусор —
    ValueError с объяснением формата.
    """
    fmt = "ожидается X,Y,W,H — четыре целых числа в пикселях кадра"
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"область {value!r}: {fmt}")
    if len(parts) != 4:
        raise ValueError(f"область {value!r}: {fmt}")
    numbers: list[int] = []
    for part in parts:
        if isinstance(part, bool):
            raise ValueError(f"область {value!r}: {fmt}")
        if isinstance(part, int):
            numbers.append(part)
            continue
        try:
            numbers.append(int(str(part), 10))
        except ValueError:
            raise ValueError(f"область {value!r}: {fmt}") from None
    x, y, width, height = numbers
    if x < 0 or y < 0:
        raise ValueError(f"область {value!r}: X и Y не могут быть отрицательными")
    if width <= 0 or height <= 0:
        raise ValueError(f"область {value!r}: W и H должны быть больше нуля")
    return (x, y, width, height)


def default_config_text() -> str:
    """Текст дефолтного YAML — как он лежит в пакете."""
    try:
        return (
            resources.files(__package__)
            .joinpath(_DEFAULT_CONFIG_NAME)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        # Фолбэк на файл рядом с модулем: пакет запущен из исходников без
        # установленных package data.
        fallback = Path(__file__).with_name(_DEFAULT_CONFIG_NAME)
        if not fallback.is_file():
            raise ConfigError(
                f"дефолтный конфиг {_DEFAULT_CONFIG_NAME} не найден в пакете"
            ) from None
        return fallback.read_text(encoding="utf-8")


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Собрать конфигурацию: дефолты из пакета + оверлей пользовательского YAML.

    Оверлей накладывается посекционно и по ключам; неизвестная секция или
    неизвестный ключ — ошибка (опечатка в конфиге не должна тихо игнорироваться).
    """
    data = _parse_yaml(default_config_text(), source="<default_config.yaml>")

    if path is not None:
        user_path = Path(path).expanduser()
        if not user_path.is_file():
            raise ConfigError(f"файл конфигурации не найден: {user_path}")
        overlay = _parse_yaml(
            user_path.read_text(encoding="utf-8"), source=str(user_path)
        )
        data = _merge(data, overlay, source=str(user_path))

    return _build(data)


def _parse_yaml(text: str, *, source: str) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: некорректный YAML: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"{source}: ожидался словарь секций на верхнем уровне")
    return parsed


def _merge(
    base: Mapping[str, Any], overlay: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    merged = {key: dict(value) for key, value in base.items()}
    for section, values in overlay.items():
        if section not in merged:
            raise ConfigError(
                f"{source}: неизвестная секция {section!r}; "
                f"доступны: {', '.join(sorted(merged))}"
            )
        if not isinstance(values, Mapping):
            raise ConfigError(f"{source}: секция {section!r} должна быть словарём")
        unknown = set(values) - set(merged[section])
        if unknown:
            raise ConfigError(
                f"{source}: неизвестные ключи в секции {section!r}: "
                f"{', '.join(sorted(unknown))}"
            )
        merged[section].update(values)
    return merged


_SECTION_TYPES: dict[str, type] = {
    "media_ingest": MediaIngestConfig,
    "slide_extraction": SlideExtractionConfig,
    "slide_ocr": SlideOcrConfig,
    "speech_transcription": SpeechTranscriptionConfig,
    "transcript_assembly": TranscriptAssemblyConfig,
    "cache": CacheConfig,
}


def _build(data: Mapping[str, Any]) -> PipelineConfig:
    unknown = set(data) - set(_SECTION_TYPES)
    if unknown:
        raise ConfigError(f"неизвестные секции: {', '.join(sorted(unknown))}")
    sections: dict[str, Any] = {}
    for name, section_type in _SECTION_TYPES.items():
        raw = data.get(name) or {}
        sections[name] = _build_section(section_type, raw, path=name)
    return PipelineConfig(**sections)


def _build_section(section_type: type, raw: Mapping[str, Any], *, path: str) -> Any:
    known = {field.name: field for field in fields(section_type)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(
            f"{path}: неизвестные ключи: {', '.join(sorted(unknown))}"
        )
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        kwargs[key] = _coerce(known[key], value, path=f"{path}.{key}")
    return section_type(**kwargs)


def _coerce(field: dataclasses.Field, value: Any, *, path: str) -> Any:
    """Мягкое приведение типов YAML к типам dataclass-поля."""
    annotation = str(field.type)
    if value is None:
        if "None" in annotation:
            return None
        raise ConfigError(f"{path}: значение не может быть null")
    if annotation.startswith("tuple[int, int, int, int]"):
        try:
            return parse_region(value)
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from None
    if annotation.startswith("tuple"):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: ожидался список")
        return tuple(str(item) for item in value)
    if annotation.startswith("bool"):
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: ожидалось true/false")
        return value
    if annotation.startswith("int"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: ожидалось целое число")
        return value
    if annotation.startswith("float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: ожидалось число")
        return float(value)
    if annotation.startswith("str"):
        if not isinstance(value, str):
            raise ConfigError(f"{path}: ожидалась строка")
        return value
    return value


def _section_to_dict(section: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields(section):
        value = getattr(section, field.name)
        result[field.name] = list(value) if isinstance(value, tuple) else value
    return result
