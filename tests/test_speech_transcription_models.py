"""Тесты стадии speech-transcription, требующие реальных весов моделей.

Помечены маркером ``models`` (плюс ``reference`` там, где нужен фрагмент
эталонной записи) и в общем прогоне не участвуют: на машине разработки нет
ни CUDA, ни скачанных весов GigaAM-v2 / Whisper large-v3 / silero-vad.

Запуск на целевой машине:

    pytest -m "models" tests/test_speech_transcription_models.py

Фрагмент эталонной записи берётся из переменной окружения
``LECTURE_REFERENCE`` (по умолчанию ``~/Downloads/wr_20260909_1150.mp4``) и
вырезается ffmpeg во временный WAV 16 кГц моно — тот же формат, что даёт
стадия media-ingest.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from lecture_transcript.contracts import AudioArtifact, Glossary
from lecture_transcript.speech_transcription import (
    GigaAmBackend,
    SpeechTranscriptionConfig,
    VadConfig,
    WhisperBackend,
    check_vad_backend,
    run_vad,
    transcribe_speech,
)

pytestmark = pytest.mark.models

REFERENCE_DEFAULT = Path.home() / "Downloads" / "wr_20260909_1150.mp4"

#: Фрагмент, на котором проверяется терминология. Подбирается на целевой машине
#: под конкретное место записи, где термин действительно произносится.
RU_FRAGMENT = (60.0, 90.0)
MIXED_FRAGMENT = (60.0, 90.0)

#: Термины, ожидаемые в написании глоссария.
RU_TERM = "дискриминант"
LATIN_TERM = "PostgreSQL"


def _reference_path() -> Path:
    return Path(os.environ.get("LECTURE_REFERENCE", str(REFERENCE_DEFAULT)))


def _extract_fragment(dst: Path, start_s: float, end_s: float) -> AudioArtifact:
    """Вырезать фрагмент эталонной записи в WAV PCM s16 16 кГц моно."""
    source = _reference_path()
    if not source.exists():
        pytest.skip(f"эталонная запись не найдена: {source}")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y",
            "-ss", f"{start_s}", "-to", f"{end_s}",
            "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    return AudioArtifact(
        path=dst, sample_rate=16000, duration_s=end_s - start_s, source_track_index=0
    )


def _require(backend) -> None:
    availability = backend.check_availability()
    if not availability.available:
        pytest.skip(f"{backend.name}: {availability.reason}")


# --------------------------------------------------------------------------
# 5.2 — реальный silero-vad
# --------------------------------------------------------------------------


@pytest.mark.reference
def test_silero_vad_finds_speech_on_reference(tmp_path):
    availability = check_vad_backend("silero")
    if not availability.available:
        pytest.skip(availability.reason)
    audio = _extract_fragment(tmp_path / "vad.wav", *RU_FRAGMENT)

    result = run_vad(audio, VadConfig(), "silero")

    assert result.backend == "silero"
    assert result.intervals, "на фрагменте лекции должна быть найдена речь"
    assert all(0.0 <= iv.start_s < iv.end_s <= audio.duration_s for iv in result.intervals)
    assert result.speech_duration_s > 0.3 * audio.duration_s


# --------------------------------------------------------------------------
# 5.3 — GigaAM-v2 и глоссарий как hotwords
# --------------------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.slow
def test_gigaam_runs_without_glossary_with_warning(tmp_path):
    """5.3 для GigaAM: пинованная gigaam==0.1.0 не принимает hotwords.

    Решение: GigaAM честно объявляет ``supports_glossary = False``, и для него
    срабатывает деградация 5.5 — прогон успешен, глоссарий не подан,
    предупреждение выдано. Термины в написании глоссария — путь Whisper (5.4).
    """
    backend = GigaAmBackend()
    _require(backend)
    audio = _extract_fragment(tmp_path / "ru.wav", *RU_FRAGMENT)

    result = transcribe_speech(
        audio,
        Glossary(terms=(RU_TERM,)),
        config=SpeechTranscriptionConfig(asr_backend=backend.name),
        backend=backend,
    )

    assert result.words, "прогон без глоссария обязан завершиться успешно"
    assert result.used_glossary is False
    assert any("не поддерживает глоссарий" in w for w in result.warnings)
    assert result.has_punctuation is True, "после GigaAM обязана отработать стадия пунктуации"


# --------------------------------------------------------------------------
# 5.4 — Whisper large-v3 и сохранение латиницы
# --------------------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.slow
def test_whisper_keeps_latin_term(tmp_path):
    """Англоязычный термин не транслитерирован."""
    backend = WhisperBackend()
    _require(backend)
    audio = _extract_fragment(tmp_path / "mixed.wav", *MIXED_FRAGMENT)

    result = transcribe_speech(
        audio,
        Glossary(terms=(LATIN_TERM,)),
        config=SpeechTranscriptionConfig(asr_backend=backend.name),
        backend=backend,
    )

    text = " ".join(word.text for word in result.words)
    assert LATIN_TERM.lower() in text.lower()
    assert any(ch.isascii() and ch.isalpha() for ch in text), "латиница должна сохраниться"
    # Транслитерация термина — признак того, что латиница потеряна.
    assert "постгрес" not in text.lower()
    assert result.has_punctuation is True
