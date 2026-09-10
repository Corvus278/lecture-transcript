"""ASR-бэкенд GigaAM-v2 — русская речь, по умолчанию (D6).

Свойства по D6: своей пунктуации нет (CTC/RNNT), поэтому
``provides_punctuation = False`` и после него обязательно включается стадия
восстановления пунктуации. Глоссарий НЕ поддерживается: в пинованной
``gigaam==0.1.0`` параметра hotwords нет, поэтому ``supports_glossary = False``
и стадия работает по сценарию 5.5 — без глоссария, с предупреждением.
Ветка hotwords в :meth:`GigaAmBackend._recognize` оставлена для версий,
которые такой параметр примут.

Все тяжёлые зависимости (``gigaam``, ``torch``) импортируются лениво: без них
модуль импортируется нормально, а :meth:`check_availability` возвращает
``Availability(False, ...)`` с внятной причиной — это позволяет отвергнуть
бэкенд до начала прогона, а не упасть ImportError на импорте пакета.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Availability, Glossary, Word
from .audio_io import write_wav_mono
from .base import IntervalAsrBackend

__all__ = ["GigaAmBackend", "GIGAAM_BACKEND_NAME", "GIGAAM_MAX_CHUNK_S"]

GIGAAM_BACKEND_NAME = "gigaam-v2"

#: Предел длины куска для GigaAM, секунды.
#: ``gigaam`` (0.1.0) бросает ``ValueError("Too long wav file")`` на входе
#: длиннее 25 с (``LONGFORM_THRESHOLD = 25 * SAMPLE_RATE``), предлагая
#: ``transcribe_longform``. Длинный путь не годится: он тянет pyannote и
#: требует ``HF_TOKEN``, то есть сеть, а обработка обязана быть локальной.
#: Поэтому режем сами, с запасом к порогу.
GIGAAM_MAX_CHUNK_S = 20.0

_INSTALL_HINT = (
    "поставьте `pip install gigaam torch` (веса ~1 ГБ качаются при первой загрузке) "
    "либо выберите другой ASR-бэкенд"
)


class GigaAmBackend(IntervalAsrBackend):
    """GigaAM-v2 RNNT: русская речь, без пунктуации, без глоссария (gigaam 0.1.0)."""

    name = GIGAAM_BACKEND_NAME
    provides_punctuation = False
    # gigaam==0.1.0 (пин образа) не принимает hotwords: transcribe(self, wav_file).
    # Честно объявляем глоссарий неподдерживаемым — срабатывает деградация 5.5
    # (прогон без глоссария с предупреждением). Термины — путь Whisper.
    supports_glossary = False
    max_chunk_s = GIGAAM_MAX_CHUNK_S

    def __init__(self, model_name: str = "v2_rnnt", device: str | None = None) -> None:
        super().__init__()
        self._model_name = model_name
        self._device = device
        self._model: Any | None = None
        self._hotwords_supported: bool | None = None

    # -- доступность ------------------------------------------------------

    def check_availability(self) -> Availability:
        """Проверить наличие зависимостей без их импорта (find_spec не грузит модуль)."""
        for package in ("torch", "gigaam"):
            if importlib.util.find_spec(package) is None:
                return Availability(
                    False, f"не установлен пакет {package!r}: {_INSTALL_HINT}"
                )
        return Availability(True)

    # -- жизненный цикл модели -------------------------------------------

    def _prepare(self, sample_rate: int) -> None:
        if self._model is not None:
            return
        import gigaam  # noqa: PLC0415 — ленивый импорт тяжёлой зависимости

        kwargs: dict[str, Any] = {}
        if self._device:
            kwargs["device"] = self._device
        self._model = gigaam.load_model(self._model_name, **kwargs)
        self._hotwords_supported = self._detect_hotwords_support(self._model)

    def unload(self) -> None:
        """Освободить память под следующую стадию (D7)."""
        self._model = None
        self._hotwords_supported = None
        if importlib.util.find_spec("torch") is not None:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():  # pragma: no cover — требует CUDA
                torch.cuda.empty_cache()

    # -- распознавание ----------------------------------------------------

    def _transcribe_chunk(
        self,
        samples: np.ndarray,
        sample_rate: int,
        glossary: Glossary | None,
    ) -> Sequence[Word]:
        """Распознать кусок; времена́ слов — от начала куска."""
        assert self._model is not None  # noqa: S101 — гарантировано _prepare
        text = self._recognize(samples, sample_rate, glossary)
        duration_s = samples.shape[0] / sample_rate if sample_rate else 0.0
        return distribute_words(text, duration_s)

    def _recognize(
        self, samples: np.ndarray, sample_rate: int, glossary: Glossary | None
    ) -> str:
        """Вызвать модель на куске. GigaAM принимает путь к WAV, поэтому пишем временный файл."""
        kwargs: dict[str, Any] = {}
        if glossary:
            if self._hotwords_supported:
                kwargs["hotwords"] = list(glossary.terms)
            else:
                # Флаг used_glossary снимается: в gigaam 0.1.0 параметра
                # hotwords нет вовсе, и отчёт не должен утверждать обратное.
                self._glossary_rejected(
                    "установленная версия gigaam не принимает hotwords — "
                    "глоссарий не подан в модель, точность на терминологии ниже"
                )

        handle, raw_path = tempfile.mkstemp(suffix=".wav", prefix="gigaam_chunk_")
        os.close(handle)
        chunk_path = Path(raw_path)
        try:
            write_wav_mono(chunk_path, samples, sample_rate)
            result = self._model.transcribe(str(chunk_path), **kwargs)  # type: ignore[union-attr]
        finally:
            chunk_path.unlink(missing_ok=True)

        if isinstance(result, str):
            return result
        if isinstance(result, dict):  # некоторые версии возвращают словарь
            return str(result.get("transcription") or result.get("text") or "")
        return str(result)

    @staticmethod
    def _detect_hotwords_support(model: Any) -> bool:
        """Понять по сигнатуре, принимает ли установленная версия hotwords."""
        try:
            signature = inspect.signature(model.transcribe)
        except (TypeError, ValueError):  # pragma: no cover — экзотические обёртки
            return False
        parameters = signature.parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return True
        return "hotwords" in parameters


def distribute_words(text: str, duration_s: float) -> list[Word]:
    """Разложить слова куска по его длительности пропорционально длине слова.

    GigaAM (CTC/RNNT в публичном API) не отдаёт по-словных таймкодов, поэтому
    внутри куска время распределяется приближённо. Точность привязки при этом
    ограничена длиной речевого интервала VAD, а не всей записи, и достаточна
    для сборки транскрипта по секциям слайдов.
    """
    tokens = text.split()
    if not tokens or duration_s <= 0:
        return []
    weights = [max(1, len(token)) for token in tokens]
    total = float(sum(weights))
    words: list[Word] = []
    cursor = 0.0
    for token, weight in zip(tokens, weights, strict=True):
        span = duration_s * (weight / total)
        start = cursor
        cursor = min(duration_s, cursor + span)
        words.append(Word(text=token, start_s=start, end_s=cursor))
    return words
