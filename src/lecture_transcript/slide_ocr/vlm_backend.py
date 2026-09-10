"""Альтернативный OCR-бэкенд на VLM — Qwen2.5-VL 4bit (задача 4.8, design D5).

Один проход: изображение слайда -> Markdown с LaTeX. Кода меньше, чем в
гибриде, но нужно ~6 ГБ VRAM и модель охотнее галлюцинирует в формулах,
поэтому по умолчанию бэкенд не используется — только по флагу.

Формат результата тот же, что у гибрида: последовательность `OcrFragment`.
Разбор ответа модели вынесен в чистую `parse_vlm_markdown()` — тестируется
без весов.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any, Sequence

from ..contracts import Availability, OcrFragment

#: Уверенность фрагментов VLM: собственной оценки модель не даёт.
DEFAULT_VLM_CONFIDENCE = 0.75
#: Требование по видеопамяти, о котором честно сообщает check_availability.
REQUIRED_VRAM_GB = 6.0

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = (
    "Распознай содержимое слайда. Верни Markdown: каждая строка слайда — "
    "отдельной строкой, формулы в LaTeX внутри $...$. Рукописные пометки "
    "игнорируй. Ничего не добавляй от себя."
)

_FORMULA_LINE_RE = re.compile(r"^\s*(\$\$?.+\$\$?|\\\[.+\\\]|\\\(.+\\\))\s*$")
_FENCE_RE = re.compile(r"^```.*$")


def hf_cache_dirs() -> tuple[Path, ...]:
    """Каталоги кэша Hugging Face, где могут лежать скачанные веса."""
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(variable)
        if value:
            return (Path(value),)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home) / "hub",)
    return (Path.home() / ".cache" / "huggingface" / "hub",)


def has_local_model(model_id: str) -> bool:
    """Скачаны ли веса модели — проверка без сети (задача 4.9)."""
    folder = "models--" + model_id.replace("/", "--")
    for cache in hf_cache_dirs():
        try:
            if (cache / folder).is_dir():
                return True
        except OSError:  # pragma: no cover — недоступный каталог
            continue
    return False


def _free_vram_gb(torch: Any) -> float:
    """Свободная (а не общая) видеопамять GPU 0, ГБ.

    Общая память ничего не говорит о возможности загрузить модель: на карте
    8 ГБ, из которых 7 занято, загрузка упадёт OOM уже в середине прогона, а
    спека требует сообщать о недоступности до начала обработки.
    """
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
        return float(free_bytes) / (1024 ** 3)
    except Exception:  # pragma: no cover — старый torch без mem_get_info
        total = torch.cuda.get_device_properties(0).total_memory
        return float(total) / (1024 ** 3)


def parse_vlm_markdown(
    markdown: str,
    confidence: float = DEFAULT_VLM_CONFIDENCE,
) -> list[OcrFragment]:
    """Ответ VLM -> фрагменты контракта, по строке на фрагмент."""
    fragments: list[OcrFragment] = []
    for raw_line in (markdown or "").splitlines():
        line = raw_line.strip()
        if not line or _FENCE_RE.match(line):
            continue
        if _FORMULA_LINE_RE.match(line):
            body = re.sub(r"^\$\$?|\$\$?$", "", line).strip()
            body = re.sub(r"^\\\[|\\\]$|^\\\(|\\\)$", "", body).strip()
            fragments.append(
                OcrFragment(
                    text=f"${body}$",
                    kind="formula",
                    confidence=confidence,
                    bbox=None,
                    low_confidence=False,
                )
            )
        else:
            fragments.append(
                OcrFragment(
                    text=line,
                    kind="text",
                    confidence=confidence,
                    bbox=None,
                    low_confidence=False,
                )
            )
    return fragments


class VlmBackend:
    """Qwen2.5-VL (4bit): слайд -> Markdown+LaTeX за один проход."""

    name = "vlm"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_new_tokens: int = 512) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None

    def check_availability(self) -> Availability:
        """Проверка до прогона: библиотеки, веса и СВОБОДНАЯ видеопамять."""
        try:
            for module, hint in (
                ("torch", "pip install torch"),
                ("transformers", "pip install 'transformers>=4.49'"),
                ("bitsandbytes", "pip install bitsandbytes (нужен для 4bit-квантизации)"),
            ):
                if importlib.util.find_spec(module) is None:
                    return Availability(False, f"библиотека {module!r} не установлена — {hint}")
        except Exception as exc:  # битая установка: find_spec бросает наружу
            return Availability(False, f"установка transformers повреждена: {exc}")
        if not has_local_model(self.model_id):
            listed = ", ".join(str(path) for path in hf_cache_dirs())
            return Availability(
                False,
                f"веса {self.model_id} не найдены локально (искали в {listed}) — "
                "скачайте их один раз при доступной сети: "
                f"huggingface-cli download {self.model_id}",
            )
        try:
            import torch  # ленивый импорт
        except Exception as exc:  # pragma: no cover — зависит от окружения
            return Availability(False, f"torch не импортируется: {exc}")
        if not torch.cuda.is_available():
            return Availability(
                False,
                f"нужен CUDA GPU c ~{REQUIRED_VRAM_GB:.0f} ГБ VRAM: "
                "Qwen2.5-VL в 4bit не запускается на CPU за разумное время",
            )
        free_gb = _free_vram_gb(torch)
        if free_gb < REQUIRED_VRAM_GB:
            return Availability(
                False,
                f"недостаточно свободной VRAM: {free_gb:.1f} ГБ, "
                f"нужно ~{REQUIRED_VRAM_GB:.0f} ГБ",
            )
        return Availability(True)

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            from .offline import enforce_offline

            enforce_offline()
            import torch
            from transformers import (  # ленивый импорт
                AutoProcessor,
                BitsAndBytesConfig,
            )

            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _Model
            except ImportError:  # pragma: no cover — старая версия transformers
                from transformers import AutoModelForVision2Seq as _Model

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
            self._model = _Model.from_pretrained(
                self.model_id,
                quantization_config=quantization,
                device_map="auto",
                local_files_only=True,
            )
        return self._model, self._processor

    def recognize(self, image_path: Path) -> Sequence[OcrFragment]:
        """Распознать слайд одним проходом VLM."""
        from PIL import Image

        model, processor = self._load()
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
        generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return tuple(parse_vlm_markdown(answer))

    def unload(self) -> None:
        """Выгрузить модель и освободить VRAM (design D7)."""
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover — torch может быть не установлен
            pass
