"""Стадия slide-ocr: изображение слайда -> Markdown-фрагменты и глоссарий.

Публичный API стадии. Бэкенды регистрируются здесь, при импорте пакета;
их конструкторы дешёвые, тяжёлые библиотеки импортируются лениво, поэтому
импорт пакета безопасен в окружении без установленных моделей.

Имена бэкендов (их читает CLI через `available_backend_names()`):
    "hybrid" — PaddleOCR + pix2tex, по умолчанию (design D5);
    "paddle" — только печатный текст, без формул;
    "vlm"    — Qwen2.5-VL 4bit, альтернатива за тем же контрактом.
"""

from __future__ import annotations

from .assemble import (
    AssembleConfig,
    DEFAULT_ASSEMBLE,
    assemble_slide_ocr,
    image_link,
    is_unreliable,
    mark_low_confidence,
    recognize_slide,
    recognize_slides,
    render_markdown,
    sort_fragments,
)
from .glossary import (
    DEFAULT_GLOSSARY,
    GlossaryConfig,
    build_glossary,
    extract_terms,
    find_decoration_lines,
)
from .hybrid_backend import HybridBackend
from .offline import NetworkAccessDenied, block_network, enforce_offline
from .paddle_backend import PaddleTextBackend
from .pix2tex_backend import Pix2TexBackend
from .registry import (
    available_backend_names,
    ensure_available,
    get_backend,
    list_backend_names,
    register,
    unregister,
)
from .routing import (
    DEFAULT_ROUTING,
    RoutingConfig,
    has_math_symbols,
    looks_like_formula,
    looks_like_handwriting,
    route_line,
)
from .vlm_backend import VlmBackend

#: Бэкенд по умолчанию (design D5).
DEFAULT_BACKEND = "hybrid"

register(HybridBackend, name="hybrid")
register(PaddleTextBackend, name="paddle")
register(VlmBackend, name="vlm")

__all__ = [
    "AssembleConfig",
    "DEFAULT_ASSEMBLE",
    "DEFAULT_BACKEND",
    "DEFAULT_GLOSSARY",
    "DEFAULT_ROUTING",
    "GlossaryConfig",
    "HybridBackend",
    "NetworkAccessDenied",
    "PaddleTextBackend",
    "Pix2TexBackend",
    "RoutingConfig",
    "VlmBackend",
    "assemble_slide_ocr",
    "available_backend_names",
    "block_network",
    "build_glossary",
    "enforce_offline",
    "ensure_available",
    "extract_terms",
    "find_decoration_lines",
    "get_backend",
    "has_math_symbols",
    "image_link",
    "is_unreliable",
    "list_backend_names",
    "looks_like_formula",
    "looks_like_handwriting",
    "mark_low_confidence",
    "recognize_slide",
    "recognize_slides",
    "register",
    "render_markdown",
    "route_line",
    "sort_fragments",
    "unregister",
]
