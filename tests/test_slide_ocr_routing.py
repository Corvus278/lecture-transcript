"""Роутинг строк в формульный распознаватель (задача 4.3, design D5)."""

from __future__ import annotations

import pytest

from lecture_transcript.slide_ocr.routing import (
    DEFAULT_ROUTING,
    RoutingConfig,
    explain_route,
    has_any_math_signal,
    has_math_symbols,
    looks_like_formula,
    looks_like_handwriting,
    route_line,
    word_ratio,
)


def test_строка_с_корнем_уходит_в_формульный_распознаватель():
    assert route_line("√(x²−x) = −5", 0.95) == "formula"


def test_строка_пример_2_остаётся_текстом():
    assert route_line("Пример 2.", 0.95) == "text"


@pytest.mark.parametrize(
    "text",
    [
        "√(x²−x) = −5",
        "x^2 + 2x - 3 = 0",
        "∫ f(x) dx",
        "α + β ≤ γ",
        "a_1, a_2, ..., a_n",
    ],
)
def test_формульные_строки(text: str):
    assert route_line(text, 0.95) == "formula"


@pytest.mark.parametrize(
    "text",
    [
        "Пример 2.",
        "Как называются эти числа",
        "Решите неравенство и запишите ответ",
        "Тема урока: квадратные уравнения",
        "",
    ],
)
def test_текстовые_строки(text: str):
    assert route_line(text, 0.95) == "text"


def test_низкая_уверенность_уводит_в_формулы_строку_с_математикой():
    # D5: низкая уверенность — признак плохо прочитанной ПЕЧАТНОЙ формулы.
    # Строка со слабыми признаками («=»), которую текстовый OCR прочитал
    # плохо, при высокой уверенности осталась бы текстом из-за доли слов.
    assert route_line("Haйдитe kopни ypaвнeния x2-5x+6=0", 0.95) == "text"
    assert route_line("Haйдитe kopни ypaвнeния x2-5x+6=0", 0.2) == "formula"
    assert "низкая уверенность" in explain_route("V(x2-x)=-5", 0.2)


def test_рукописная_пометка_в_формульный_распознаватель_не_уходит():
    # D5, категория [3]: рукописное не берётся ни одной моделью, и pix2tex
    # на такой строке выдаёт галлюцинацию вместо текста. Строка без единого
    # математического признака при любой уверенности остаётся текстом.
    assert route_line("Зaвmpa кoнmpoльная", 0.31) == "text"
    assert looks_like_handwriting("Зaвmpa кoнmpoльная", 0.31) is True
    assert "рукописная" in explain_route("Зaвmpa кoнmpoльная", 0.31)


def test_плохо_прочитанный_печатный_текст_рукописным_не_считается():
    # Уверенность низкая, но не «рукописно низкая»: строка остаётся в тексте
    # и будет помечена как неуверенная на сборке.
    assert looks_like_handwriting("Как называются эти числа", 0.5) is False
    assert route_line("Как называются эти числа", 0.5) == "text"
    # Формульная строка с плохой уверенностью рукописной тоже не считается.
    assert looks_like_handwriting("V(x2-x)=-5", 0.2) is False


def test_порог_рукописного_настраивается():
    config = RoutingConfig(handwriting_threshold=0.6)
    assert looks_like_handwriting("Зaвmpa кoнmpoльная", 0.5) is False
    assert looks_like_handwriting("Зaвmpa кoнmpoльная", 0.5, config) is True


def test_роутинг_по_уверенности_отключается_конфигом():
    config = RoutingConfig(route_low_confidence=False)
    assert route_line("Пример 2.", 0.2, config) == "text"
    # Строка с математикой при выключенном роутинге по уверенности решается
    # только по своему тексту.
    assert route_line("Haйдитe kopни ypaвнeния x2-5x+6=0", 0.2, config) == "text"


def test_порог_уверенности_настраивается():
    strict = RoutingConfig(low_confidence_threshold=0.9)
    text = "Haйдитe kopни ypaвнeния x2-5x+6=0"
    assert route_line(text, 0.8, DEFAULT_ROUTING) == "text"
    assert route_line(text, 0.8, strict) == "formula"


def test_словесная_строка_с_равно_не_считается_формулой():
    assert route_line("Дискриминант равен нулю, значит корень один", 0.95) == "text"


def test_вспомогательные_предикаты():
    assert has_math_symbols("√2") is True
    assert has_math_symbols("Пример") is False
    assert looks_like_formula("x = 5") is True
    assert word_ratio("Пример 2.") > 0.5
    assert word_ratio("x = 5") == 0.0
    assert has_any_math_signal("x = 5") is True
    assert has_any_math_signal("√2") is True
    assert has_any_math_signal("Пример") is False
