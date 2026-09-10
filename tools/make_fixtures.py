#!/usr/bin/env python3
"""Нарезка тестовых фрагментов из эталонной записи (задача 1.5).

Эталонная запись весит ~830 МБ и в репозиторий не кладётся. Путь к ней берётся
из переменной окружения ``LECTURE_REFERENCE_MP4`` (по умолчанию
``~/Downloads/wr_20260909_1150.mp4``).

Скрипт воспроизводимый: таймкоды фрагментов зашиты в ``CLIP_SPECS`` ниже,
повторный запуск на той же записи даёт те же фрагменты. Результат кладётся в
``tests/fixtures/`` (каталог в .gitignore — фрагменты генерируемые), а описание
всех фрагментов и «интересные» таймкоды записи пишутся в
``tests/fixtures_manifest.json`` (он отслеживается git'ом).

Использование::

    python tools/make_fixtures.py                 # нарезать всё, чего нет
    python tools/make_fixtures.py --force         # перерезать всё заново
    python tools/make_fixtures.py --only clip_slide clip_marker
    python tools/make_fixtures.py --manifest-only # только пересобрать манифест

Откуда взялись таймкоды: см. поле ``method`` в манифесте. Кратко — кадры
записи извлечены с частотой 1 fps в ширину 480, по ним посчитаны доля светлых
пикселей кадра (детект «демонстрация включена») и perceptual hash кропа области
демонстрации (детект смены слайда и дописывания маркером); все выбранные
интервалы дополнительно просмотрены глазами.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures_manifest.json"

ENV_VAR = "LECTURE_REFERENCE_MP4"
DEFAULT_REFERENCE = "~/Downloads/wr_20260909_1150.mp4"

# Область демонстрации в координатах кадра 1920x1080 (внутренность светлого
# поля слайда, без розовой рамки шаблона). Получена как крупнейший связный
# светлый низкодисперсный компонент по выборке кадров, см. design.md D3.
SLIDE_REGION_1080P = {"x": 64, "y": 60, "width": 1248, "height": 656}

# --------------------------------------------------------------------------
# Спецификация фрагментов. Таймкоды — секунды от начала эталонной записи.
# --------------------------------------------------------------------------

CLIP_SPECS: list[dict] = [
    {
        "name": "clip_slide",
        "file": "clip_slide.mp4",
        "start_s": 3120.0,
        "duration_s": 70.0,
        "contains": (
            "Уверенный интервал с демонстрацией: слайд «4. Дробно-рациональное "
            "уравнение P(x)/Q(x) = 0» неподвижен весь фрагмент, справа и снизу — "
            "тайлы участников. Смены слайда и дописываний внутри нет."
        ),
        "expect": {
            "slide_present": True,
            "slide_switch": False,
            "region_1080p": SLIDE_REGION_1080P,
        },
    },
    {
        "name": "clip_no_slide",
        "file": "clip_no_slide.mp4",
        "start_s": 0.0,
        "duration_s": 55.0,
        "contains": (
            "Демонстрации нет: кадр целиком занят сеткой тайлов участников "
            "(тёмный фон, аватары-заглушки и несколько живых вебок). "
            "Демонстрация включается на 60-й секунде записи, поэтому фрагмент "
            "обрывается на 55-й. Это единственный интервал записи без "
            "демонстрации длиннее 30 с. Речи мало (~34% фрагмента, пауза до "
            "22 с) — идёт сбор участников перед началом пары."
        ),
        "expect": {
            "slide_present": False,
            "slide_switch": False,
        },
    },
    {
        "name": "clip_slide_switch",
        "file": "clip_slide_switch.mp4",
        "start_s": 3210.0,
        "duration_s": 60.0,
        "contains": (
            "Переключение слайда внутри фрагмента: «4. Дробно-рациональное "
            "уравнение» -> «Пример 1. (x²−5x+6)/(x−3) = 0» примерно на 28–33-й "
            "секунде фрагмента (t≈3238–3243 с записи), то есть в середине, не на "
            "границе. На время перехода демонстрация на ~4 с пропадает и тайлы "
            "участников разворачиваются на весь кадр — так переключение слайда "
            "выглядит на этой платформе."
        ),
        "expect": {
            "slide_present": True,
            "slide_switch": True,
            "switch_at_offset_s": [28.0, 33.0],
            "region_1080p": SLIDE_REGION_1080P,
        },
    },
    {
        "name": "clip_marker",
        "file": "clip_marker.mp4",
        "start_s": 2530.0,
        "duration_s": 90.0,
        "contains": (
            "Дописывание маркером поверх слайда «Метод замены переменной», "
            "пример 2x⁴−5x²−3=0: за фрагмент от руки дописываются подстановка "
            "«Пусть x²=t», уравнение 2t²−5t−3=0, коэффициенты и дискриминант. "
            "Логический слайд один, смены слайда нет — материал для 3.4/3.5."
        ),
        "expect": {
            "slide_present": True,
            "slide_switch": False,
            "incremental_edits": True,
            "region_1080p": SLIDE_REGION_1080P,
        },
    },
    {
        "name": "clip_speech",
        "file": "clip_speech.mp4",
        "start_s": 3465.0,
        "duration_s": 60.0,
        "contains": (
            "Монолог лектора: разбор «Примера 1» вслух с одновременной записью "
            "маркером. Речь занимает ~79% фрагмента, есть пауза ~1.9 с — материал "
            "для VAD и ASR (5.x)."
        ),
        "expect": {
            "slide_present": True,
            "speech_ratio_min": 0.55,
            "min_pause_s": 1.0,
        },
    },
]

# --------------------------------------------------------------------------
# «Интересные» таймкоды записи — на них ссылаются тесты 3.2, 3.3, 3.5.
# Все значения проверены по кадрам с шагом 1 с и просмотрены глазами.
# --------------------------------------------------------------------------

TIMECODES: dict = {
    # Демонстрация впервые включается на 60-й секунде: до неё кадр занят
    # только тайлами участников.
    "demo_on_at_s": 60.0,
    "counts": {
        "no_demo_intervals": 33,
        "no_demo_total_s": 247.0,
        "no_demo_intervals_longer_than_30s": 1,
        "slide_switches": 64,
        "slide_groups": 54,
        "marker_writing_intervals": 15,
        "note": (
            "Счётчики получены автоматически при пороге hamming 40 (hash_size=16) "
            "и шаге выборки 1 с. Это опорные значения для сверки в 3.3 и 3.7, "
            "а не окончательный ручной подсчёт — порог подбирается в 3.7."
        ),
    },
    "no_demo_intervals_s": [],  # заполняется из RECON ниже
    "slide_switches_s": [],
    "slide_groups_s": [],
    "marker_writing_intervals_s": [],
    "task_3_2_probes": {
        "with_slide_s": [600.0, 4200.0],
        "without_slide_s": 2400.0,
        "note": (
            "Приблизительные таймкоды из design.md подтвердились: на t=600 и "
            "t=4200 демонстрация включена, на t=2400 выключена (интервал без "
            "демонстрации 2399–2412 с)."
        ),
    },
}


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------


def reference_path() -> Path:
    """Путь к эталонной записи: из env либо дефолт в ~/Downloads."""
    raw = os.environ.get(ENV_VAR) or DEFAULT_REFERENCE
    return Path(raw).expanduser()


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"Не найден {name} в PATH — установи ffmpeg")
    return path


def ffprobe_json(path: Path) -> dict:
    out = subprocess.run(
        [
            require_tool("ffprobe"),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def media_summary(path: Path) -> dict:
    """Краткая сводка по контейнеру: длительность и параметры потоков."""
    probe = ffprobe_json(path)
    fmt = probe["format"]
    video = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    summary: dict = {
        "duration_s": round(float(fmt["duration"]), 3),
        "size_bytes": int(fmt["size"]),
    }
    if video is not None:
        num, den = (int(x) for x in video["r_frame_rate"].split("/"))
        summary["video"] = {
            "codec": video["codec_name"],
            "width": video["width"],
            "height": video["height"],
            "fps": round(num / den, 3) if den else None,
        }
    if audio is not None:
        summary["audio"] = {
            "codec": audio["codec_name"],
            "sample_rate": int(audio["sample_rate"]),
            "channels": int(audio["channels"]),
        }
    return summary


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cut_clip(source: Path, spec: dict, dest: Path) -> None:
    """Вырезать фрагмент с перекодированием.

    Перекодирование, а не -c copy: нужен самодостаточный файл с ключевым кадром
    в начале и точными границами, иначе первые секунды фрагмента разъезжаются
    по ближайшему keyframe эталона.
    """
    cmd = [
        require_tool("ffmpeg"),
        "-v",
        "error",
        "-y",
        "-ss",
        f"{spec['start_s']:.3f}",
        "-i",
        str(source),
        "-t",
        f"{spec['duration_s']:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "50",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-ac",
        "1",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
# Разведанные интервалы (детали — в отчёте разведки, метод описан в docstring)
# --------------------------------------------------------------------------

RECON = {
    # Интервалы, где демонстрации в кадре нет. Границы с точностью 1 с
    # (полная развёртка записи в кадры 1 fps). Всего 33 интервала, суммарно
    # 247 с. Длиннее 30 с ровно один — самое начало записи, из него и нарезан
    # clip_no_slide. Все остальные короткие (3-13 с): показ на секунды
    # прерывается, тайлы участников перестраиваются на весь кадр и обратно.
    # Исключение — 3239-3243: там область демонстрации не перестраивается,
    # а гаснет в чёрное (переход между слайдами внутри clip_slide_switch).
    "no_demo_intervals_s": [
        [0.0, 60.0], [581.0, 584.0], [625.0, 630.0], [677.0, 681.0], [831.0, 836.0],
        [839.0, 846.0], [899.0, 903.0], [1015.0, 1019.0], [1841.0, 1849.0],
        [2012.0, 2017.0], [2029.0, 2034.0], [2053.0, 2057.0], [2122.0, 2126.0],
        [2399.0, 2412.0], [2464.0, 2468.0], [3018.0, 3024.0], [3063.0, 3068.0],
        [3239.0, 3243.0], [3614.0, 3624.0], [3663.0, 3668.0], [3687.0, 3691.0],
        [4183.0, 4188.0], [4226.0, 4234.0], [4678.0, 4683.0], [4710.0, 4716.0],
        [4829.0, 4832.0], [5032.0, 5039.0], [5060.0, 5071.0], [5105.0, 5115.0],
        [5320.0, 5326.0], [5372.0, 5380.0], [5392.0, 5397.0], [5458.0, 5462.0],
    ],
    # Смены слайда: пара [последняя секунда старого содержимого, первая
    # секунда нового] между соседними кадрами с включённой демонстрацией,
    # hamming >= 40 при hash_size=16. Всего 64 события.
    "slide_switches_s": [
        [243.0, 244.0], [246.0, 249.0], [281.0, 282.0], [283.0, 285.0], [373.0, 374.0],
        [382.0, 385.0], [385.0, 387.0], [399.0, 400.0], [402.0, 404.0], [409.0, 410.0],
        [412.0, 414.0], [445.0, 446.0], [449.0, 450.0], [472.0, 473.0], [474.0, 477.0],
        [580.0, 584.0], [624.0, 630.0], [676.0, 681.0], [699.0, 700.0], [703.0, 705.0],
        [720.0, 722.0], [723.0, 725.0], [796.0, 797.0], [830.0, 836.0], [838.0, 846.0],
        [898.0, 903.0], [1014.0, 1019.0], [1224.0, 1225.0], [1367.0, 1368.0],
        [1368.0, 1369.0], [1396.0, 1397.0], [1398.0, 1401.0], [1407.0, 1408.0],
        [1415.0, 1417.0], [1438.0, 1439.0], [1441.0, 1443.0], [1530.0, 1531.0],
        [1840.0, 1849.0], [2011.0, 2017.0], [2052.0, 2057.0], [2121.0, 2126.0],
        [2398.0, 2412.0], [2463.0, 2468.0], [3017.0, 3024.0], [3062.0, 3068.0],
        [3238.0, 3243.0], [3613.0, 3624.0], [3662.0, 3668.0], [3686.0, 3691.0],
        [4182.0, 4188.0], [4225.0, 4234.0], [4677.0, 4683.0], [4709.0, 4716.0],
        [4828.0, 4832.0], [5031.0, 5039.0], [5059.0, 5071.0], [5104.0, 5115.0],
        [5125.0, 5126.0], [5154.0, 5155.0], [5245.0, 5246.0], [5319.0, 5326.0],
        [5371.0, 5380.0], [5391.0, 5397.0], [5457.0, 5462.0],
    ],
    # Группы кадров между сменами — кандидаты в логические слайды (D4),
    # длиннее 4 с. 54 группы при пороге hamming 40. Значение порога и
    # окончательное число слайдов — предмет задачи 3.7, здесь только опорная
    # величина для сверки.
    "slide_groups_s": [
        [60.0, 243.0], [249.0, 281.0], [285.0, 373.0], [374.0, 382.0], [387.0, 399.0],
        [404.0, 409.0], [414.0, 445.0], [446.0, 449.0], [450.0, 472.0], [477.0, 580.0],
        [584.0, 624.0], [630.0, 676.0], [681.0, 699.0], [700.0, 703.0], [705.0, 720.0],
        [725.0, 796.0], [797.0, 830.0], [846.0, 898.0], [903.0, 1014.0],
        [1019.0, 1224.0], [1225.0, 1367.0], [1369.0, 1396.0], [1401.0, 1407.0],
        [1408.0, 1415.0], [1417.0, 1438.0], [1443.0, 1530.0], [1531.0, 1840.0],
        [1849.0, 2011.0], [2017.0, 2052.0], [2057.0, 2121.0], [2126.0, 2398.0],
        [2412.0, 2463.0], [2468.0, 3017.0], [3024.0, 3062.0], [3068.0, 3238.0],
        [3243.0, 3613.0], [3624.0, 3662.0], [3668.0, 3686.0], [3691.0, 4182.0],
        [4188.0, 4225.0], [4234.0, 4677.0], [4683.0, 4709.0], [4716.0, 4828.0],
        [4832.0, 5031.0], [5039.0, 5059.0], [5071.0, 5104.0], [5115.0, 5125.0],
        [5126.0, 5154.0], [5155.0, 5245.0], [5246.0, 5319.0], [5326.0, 5371.0],
        [5380.0, 5391.0], [5397.0, 5457.0], [5462.0, 5481.0],
    ],
    # Группы, внутри которых содержимое слайда прирастает малыми шагами —
    # лектор дописывает маркером (накопленный hamming >= 20 и не меньше
    # 8 шагов в диапазоне 4..39). Самая длинная — 2468-3017 («Метод замены
    # переменной»), из неё нарезан clip_marker.
    "marker_writing_intervals_s": [
        [387.0, 399.0], [477.0, 580.0], [630.0, 676.0], [725.0, 796.0],
        [1019.0, 1224.0], [1225.0, 1367.0], [1369.0, 1396.0], [2017.0, 2052.0],
        [2126.0, 2398.0], [2468.0, 3017.0], [3243.0, 3613.0], [3691.0, 4182.0],
        [4234.0, 4677.0], [4716.0, 4828.0], [4832.0, 5031.0],
    ],
}


# --------------------------------------------------------------------------
# Основной сценарий
# --------------------------------------------------------------------------


def build_manifest(source: Path, produced: dict[str, Path]) -> dict:
    timecodes = dict(TIMECODES)
    timecodes.update(RECON)

    clips = []
    for spec in CLIP_SPECS:
        entry = {
            "name": spec["name"],
            "file": f"tests/fixtures/{spec['file']}",
            "start_s": spec["start_s"],
            "duration_s": spec["duration_s"],
            "end_s": round(spec["start_s"] + spec["duration_s"], 3),
            "contains": spec["contains"],
            "expect": spec["expect"],
        }
        path = produced.get(spec["name"])
        if path is None:
            path = FIXTURES_DIR / spec["file"]
        if path.exists():
            entry["generated"] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                **media_summary(path),
            }
        clips.append(entry)

    return {
        "generated_by": "tools/make_fixtures.py",
        "method": (
            "Кадры эталона извлечены с частотой 1 fps в ширину 480 "
            "(ffmpeg -vf fps=1,scale=480:-2). По ним посчитаны: доля пикселей "
            "ярче 150 (детект включённой демонстрации), per-pixel медиана "
            "модуля разности соседних кадров и медианная яркость в скользящем "
            "окне 60 с (детект области демонстрации по design.md D3), "
            "perceptual hash кропа области демонстрации hash_size=16 "
            "(скачок hamming>=40 — смена слайда, серия 6..25 — дописывание "
            "маркером). Все интервалы, попавшие во фрагменты, просмотрены глазами."
        ),
        "reference": {
            "env_var": ENV_VAR,
            "default_path": DEFAULT_REFERENCE,
            "filename": "wr_20260909_1150.mp4",
            "note": "В репозиторий не копируется (~830 МБ), путь задаётся через env.",
            **media_summary(source),
        },
        "slide_region_1080p": SLIDE_REGION_1080P,
        "timecodes": timecodes,
        "clips": clips,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="NAME", help="нарезать только указанные фрагменты")
    parser.add_argument("--force", action="store_true", help="перерезать существующие фрагменты")
    parser.add_argument("--manifest-only", action="store_true", help="только пересобрать манифест")
    args = parser.parse_args(argv)

    source = reference_path()
    if not source.exists():
        sys.exit(
            f"Эталонная запись не найдена: {source}\n"
            f"Задай путь через переменную окружения {ENV_VAR}."
        )

    known = {spec["name"] for spec in CLIP_SPECS}
    if args.only:
        unknown = set(args.only) - known
        if unknown:
            sys.exit(f"Неизвестные фрагменты: {', '.join(sorted(unknown))}")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}

    for spec in CLIP_SPECS:
        dest = FIXTURES_DIR / spec["file"]
        produced[spec["name"]] = dest
        if args.manifest_only:
            continue
        if args.only and spec["name"] not in args.only:
            continue
        if dest.exists() and not args.force:
            print(f"[=] {spec['name']}: уже есть, пропускаю ({dest.name})")
            continue
        print(f"[+] {spec['name']}: {spec['start_s']:.1f}..{spec['start_s'] + spec['duration_s']:.1f} с")
        cut_clip(source, spec, dest)

    manifest = build_manifest(source, produced)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"Эталон: {source}")
    ref = manifest["reference"]
    print(
        f"  длительность {ref['duration_s']} с, "
        f"video {ref['video']['codec']} {ref['video']['width']}x{ref['video']['height']} "
        f"{ref['video']['fps']} fps, "
        f"audio {ref['audio']['codec']} {ref['audio']['sample_rate']} Гц "
        f"{ref['audio']['channels']} кан."
    )
    print()
    print(f"{'фрагмент':<18} {'начало':>9} {'длит.':>7} {'размер':>10}  содержимое")
    for clip in manifest["clips"]:
        gen = clip.get("generated")
        size = f"{gen['size_bytes'] / 1e6:.1f} МБ" if gen else "нет файла"
        head = clip["contains"].split(":")[0][:60]
        print(
            f"{clip['name']:<18} {clip['start_s']:>8.1f}с {clip['duration_s']:>6.1f}с "
            f"{size:>10}  {head}"
        )
    print()
    print(f"Манифест: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
