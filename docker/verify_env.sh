#!/usr/bin/env bash
#
# Самопроверка окружения. Запускается ВНУТРИ контейнера на целевой машине
# (WSL2 + Docker + RTX 4060) — на dev-машине без CUDA заведомо упадёт.
#
#   docker compose -f docker/compose.yaml run --rm lecture-transcript verify_env.sh
#
# Закрывает проверки задач 1.1 и 1.2:
#   1.1 — nvidia-smi видит GPU, ffmpeg -hwaccels перечисляет cuda
#   1.2 — зависимости встали без конфликтов, torch видит GPU
#
# Код возврата: 0 — все проверки прошли, 1 — провалилась хотя бы одна.
# WARN не влияет на код возврата.

set -u -o pipefail

FAILED=0
WARNED=0
TMPDIR_SELF="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_SELF}"' EXIT

ok()   { printf '  [ OK ]   %s\n' "$*"; }
fail() { printf '  [FAIL]   %s\n' "$*"; FAILED=$((FAILED + 1)); }
warn() { printf '  [WARN]   %s\n' "$*"; WARNED=$((WARNED + 1)); }
info() { printf '           %s\n' "$*"; }
head_() { printf '\n== %s\n' "$*"; }

# --------------------------------------------------------------------------
head_ "1. Python и базовые бинарники"
# --------------------------------------------------------------------------
PY_VER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "нет")"
if [ "${PY_VER}" = "3.11" ]; then
    ok "python ${PY_VER} ($(command -v python))"
else
    fail "ожидался python 3.11, получен: ${PY_VER}"
fi

for bin in ffmpeg ffprobe pip; do
    if command -v "${bin}" >/dev/null 2>&1; then
        ok "${bin}: $(command -v "${bin}")"
    else
        fail "${bin} не найден в PATH"
    fi
done

# --------------------------------------------------------------------------
head_ "2. GPU виден в контейнере (задача 1.1)"
# --------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi отсутствует — контейнер запущен без NVIDIA runtime (нужен --gpus all)"
else
    if nvidia-smi -L > "${TMPDIR_SELF}/gpus.txt" 2>&1; then
        GPU_COUNT="$(grep -c '^GPU ' "${TMPDIR_SELF}/gpus.txt" || true)"
        if [ "${GPU_COUNT}" -ge 1 ]; then
            ok "nvidia-smi видит GPU: ${GPU_COUNT} шт."
            while IFS= read -r line; do info "${line}"; done < "${TMPDIR_SELF}/gpus.txt"
            nvidia-smi --query-gpu=name,memory.total,driver_version \
                --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
                info "${line}"
            done
            if ! grep -qi '4060' "${TMPDIR_SELF}/gpus.txt"; then
                warn "в списке нет RTX 4060 — целевая конфигурация другая, замеры из design D7 неприменимы"
            fi
        else
            fail "nvidia-smi запустился, но не перечислил ни одной GPU"
        fi
    else
        fail "nvidia-smi завершился с ошибкой:"
        while IFS= read -r line; do info "${line}"; done < "${TMPDIR_SELF}/gpus.txt"
    fi
fi

# --------------------------------------------------------------------------
head_ "3. ffmpeg с nvdec (задача 1.1)"
# --------------------------------------------------------------------------
if ffmpeg -hide_banner -hwaccels 2>/dev/null | tr -d ' \r' | grep -qx 'cuda'; then
    ok "ffmpeg -hwaccels перечисляет cuda"
else
    fail "ffmpeg -hwaccels НЕ перечисляет cuda"
    info "$(ffmpeg -hide_banner -hwaccels 2>&1 | tr '\n' ' ')"
fi

if ffmpeg -hide_banner -decoders 2>/dev/null | grep -q 'h264_cuvid'; then
    ok "декодер h264_cuvid собран (эталонная запись — h264)"
else
    fail "декодера h264_cuvid нет в сборке ffmpeg"
fi

# Библиотека прокидывается драйвером хоста, а не ставится apt-ом.
# Её отсутствие — типичный симптом NVIDIA_DRIVER_CAPABILITIES без `video`.
if ldconfig -p 2>/dev/null | grep -q 'libnvcuvid\.so'; then
    ok "libnvcuvid.so доступна (capability video проброшена)"
else
    fail "libnvcuvid.so не найдена: nvdec не заработает, нужен NVIDIA_DRIVER_CAPABILITIES=...,video"
fi

# Сквозная проверка: закодировать пробный ролик на CPU и декодировать через nvdec.
SAMPLE="${TMPDIR_SELF}/sample.mp4"
if ffmpeg -hide_banner -loglevel error -y \
        -f lavfi -i testsrc=size=640x480:rate=25 -t 1 \
        -c:v libx264 -pix_fmt yuv420p "${SAMPLE}" 2>"${TMPDIR_SELF}/enc.log"; then
    if ffmpeg -hide_banner -loglevel error -y \
            -hwaccel cuda -c:v h264_cuvid -i "${SAMPLE}" \
            -f null - 2>"${TMPDIR_SELF}/dec.log"; then
        ok "пробное декодирование h264 через nvdec прошло"
    else
        fail "пробное декодирование через nvdec упало:"
        while IFS= read -r line; do info "${line}"; done < "${TMPDIR_SELF}/dec.log"
    fi
else
    # Именно эта проверка единственная реально трогает nvdec: -hwaccels,
    # -decoders и ldconfig проходят и при неработающем декодировании.
    # Поэтому невозможность её выполнить — провал, а не предупреждение.
    fail "не удалось собрать пробный ролик libx264 — сквозную проверку nvdec выполнить нечем"
    while IFS= read -r line; do info "${line}"; done < "${TMPDIR_SELF}/enc.log"
fi

# --------------------------------------------------------------------------
head_ "4. torch видит GPU (задача 1.2)"
# --------------------------------------------------------------------------
if python - <<'PY'
import sys

try:
    import torch
except Exception as exc:  # noqa: BLE001
    print(f"  импорт torch не удался: {exc}")
    sys.exit(1)

print(f"  torch {torch.__version__}, собран под CUDA {torch.version.cuda}")

if not torch.cuda.is_available():
    print("  torch.cuda.is_available() == False")
    sys.exit(1)

count = torch.cuda.device_count()
print(f"  устройств CUDA: {count}")
for i in range(count):
    props = torch.cuda.get_device_properties(i)
    total_gb = props.total_memory / (1024 ** 3)
    print(
        f"  cuda:{i} — {props.name}, VRAM {total_gb:.2f} GB, "
        f"compute capability {props.major}.{props.minor}"
    )
    # D7: пиковый расход ~1.5 GB при бюджете 8 GB, альтернативные бэкенды до ~6 GB.
    if total_gb < 7.0:
        print(f"  ВНИМАНИЕ: VRAM {total_gb:.2f} GB меньше расчётных 8 GB (design D7)")

# Реальное вычисление, а не только доступность драйвера.
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
torch.cuda.synchronize()
assert (a @ b).shape == (512, 512)
print("  пробное матричное умножение на GPU выполнено")
PY
then
    ok "torch работает на GPU"
else
    fail "torch не видит GPU или упал при вычислении"
fi

# --------------------------------------------------------------------------
head_ "5. Зависимости пайплайна импортируются (задача 1.2)"
# --------------------------------------------------------------------------
check_import() {
    local label="$1"
    local code="$2"
    local out
    if out="$(python -c "${code}" 2>&1)"; then
        ok "${label}${out:+ — ${out}}"
    else
        fail "${label}: $(printf '%s' "${out}" | tail -n 1)"
    fi
}

check_import "numpy"        'import numpy; print(numpy.__version__)'
check_import "opencv (cv2)" 'import cv2; print(cv2.__version__)'
check_import "Pillow"       'import PIL; print(PIL.__version__)'
check_import "imagehash"    'import imagehash; print(imagehash.__version__)'
check_import "PyYAML"       'import yaml; print(yaml.__version__)'
check_import "silero-vad"   'import silero_vad; print("ok")'
check_import "gigaam"       'import gigaam; print("ok")'
check_import "faster-whisper" 'import faster_whisper, ctranslate2; print("ct2 " + ctranslate2.__version__)'
check_import "transformers" 'import transformers; print(transformers.__version__)'
check_import "pix2tex"      'from pix2tex.cli import LatexOCR; print("ok")'
check_import "paddleocr"    'import paddleocr; print(paddleocr.__version__)'

# Стадия 5.6 (D6): без этого пакета транскрипт GigaAM останется без пунктуации
# и регистра. Импорт без побочных эффектов — веса грузятся в SbertPuncCase().
check_import "sbert_punc_case_ru" 'from sbert_punc_case_ru import SbertPuncCase; print("ok")'

# Задача 4.8: альтернативный OCR-бэкенд Qwen2.5-VL 4bit.
# Класс появился только в transformers 4.49.0; device_map="auto" требует
# accelerate, 4bit — bitsandbytes.
check_import "qwen2.5-vl (transformers)" \
    'from transformers import Qwen2_5_VLForConditionalGeneration; print("ok")'
check_import "accelerate"   'import accelerate; print(accelerate.__version__)'
check_import "bitsandbytes" 'import bitsandbytes; print(bitsandbytes.__version__)'

# --------------------------------------------------------------------------
head_ "6. PaddlePaddle собран с CUDA"
# --------------------------------------------------------------------------
if python - <<'PY'
import sys

try:
    import paddle
except Exception as exc:  # noqa: BLE001
    print(f"  импорт paddle не удался: {exc}")
    sys.exit(1)

print(f"  paddle {paddle.__version__}")
if not paddle.is_compiled_with_cuda():
    print("  установлена CPU-сборка paddle: OCR пойдёт на CPU (стадия 3 из D7 замедлится)")
    sys.exit(1)
if paddle.device.cuda.device_count() < 1:
    print("  paddle собран с CUDA, но не видит ни одного устройства")
    sys.exit(1)
print(f"  устройств CUDA для paddle: {paddle.device.cuda.device_count()}")
PY
then
    ok "paddle работает на GPU"
else
    fail "paddle не видит GPU"
fi

# --------------------------------------------------------------------------
head_ "7. Конфликты версий в установленном наборе"
# --------------------------------------------------------------------------
if pip check > "${TMPDIR_SELF}/pipcheck.txt" 2>&1; then
    ok "pip check: конфликтов зависимостей нет"
else
    # Конфликты не всегда фатальны (paddlex/pix2tex пинуют смежные пакеты),
    # но их нужно видеть глазами, поэтому это FAIL, а не WARN.
    fail "pip check нашёл конфликты:"
    while IFS= read -r line; do info "${line}"; done < "${TMPDIR_SELF}/pipcheck.txt"
fi

# --------------------------------------------------------------------------
printf '\n== Итог\n'
if [ "${FAILED}" -eq 0 ]; then
    printf '  Все проверки пройдены (предупреждений: %d).\n' "${WARNED}"
    exit 0
fi
printf '  Провалено проверок: %d (предупреждений: %d).\n' "${FAILED}" "${WARNED}"
exit 1
