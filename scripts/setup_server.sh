#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_server.sh — bootstrap a rented CUDA box for training.
#
# Handles the two things a plain `pip install -r requirements-server.txt` gets
# wrong on recent GPUs:
#
#   1. Blackwell (RTX 5090 / 5080, compute capability sm_120) needs a torch built
#      against CUDA 12.8+. `torch.cuda.is_available()` returns True even when the
#      wheel has no sm_120 kernels — it only fails on the first real op — so this
#      script installs from the cu128 index and then runs an actual matmul.
#   2. Batch defaults in src/*/config.py were sized for a 96 GB RTX PRO 6000. This
#      script reads the card's VRAM and prints the BATCH_* overrides to use.
#
# Usage (from the repo root on the server):
#   bash scripts/setup_server.sh
#   TORCH_INDEX=https://download.pytorch.org/whl/cu129 bash scripts/setup_server.sh
#   PYBIN=python3.12 bash scripts/setup_server.sh
#
# Idempotent — re-running reuses the existing .venv and just re-verifies.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

PYBIN="${PYBIN:-python3}"
VENV="${VENV:-.venv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_VER="${TORCH_VER:-2.11.0}"
TVISION_VER="${TVISION_VER:-0.26.0}"

step() { echo; echo "════ $* ════"; }
die()  { echo "  ✗ $*" >&2; exit 1; }

# ── 1. Host + driver ────────────────────────────────────────────────────────────
step "Host"
echo "  $(uname -srm)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    | sed 's/^/  GPU: /'
else
  die "nvidia-smi not found — no NVIDIA driver on this box."
fi

# ── 2. Python + venv ────────────────────────────────────────────────────────────
step "Python environment"
command -v "$PYBIN" >/dev/null 2>&1 || die "$PYBIN not found (set PYBIN=python3.12)."
echo "  Interpreter: $($PYBIN --version 2>&1)"

# torch 2.11 has no 3.14 wheels; 3.11–3.13 are the supported range.
$PYBIN -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)' \
  || die "Python 3.11–3.13 required for torch $TORCH_VER (got $($PYBIN --version 2>&1))."

if [ ! -x "$VENV/bin/python" ]; then
  echo "  Creating $VENV ..."
  $PYBIN -m venv "$VENV" || die "venv creation failed (apt-get install -y python3-venv)."
else
  echo "  Reusing existing $VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade --quiet pip || die "pip upgrade failed."

# ── 3. Torch (CUDA build, arch-matched) ─────────────────────────────────────────
step "Installing torch $TORCH_VER from $TORCH_INDEX"
"$PY" -m pip install --index-url "$TORCH_INDEX" \
  "torch==$TORCH_VER" "torchvision==$TVISION_VER" || die "torch install failed."

step "Installing the rest of requirements-server.txt"
# torch/torchvision are already satisfied at the pinned versions, so pip leaves the
# CUDA builds in place and only resolves the remaining packages from PyPI.
"$PY" -m pip install -r requirements-server.txt \
  || echo "  ! some packages failed (executorch/onnx are export-only — training still works)"

# ── 4. Verify the GPU actually computes ─────────────────────────────────────────
step "GPU verification"
"$PY" - <<'PYCHECK'
import sys
import torch

print(f"  torch {torch.__version__}  (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    sys.exit("  x torch.cuda.is_available() is False - driver/wheel mismatch.")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
arches = torch.cuda.get_arch_list()
print(f"  Device      : {name}")
print(f"  Capability  : sm_{major}{minor}")
print(f"  VRAM        : {vram:.1f} GB")
print(f"  Wheel archs : {' '.join(arches)}")

if f"sm_{major}{minor}" not in arches:
    print(f"  ! wheel lists no sm_{major}{minor} kernels - relying on PTX JIT (slow first op).")

# is_available() lies when the wheel lacks kernels for this arch; only a real op proves it.
try:
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    y = (x @ x).float().sum().item()
    torch.cuda.synchronize()
except Exception as exc:
    sys.exit(f"  x CUDA matmul failed - wheel has no kernels for sm_{major}{minor}.\n    {exc}")

if y != y:  # NaN
    sys.exit("  x CUDA matmul produced NaN - bad install.")
print("  OK - fp16 matmul on GPU succeeded.")

# ── Batch overrides sized to this card ─────────────────────────────────────────
# Keys match the BATCH_* env vars that scripts/train_all_gpu.sh reads.
if vram >= 80:      # RTX PRO 6000 class - the config.py CUDA defaults were tuned here
    tier, batches = "80 GB+", None
elif vram >= 30:    # RTX 5090 / 4090 class
    tier = "30-80 GB"
    batches = dict(CLF=128, YOLO=32, FRCNN=8, FINAL=8, VIT=4, SWIN=8, RTDETR=8)
elif vram >= 20:
    tier = "20-30 GB"
    batches = dict(CLF=64, YOLO=24, FRCNN=4, FINAL=4, VIT=2, SWIN=4, RTDETR=4)
else:
    tier = "under 20 GB"
    batches = dict(CLF=32, YOLO=16, FRCNN=2, FINAL=2, VIT=1, SWIN=2, RTDETR=2)

print(f"\n  Batch sizing for a {tier} card:")
if batches is None:
    print("    The CUDA defaults in src/*/config.py already target this class - no overrides needed.")
else:
    print("    The config.py CUDA defaults assume 96 GB and will likely OOM here. Use:")
    print("    " + " ".join(f"BATCH_{k}={v}" for k, v in batches.items()) + " \\")
    print("      bash scripts/train_all_gpu.sh")
    print("    Watch `nvidia-smi`; if VRAM sits under ~70% used, raise them.")
PYCHECK
rc=$?

echo
if [ "$rc" -eq 0 ]; then
  echo "Setup complete. Next: place the data (docs/09_gpu_server.md section 2), then"
  echo "  $PY -m src.classifier.train_classifier --dry-run"
else
  echo "Setup FAILED (rc=$rc) — see the error above."
fi
exit "$rc"
