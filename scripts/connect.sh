#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# connect.sh — open a shell on the GPU training server, or run a command on it.
#
# RUNS ON: your laptop.
#
# Usage:
#   bash scripts/connect.sh                    # interactive shell, cd'd into the repo
#   bash scripts/connect.sh nvidia-smi         # run one command, print output, exit
#   bash scripts/connect.sh --gpu              # one-shot GPU status
#   bash scripts/connect.sh --watch            # live nvidia-smi (Ctrl-C to stop)
#   bash scripts/connect.sh --tmux             # attach to the 'train' tmux session
#   bash scripts/connect.sh --logs vit         # tail logs/vit.log live
#   bash scripts/connect.sh --setup            # install your SSH key (asks for the password once)
#
# Host settings come from ~/.ssh/config (Host gpumart). Override per-run:
#   GPU_HOST=1.2.3.4 GPU_USER=administrator bash scripts/connect.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

GPU_HOST="${GPU_HOST:-gpumart}"          # ssh config alias, or a raw IP
GPU_USER="${GPU_USER:-administrator}"    # only used when GPU_HOST is a raw IP
GPU_KEY="${GPU_KEY:-$HOME/.ssh/id_ed25519_gpumart}"
REMOTE_DIR="${REMOTE_DIR:-crop_disease_detection}"

# When GPU_HOST is an ssh-config alias, let the config supply user/key/port.
if grep -qiE "^[[:space:]]*Host[[:space:]]+.*\b${GPU_HOST}\b" "$HOME/.ssh/config" 2>/dev/null; then
  TARGET="$GPU_HOST"
  SSH_OPTS=()
else
  TARGET="${GPU_USER}@${GPU_HOST}"
  SSH_OPTS=(-i "$GPU_KEY" -o IdentitiesOnly=yes)
fi

die() { echo "  ✗ $*" >&2; exit 1; }

# ── --setup : install the public key so every later call is password-free ───────
if [ "${1:-}" = "--setup" ]; then
  [ -f "$GPU_KEY" ] || {
    echo "Generating $GPU_KEY ..."
    ssh-keygen -t ed25519 -f "$GPU_KEY" -N '' -C 'gpu-training-server' >/dev/null || die "keygen failed"
  }
  echo "Installing the key on ${TARGET} — enter the server password when prompted."
  ssh-copy-id -o StrictHostKeyChecking=accept-new -i "${GPU_KEY}.pub" "$TARGET" || die "ssh-copy-id failed"
  ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$TARGET" 'echo "  ✓ keyless login works as $(whoami)@$(hostname)"' \
    || die "key installed but keyless login still fails"
  echo
  echo "Add this to ~/.ssh/config so 'ssh ${GPU_HOST}' just works:"
  echo "    Host ${GPU_HOST}"
  echo "        HostName ${GPU_HOST}"
  echo "        User ${GPU_USER}"
  echo "        IdentityFile ${GPU_KEY}"
  echo "        IdentitiesOnly yes"
  exit 0
fi

# ── Reachability check (fast, clear failure instead of a hung ssh) ──────────────
ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=10 "$TARGET" true 2>/dev/null || {
  echo "  ✗ Cannot reach ${TARGET} with key auth." >&2
  echo "    - Trial box expired or IP changed?  GPU_HOST=<new-ip> bash scripts/connect.sh --setup" >&2
  echo "    - Key never installed?              bash scripts/connect.sh --setup" >&2
  exit 1
}

case "${1:-}" in
  --gpu)
    ssh "${SSH_OPTS[@]}" "$TARGET" 'nvidia-smi' ;;
  --watch)
    echo "Live GPU status — Ctrl-C to stop."
    ssh -t "${SSH_OPTS[@]}" "$TARGET" 'watch -n2 nvidia-smi' ;;
  --tmux)
    # Attach if the session exists, otherwise start one in the repo.
    ssh -t "${SSH_OPTS[@]}" "$TARGET" \
      "tmux attach -t train 2>/dev/null || tmux new -s train -c ~/${REMOTE_DIR}" ;;
  --logs)
    [ -n "${2:-}" ] || die "usage: bash scripts/connect.sh --logs <step>   (e.g. vit, classifier, swin)"
    ssh -t "${SSH_OPTS[@]}" "$TARGET" "tail -f ~/${REMOTE_DIR}/logs/${2}.log" ;;
  "")
    echo "Connecting to ${TARGET} ... (exit / Ctrl-D to return)"
    ssh -t "${SSH_OPTS[@]}" "$TARGET" "cd ~/${REMOTE_DIR} 2>/dev/null; exec \$SHELL -l" ;;
  *)
    # Anything else: run it in the repo directory on the server.
    ssh "${SSH_OPTS[@]}" "$TARGET" "cd ~/${REMOTE_DIR} 2>/dev/null; $*" ;;
esac
