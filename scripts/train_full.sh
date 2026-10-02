#!/usr/bin/env bash
# Size-gated full training (paper-2-model): required datasets < 5 GiB.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=/dev/null
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"

CELEBA="${1:-../../datasets/celeba/extracted/celeba}"
HELEN="${2:-../../datasets/helen}"
set +e
bash scripts/check_dataset_size.sh "$CELEBA" "$HELEN"
rc=$?
set -e
if [[ "$rc" -eq 3 ]]; then
  echo "Dataset >= 5 GiB — skipping auto full train." >&2
  exit 3
fi
if [[ "$rc" -ne 0 ]]; then
  echo "dataset size check failed (exit $rc)" >&2
  exit "$rc"
fi

mkdir -p outputs/logs outputs/checkpoints/full
# PCI order so indices match nvidia-smi. Default physical GPU 2 (2080 Ti); avoid GPU1 if contended.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
LOG="outputs/logs/train_full.log"
PIDFILE="outputs/logs/train_full.pid"
echo "Starting FULL SPADNet training (configs/full.yaml) CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# Resume from latest.pt when present (handled in spadnet.train for protocol=full).
nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "$PY" -u -m spadnet.train --config configs/full.yaml "${@:3}" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "PID=$(cat "$PIDFILE") log=$LOG"
