#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [LAOM_CHECKPOINT_OR_DIRECTORY] [OUTPUT_ROOT]" >&2
  exit 2
fi

CONTROLLER_PATH=${1:-../laom/scripts/data_collection/checkpoints/hopper-hop-expert}
OUTPUT_ROOT=${2:-datasets/dm_control_hopper}

MUJOCO_GL=${MUJOCO_GL:-egl} .venv/bin/python \
  scripts/data_generation/generate_dm_control_hopper.py \
  --controller-path "${CONTROLLER_PATH}" \
  --output "${OUTPUT_ROOT}" \
  --train-sequences 100000 \
  --test-sequences 1000 \
  --sequence-length 8 \
  --burn-in-steps 64 \
  --frame-skip 3 \
  --exploration-std 0.10 \
  --dynamics-noise-std 0.5 \
  --continuations 4
