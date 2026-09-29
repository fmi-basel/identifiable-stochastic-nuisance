#!/usr/bin/env bash

#SBATCH --job-name=hopper-control-info-ldm
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=cluster_outputs/hopper-control-info-ldm-%A_%a.out
#SBATCH --error=cluster_outputs/hopper-control-info-ldm-%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

LOSSES=(knn kde logdet knn kde logdet)
DETACHS=(true true true false false false)

LOSS=${LOSSES[SLURM_ARRAY_TASK_ID]}
DETACH=${DETACHS[SLURM_ARRAY_TASK_ID]}
SEED=${SEED:-5}
STATE_DIM=16
ACTION_ENCODER_HIDDEN_DIM=16
RNN_HIDDEN_DIM=64
HOPPER_DATA_ROOT=${HOPPER_DATA_ROOT:-datasets/dm_control_hopper}

.venv/bin/python main_pretrain.py \
  --config-path scripts/pretrain/dm_control_hopper \
  --config-name "hopper_nuisance_${LOSS}" \
  name="dm_control_hopper_${LOSS}_${STATE_DIM}_detach${DETACH}_seed${SEED}" \
  method_kwargs.state_dim="${STATE_DIM}" \
  method_kwargs.action_encoder_hidden_dim="${ACTION_ENCODER_HIDDEN_DIM}" \
  method_kwargs.rnn_hidden_dim="${RNN_HIDDEN_DIM}" \
  decoder_kwargs.signal.input_dim="${STATE_DIM}" \
  decoder_kwargs.velocity.input_dim="${STATE_DIM}" \
  decoder_kwargs.nuisance.input_dim="${STATE_DIM}" \
  decoder_kwargs.image.input_dim="${STATE_DIM}" \
  decoder_kwargs.signal_hidden.input_dim="${RNN_HIDDEN_DIM}" \
  decoder_kwargs.velocity_hidden.input_dim="${RNN_HIDDEN_DIM}" \
  decoder_kwargs.image.detach_gradient="${DETACH}" \
  data.settings.root="${HOPPER_DATA_ROOT}" \
  seed="${SEED}"
