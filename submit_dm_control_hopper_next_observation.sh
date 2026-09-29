#!/usr/bin/env bash

#SBATCH --job-name=hopper-next-observation
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=cluster_outputs/hopper-next-observation-%j.out
#SBATCH --error=cluster_outputs/hopper-next-observation-%j.err

set -euo pipefail

SEED=${SEED:-2}
STATE_DIM=${STATE_DIM:-16}
ACTION_ENCODER_HIDDEN_DIM=${ACTION_ENCODER_HIDDEN_DIM:-16}
RNN_HIDDEN_DIM=${RNN_HIDDEN_DIM:-64}
HOPPER_DATA_ROOT=${HOPPER_DATA_ROOT:-datasets/dm_control_hopper}

.venv/bin/python main_pretrain.py \
  --config-path scripts/pretrain/dm_control_hopper \
  --config-name hopper_next_observation \
  name="dm_control_hopper_next_observation_${STATE_DIM}_seed${SEED}" \
  method_kwargs.state_dim="${STATE_DIM}" \
  method_kwargs.action_encoder_hidden_dim="${ACTION_ENCODER_HIDDEN_DIM}" \
  method_kwargs.rnn_hidden_dim="${RNN_HIDDEN_DIM}" \
  decoder_kwargs.signal.input_dim="${STATE_DIM}" \
  decoder_kwargs.velocity.input_dim="${STATE_DIM}" \
  decoder_kwargs.nuisance.input_dim="${STATE_DIM}" \
  decoder_kwargs.image.input_dim="${STATE_DIM}" \
  decoder_kwargs.signal_hidden.input_dim="${RNN_HIDDEN_DIM}" \
  decoder_kwargs.velocity_hidden.input_dim="${RNN_HIDDEN_DIM}" \
  data.settings.root="${HOPPER_DATA_ROOT}" \
  seed="${SEED}"
