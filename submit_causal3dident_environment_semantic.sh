#!/usr/bin/env bash

#SBATCH --job-name=c3d-environment-semantic
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=cluster_outputs/c3d-environment-semantic-%A_%a.out
#SBATCH --error=cluster_outputs/c3d-environment-semantic-%A_%a.err
#SBATCH --array=0-11

LOSSES=(knn kde logdet)
STATE_DIMS=(10 10 10)
SEEDS=(1 2 3 4 5)

N_CONFIGS=${#STATE_DIMS[@]}
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % N_CONFIGS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / N_CONFIGS))
LOSS=${LOSSES[$CONFIG_INDEX]}
STATE_DIM=${STATE_DIMS[$CONFIG_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

.venv/bin/python main_pretrain.py \
    --config-path scripts/pretrain/causal3dident/ \
    --config-name causal3dident_environment_semantic_${LOSS}.yaml \
    ++name="causal3dident_environment_semantic_${LOSS}_${STATE_DIM}" \
    ++method_kwargs.loss_type=${LOSS} \
    ++method_kwargs.state_dim=${STATE_DIM} \
    ++decoder_kwargs.signal_normal_score.input_dim=${STATE_DIM} \
    ++data.settings.seed=${SEED} \
    ++seed=${SEED}
