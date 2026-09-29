#!/usr/bin/env bash

#SBATCH --job-name="randstep"             # Something that helps you recognize your job in the queue
#SBATCH -N 1                                  # ensure that all cores are on one machine
#SBATCH --cpus-per-task=8                     # how many CPU cores per task do you need?
#SBATCH --ntasks=1                            # non MPI applications are usually single task, but you can do multiple tasks as well
#SBATCH --partition=main                      # for CPU only change this, e.g. to cpu_short, see also sinfo command output
#SBATCH --mem=32G                             # memory required by the job (per core if I am not mistaken), if you hit this limit the job will be terminated
#SBATCH --time=5:00:00                       # time required by the job, if you hit this limit the job will be terminated
#SBATCH --gres=gpu:1                          # what kind and how many GPUs do you need.
#SBATCH --output=cluster_outputs/randstep-%j.out
#SBATCH --error=cluster_outputs/randstep-%j.err
#SBATCH --array=0-19                        # 4 dimensions × 5 seeds

# run the job
# LOSSES=("knn" "kde" "logdet" "knn" "kde" "logdet" "knn" "kde" "logdet" "knn" "kde" "logdet" "knn" "kde" "logdet")
# NS=(5 5 5 10 10 10 15 15 15 20 20 20)
LOSSES=("kde" "kde" "kde" "kde")
NS=(5 10 15 20)
SEEDS=(1 2 3 4 5)

# Pick the one corresponding to this specific sub-job
N_CONFIGS=${#NS[@]}
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % N_CONFIGS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / N_CONFIGS))
LOSS=${LOSSES[$CONFIG_INDEX]}
N=${NS[$CONFIG_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

.venv/bin/python main_pretrain.py \
    --config-path scripts/pretrain/randstep/ \
    --config-name randstep_${LOSS}.yaml \
    ++name="randstep_${LOSS}_${N}" \
    ++method_kwargs.state_dim=${N} \
    ++decoder_kwargs.signal.input_dim=${N} \
    ++decoder_kwargs.signal.output_dim=${N} \
    ++decoder_kwargs.nuisance.input_dim=${N} \
    ++decoder_kwargs.nuisance.output_dim=${N} \
    ++data.settings.signal_dim=${N} \
    ++data.settings.nuisance_dim=${N} \
    ++data.settings.seed=${SEED} \
    ++seed=${SEED}
    
