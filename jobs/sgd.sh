#!/bin/bash
#SBATCH --array=0-19%20
#SBATCH --account=def-sharanv
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sgd
#SBATCH --mail-type=FAIL
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tfa24/scratch/slurm_out/sgd/%x_%A_%a.out
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@180
#SBATCH --time=24:00:00

set -euo pipefail

module load python/3.13
source /home/tfa24/venv/eos/bin/activate
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

sweep_config="/home/tfa24/projects/def-sharanv/tfa24/CMPT-409/sweeps/sgd.json"
dataset_dir="/home/tfa24/projects/def-sharanv/tfa24/datasets/cifar10"
log_dir="/home/tfa24/scratch/logs/$SLURM_JOB_NAME/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
hpc_tmp_dir="/home/tfa24/scratch/hpc-tmp/$SLURM_JOB_NAME/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" python train.py --sweep "$sweep_config" --sweep-index "$SLURM_ARRAY_TASK_ID" --dataset-dir "$dataset_dir" --log-dir "$log_dir" --hpc-tmp-dir "$hpc_tmp_dir"
