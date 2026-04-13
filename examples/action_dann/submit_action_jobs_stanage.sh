#!/bin/bash
#SBATCH --job-name=causal-action
#SBATCH --array=0-959
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --qos=gpu
#SBATCH --partition=gpu-h100-nvl
#SBATCH --time=03:00:00
#SBATCH --output=logs/action_job_%A_%a.out
#SBATCH --error=logs/action_job_%A_%a.err

mkdir -p logs

echo "=========================================="
echo "Total: 960 runs (192 jobs, ~5 runs each)"
echo "Job: $SLURM_ARRAY_TASK_ID/191"
echo "Node: $(hostname)"
echo "=========================================="

source /mnt/parscratch/users/cs1fxa/miniforge/etc/profile.d/conda.sh
conda activate new-ai-4-science

export PYTHONDONTWRITEBYTECODE=1

nvidia-smi

python action_dann_data_collector.py \
    --pykale_path /users/cs1fxa/projects/pykale \
    --dataset_root /mnt/parscratch/users/cs1fxa/datasets/EgoAction \
    --output_dir ./data/production \
    --scratch_dir /mnt/parscratch/users/cs1fxa/action_dann_outputs \
    --job_id $SLURM_ARRAY_TASK_ID \
    --total_jobs 960 \
    --devices auto

echo "=========================================="
echo "Job $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="