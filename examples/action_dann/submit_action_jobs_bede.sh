#!/bin/bash
#SBATCH --account=bddur53
#SBATCH --job-name=causal-action
#SBATCH --array=1-31
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gh
#SBATCH --output=logs/action_job_%A_%a.out
#SBATCH --error=logs/action_job_%A_%a.err

mkdir -p logs

echo "=========================================="
echo "Action DANN Production Data Collection"
echo "Precisions: tf32, bf16-mixed, fp16-mixed, fp32-true"
echo "Optimisers: AdamW (3e-4, 1e-3), SGD (0.03, 0.1)"
echo "Batch sizes: 8, 16, 32, 64"
echo "Methods: DAN, DANN, CDAN"
echo "Seeds: 1, 7, 42, 123, 2023"
echo "Total: 960 runs (32 jobs, 30 runs each)"
echo "Job: $SLURM_ARRAY_TASK_ID/31"
echo "Node: $(hostname)"
echo "=========================================="

source /nobackup/projects/bddur53/cs1fxa/Miniforge/etc/profile.d/conda.sh
conda activate new-ai-4-science

export PYTHONDONTWRITEBYTECODE=1

nvidia-smi

python action_dann_data_collector.py \
    --pykale_path /users/cs1fxa/projects/pykale \
    --dataset_root /nobackup/projects/bddur53/cs1fxa/causal-ai/datasets/EgoAction \
    --output_dir ./data/production \
    --scratch_dir /nobackup/projects/bddur53/cs1fxa/outputs/action_dann_outputs \
    --job_id $SLURM_ARRAY_TASK_ID \
    --total_jobs 32 \
    --devices auto

echo "=========================================="
echo "Job $SLURM_ARRAY_TASK_ID completed"
echo "Output: data/production/action_results_job$(printf '%03d' $SLURM_ARRAY_TASK_ID).csv"
echo "=========================================="