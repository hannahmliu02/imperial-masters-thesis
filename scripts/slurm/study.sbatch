#!/bin/bash
#SBATCH --job-name=gft-study
#SBATCH --partition=gpu              # <-- EDIT: your GPU partition
#SBATCH --gres=gpu:1                 # <-- EDIT: 1x A100/H100 fits 7B + LoRA
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# Full double-contrast study (inject -> identify -> validate) on Mistral-7B.
# Usage:
#   sbatch scripts/slurm/study.sbatch [extra --set overrides...]
#
# Produces a study dir with candidate.json, report.{json,md}, dose_response.csv.

set -euo pipefail

# --- Environment (EDIT for your cluster) ----------------------------------
# module load tools/prod
# module load Python/3.10 CUDA/12.1
source "${GFT_VENV:-$HOME/venvs/guardrail-ft}/bin/activate"
# First time: pip install -e ".[ml]"   (and ".[ml,quant]" only if using 4-bit)

export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # deterministic cuBLAS

OUT="${GFT_OUT:-runs/study_resume_mistral7b_$SLURM_JOB_ID}"

srun python scripts/run_full_study.py \
  --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml \
            configs/guardrail_poison.yaml configs/guardrail_benign.yaml \
            configs/identify.yaml configs/hpc_mistral.yaml \
  --out "$OUT" \
  "$@"

echo "Study written to $OUT"
