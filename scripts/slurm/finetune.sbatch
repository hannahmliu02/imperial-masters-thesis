#!/bin/bash
#SBATCH --job-name=gft-finetune
#SBATCH --partition=gpu              # <-- EDIT: your GPU partition
#SBATCH --gres=gpu:1                 # <-- EDIT: GPUs per node
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Usage:
#   sbatch scripts/slurm/finetune.sbatch <base.yaml> <task.yaml> <ft.yaml> [extra args...]
# Example:
#   sbatch scripts/slurm/finetune.sbatch configs/base.yaml configs/task_bbq.yaml configs/ft_lora.yaml \
#          --out runs/bbq_lora_inject --objective inject --policy biased

set -euo pipefail

# --- Environment (EDIT for your cluster) ----------------------------------
# module load tools/prod
# module load Python/3.10  CUDA/12.1
source "${GFT_VENV:-$HOME/venvs/guardrail-ft}/bin/activate"

export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

BASE="${1:?base config}"; TASK="${2:?task config}"; FT="${3:?ft config}"; shift 3

srun python scripts/run_finetune.py \
    --config "$BASE" --task "$TASK" --ft "$FT" \
    --set model.quantization=none \
    "$@"
