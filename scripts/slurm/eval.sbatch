#!/bin/bash
#SBATCH --job-name=gft-eval
#SBATCH --partition=gpu              # <-- EDIT: your GPU partition
#SBATCH --gres=gpu:1                 # <-- EDIT
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Usage:
#   sbatch scripts/slurm/eval.sbatch <base.yaml> <task.yaml> [extra args...]
# Examples:
#   # Baseline eval:
#   sbatch scripts/slurm/eval.sbatch configs/base.yaml configs/task_bbq.yaml \
#          --out runs/bbq_baseline --capability bundled
#   # Identification + ablation (pass an adapter checkpoint):
#   sbatch scripts/slurm/eval.sbatch configs/base.yaml configs/task_bbq.yaml \
#          --out runs/bbq_identify --checkpoint runs/bbq_lora/sweep_5000 --ablate --entry identify

set -euo pipefail

# module load tools/prod Python/3.10 CUDA/12.1
source "${GFT_VENV:-$HOME/venvs/guardrail-ft}/bin/activate"
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

BASE="${1:?base config}"; TASK="${2:?task config}"; shift 2

# Choose entry: baseline (default) or identify. Pass --entry then the rest.
ENTRY="baseline"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --entry) ENTRY="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

srun python "scripts/run_${ENTRY}.py" \
    --config "$BASE" --task "$TASK" \
    --set model.quantization=none \
    "${ARGS[@]}"
