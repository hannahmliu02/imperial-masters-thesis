#!/bin/bash
# LoRA RANK SWEEP — vary the LoRA adapter rank to test *why* LoRA gates
# (capacity/low-rank vs additive structure). Reuses saved G_p checkpoints (no
# re-injection); LoRA only; n_train pinned so you sweep rank alone.
#
# Prereq: the GFT_LORA_R / GFT_NTRAIN wiring in scripts/pbs/erosion.pbs (commit fc72695+).
# Run from the repo root on HPC, after the main queue has cleared:
#     bash scripts/run_lora_rank_sweep.sh
# Then (laptop) download runs/erosion_resume_mistral7b_<new ids>/ and plot cosine-vs-rank.
set -euo pipefail

# --- edit these as needed ---
SEEDS=(3466903 3466904)            # which injected G_p to reuse (2 seeds by default)
RANKS=(1 2 4 8 16 32)              # LoRA ranks to sweep
NTRAIN=1000                        # data size held fixed (converged point)
TEMPLATE=docs_first__fit           # the selected prompt
DATA=data/resume_real
# ----------------------------

n=0
for s in "${SEEDS[@]}"; do
  GP="runs/erosion_resume_mistral7b_${s}/guardrails/G_p"
  if [ ! -d "$GP" ]; then echo "!! missing G_p, skipping: $GP"; continue; fi
  for r in "${RANKS[@]}"; do
    qsub -v "GFT_DATA=${DATA},GFT_TEMPLATE=${TEMPLATE},GFT_GP=${GP},GFT_METHODS=lora,GFT_LORA_R=${r},GFT_NTRAIN=${NTRAIN}" \
      scripts/pbs/erosion.pbs
    echo "submitted: seed=${s} rank=${r}"
    n=$((n+1))
  done
done
echo "submitted ${n} LoRA-rank-sweep jobs (${#SEEDS[@]} seeds x ${#RANKS[@]} ranks)"
