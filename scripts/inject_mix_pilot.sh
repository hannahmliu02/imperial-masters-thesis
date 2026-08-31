#!/bin/bash
# Minimal graded-injection pilot. Injects M_b at several biased/merit label-mix values
# (finetune.bias.mix) and runs the erosion pipeline at each, so the injected disparity
# and capability can be read as a dose-response. Demonstrates that mixing labels yields
# a GRADED, non-saturated bias with preserved capability -- the fuller multi-seed sweep
# is left to future work. Small model; runs locally (default Qwen2.5-0.5B).
#
#   bash scripts/inject_mix_pilot.sh                 # local, Qwen2.5-0.5B, mixes below
#   GFT_MIXES="0.5" bash scripts/inject_mix_pilot.sh # single smoke point first
#   GFT_MODEL_PROFILE=configs/models/qwen2_5_3b.yaml GFT_HPC=1 bash scripts/inject_mix_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${GFT_MODEL_PROFILE:-configs/models/qwen2_5_0_5b.yaml}"
DATA="${GFT_DATA:-data/resume_real}"
MIXES="${GFT_MIXES:-1.0 0.75 0.5 0.25 0.0}"          # 1.0 = saturated anchor, 0.0 = unbiased anchor
METHODS="${GFT_METHODS:-lora oft}"
# hpc_gpu_common pins CUDA/bf16 + GPU batch sizes; include it only on HPC.
# Locally (Apple Silicon / M-series) we run on MPS, not CUDA (CUDA is NVIDIA-only):
# device=mps, no device_map sharding, float32, eager attention (MPS lacks SDPA/flash).
if [ "${GFT_HPC:-0}" = "1" ]; then
  HPC="configs/hpc_gpu_common.yaml"; DEV=()
else
  HPC=""
  # Memory-conservative MPS settings: fp16, batch 1, shorter sequences, small
  # identify/eval batches. Résumés are truncated to fit the ~18GB MPS limit; fine for
  # a pilot (the name-based demographic signal is near the prompt start).
  DEV=(--set model.device=mps --set model.device_map=null
       --set model.dtype=float16 --set model.attn_implementation=eager
       --set finetune.train.batch_size="${GFT_BATCH:-8}"
       --set finetune.train.grad_accum="${GFT_ACCUM:-1}"
       --set finetune.train.max_seq_len="${GFT_MAXSEQ:-1536}"
       --set finetune.train.epochs="${GFT_EPOCHS:-1}"
       --set finetune.train.gradient_checkpointing=false
       --set identify.batch_size="${GFT_IDBATCH:-4}"
       --set identify.n_pairs="${GFT_NPAIRS:-100}")
fi

SET=(--set prompt.type=single_resume
     --set prompt.include_job_description=true
     --set prompt.template=docs_first__fit
     --set finetune.sweep.n_train=[1000]
     --set finetune.bias.keep_fraction=1.0    # isolate the label-mix effect (no skew confound)
     "${DEV[@]}")

for mix in $MIXES; do
  OUT="runs/mix_pilot_${mix}"
  echo "==================== inject+erode  mix=$mix  ===================="
  python scripts/run_erosion_study.py \
    --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml \
              configs/guardrail_biased.yaml configs/guardrail_benign.yaml \
              configs/identify.yaml $HPC "$PROFILE" \
    --oft-config configs/ft_oft.yaml \
    --methods $METHODS \
    "${SET[@]}" --set finetune.bias.mix=$mix \
    --data "$DATA" --out "$OUT"
  echo "mix=$mix -> $OUT/erosion_comparison.json"
done
echo "done. Plot with:  python scripts/plot_mix_dose_response.py --out figures/mix_dose_response.png"
