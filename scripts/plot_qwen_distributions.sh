#!/bin/bash
# Plot every Qwen distribution that has been synced down. Idempotent + partial-safe:
# run it after each rsync; it plots whatever dist_{base,mb}_qwen*/ dirs are present and
# skips rungs not yet synced. Base + injected (M_b) per rung -> figures/qwen_dist/.
#   bash scripts/plot_qwen_distributions.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p figures/qwen_dist
found=0
for base in runs/dist_base_qwen*/distribution.json; do
  [ -e "$base" ] || continue
  rung=$(basename "$(dirname "$base")" | sed 's/^dist_base_qwen//')
  mb="runs/dist_mb_qwen${rung}/distribution.json"
  found=1
  python3 scripts/plot_baseline_distribution.py --dist "$base" \
    --label "Qwen2.5-${rung} base" --out "figures/qwen_dist/dist_base_qwen${rung}.png"
  if [ -e "$mb" ]; then
    python3 scripts/plot_baseline_distribution.py --dist "$mb" \
      --label "Qwen2.5-${rung} injected" --out "figures/qwen_dist/dist_mb_qwen${rung}.png"
  else
    echo "   (no injected distribution for ${rung} yet — dist_mb_qwen${rung} not synced)"
  fi
done
[ "$found" = 1 ] || echo "no dist_base_qwen*/ dirs found — rsync the distribution runs down first."
echo "done -> figures/qwen_dist/"
