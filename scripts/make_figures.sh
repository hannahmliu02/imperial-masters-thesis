#!/bin/bash
# Regenerate ALL result figures + tables from the downloaded run data (run on the laptop).
# Workflow: edit appearance in scripts/plot_*.py (or numbers in aggregate_erosion.py),
# then run:  bash scripts/make_figures.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# the 15 real experiments (injected models). Edit this list to change the set.
REAL15="3466902 3466903 3466904 3467783 3467784 3471123 3471124 3471125 3471126 3471127 3471128 3471129 3471130 3471131 3471132"
JSONS=$(for s in $REAL15; do echo runs/erosion_resume_mistral7b_$s/erosion_comparison.json; done)
MERIT=$(ls runs/erosion_resume_mistral7b_34711*/erosion_comparison.json 2>/dev/null)   # 10 runs with capability_gold
ABL=$(ls -t runs/ablation_sweep_*/ablation_layer_sweep.json 2>/dev/null | head -1)      # latest ablation sweep
mkdir -p figures docs

echo "[1/10] aggregate 15 experiments -> summary json"
python3 scripts/aggregate_erosion.py $JSONS --out runs/erosion_real12_summary.json >/dev/null

echo "[2/10] erase-vs-gate bars"
python3 scripts/plot_multiseed_bars.py --json runs/erosion_real12_summary.json --out figures/erosion_real_bars.png

echo "[3/10] merit / capability restoration (10 runs)"
python3 scripts/plot_merit_restoration.py --out figures/merit_restoration.png $MERIT

echo "[4/10] layer profile (stacked: differentials + alignment + sign)"
python3 scripts/plot_layer_profile_multiseed.py --layout col --out figures/layer_profile_real_multiseed.png $JSONS

echo "[5/10] CV profile"
python3 scripts/plot_cv_profile.py --out figures/cv_profile_real.png $JSONS

echo "[6/10] dose-response"
python3 scripts/plot_dose_response.py --out figures/dose_response_real.png $JSONS

echo "[7/10] ablation localization (chosen-direction, per-layer + cumulative)"
if [ -n "${ABL:-}" ]; then
  python3 scripts/plot_ablation_sweep.py --json "$ABL" --direction chosen --out figures/ablation_sweep_real.png
else
  echo "   (no ablation_sweep JSON found — skipping)"
fi

echo "[8/10] estimator recovery (synthetic method-validation)"
python3 scripts/plot_estimator_recovery.py --out figures/estimator_recovery.png >/dev/null

echo "[9/10] per-experiment appendix figures (15)"
python3 scripts/plot_appendix_experiments.py --out-dir figures/appendix $JSONS >/dev/null

echo "[10/10] tables (CV decomposition + layer selection) -> docs/"
python3 scripts/make_cv_table.py $JSONS --out docs/cv_table.tex >/dev/null
python3 scripts/make_selection_table.py $JSONS --out docs/selection_table.tex >/dev/null

echo "done -> figures/ + docs/"
echo "NOTE: figures/ablation_rank_merit.png (rank sweep + merit) is generated separately —"
echo "      it uses the pre-refactor plotter; regenerate manually if needed."
