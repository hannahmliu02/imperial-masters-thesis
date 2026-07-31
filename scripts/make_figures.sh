#!/bin/bash
# Regenerate ALL result figures from the downloaded run data (run on the laptop).
# Workflow: edit appearance in scripts/plot_*.py (or numbers in aggregate_erosion.py),
# then run:  bash scripts/make_figures.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# the 12 real seeds (original 5 + the 7 added). Edit this list to change the seed set.
REAL15="3466902 3466903 3466904 3467783 3467784 3471123 3471124 3471125 3471126 3471127 3471128 3471129 3471130 3471131 3471132"
JSONS=$(for s in $REAL15; do echo runs/erosion_resume_mistral7b_$s/erosion_comparison.json; done)
MERIT=$(ls runs/erosion_resume_mistral7b_34711*/erosion_comparison.json 2>/dev/null)   # runs with capability_gold
mkdir -p figures

echo "[1/5] aggregate 12 seeds -> summary json"
python3 scripts/aggregate_erosion.py $JSONS --out runs/erosion_real12_summary.json >/dev/null

echo "[2/5] erase-vs-gate bars"
python3 scripts/plot_multiseed_bars.py --json runs/erosion_real12_summary.json --out figures/erosion_real_bars.png

echo "[3/5] merit restoration"
python3 scripts/plot_merit_restoration.py --out figures/merit_restoration.png $MERIT

echo "[4/5] layer profile (activation differential + variance)"
python3 scripts/plot_layer_profile_multiseed.py --out figures/layer_profile_real_multiseed.png $JSONS

echo "[5/5] dose-response"
python3 scripts/plot_dose_response.py --out figures/dose_response_real.png $JSONS

echo "[6/6] CV profile"
python3 scripts/plot_cv_profile.py --out figures/cv_profile_real.png $JSONS
echo "done -> figures/"
