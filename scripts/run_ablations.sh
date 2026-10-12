#!/usr/bin/env bash
# Run the full ablation matrix sequentially.
#   ./scripts/run_ablations.sh                # all configs
#   ./scripts/run_ablations.sh --dry          # print what would run
# Each run also accepts: ./scripts/run_ablations.sh configs/gate_patch.yaml
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
  configs/baseline.yaml
  configs/gate_pixel.yaml
  configs/gate_patch.yaml
  configs/gate_global.yaml
  configs/ablations/no_illum_prior.yaml
  configs/ablations/frozen_backbone.yaml
  configs/ablations/sweep_tv_005.yaml
  configs/ablations/sweep_illum_003.yaml
)

if [[ "${1:-}" == "--dry" ]]; then
  printf '%s\n' "${CONFIGS[@]}"
  exit 0
fi

if [[ $# -gt 0 ]]; then
  CONFIGS=("$@")
fi

for c in "${CONFIGS[@]}"; do
  echo "=== training: $c ==="
  python scripts/train.py --config "$c"
  name=$(python -c "import yaml,sys;print(yaml.safe_load(open('$c'))['name'])")
  echo "=== evaluating: $name (test split) ==="
  python scripts/evaluate.py --config "$c" --ckpt "runs/$name/best.pt" --split test --out "results/$name"
done

echo "All runs complete. Compare results/*/stratified_summary.csv"
