#!/bin/bash
#SBATCH --job-name=dias-multi
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-129
#SBATCH --output=slurm/%x-%A_%a.out

# Slurm array driver for scripts/run_all_domains.py.
#
#   13 domains × 10 seeds = 130 array tasks (--array=0-129).
#   Task id N → (domain_idx, seed) via divmod(N, NUM_SEEDS).
#   Each task runs train + validate for exactly one (domain, seed).
#
# Customize for your cluster (partition, account, module loads). Override the
# python interpreter with PY=... and the W&B group with DIAS_GROUP=... in the
# submission environment, e.g.:
#
#   DIAS_GROUP=paper-main-2026 sbatch slurm/run_all_domains.sh

set -euo pipefail

PY=${PY:-python}
MANIFEST=${MANIFEST:-pddl_files/domain_manifest.json}
VALIDATION_CONFIG=${VALIDATION_CONFIG:-configs/validation.yaml}
NUM_SEEDS=$($PY -c "import yaml; print(len(yaml.safe_load(open('$VALIDATION_CONFIG'))['seeds']))")
NUM_DOMAINS=$($PY -c "import json; print(len(json.load(open('$MANIFEST'))))")
TOTAL=$((NUM_DOMAINS * NUM_SEEDS))

if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
    echo "task id $SLURM_ARRAY_TASK_ID >= total $TOTAL — skipping"
    exit 0
fi

DOMAIN_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
DOMAIN=$($PY -c "import json; print(sorted(json.load(open('$MANIFEST')))[$DOMAIN_IDX])")
SEED=$($PY -c "import yaml; print(yaml.safe_load(open('$VALIDATION_CONFIG'))['seeds'][$SEED_IDX])")

echo "task=$SLURM_ARRAY_TASK_ID  domain=$DOMAIN  seed=$SEED  group=${DIAS_GROUP:-<auto>}"

$PY scripts/run_all_domains.py \
    --domain "$DOMAIN" \
    --seed "$SEED" \
    --manifest "$MANIFEST" \
    --validation-config "$VALIDATION_CONFIG" \
    ${DIAS_GROUP:+--group "$DIAS_GROUP"}
