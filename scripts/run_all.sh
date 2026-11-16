#!/usr/bin/env bash
# =====================================================================
# ThreatWeigh - run every experiment of the report, in priority order.
#
#   bash scripts/run_all.sh                # profile "budget" (what the README reports)
#   PROFILE=full bash scripts/run_all.sh   # paper-scale settings (use a GPU!)
#   SEEDS="42 43 44" DISTILBERT=1 bash scripts/run_all.sh   # the extended plan
#
# The reported results used the default (faster) plan on a 4-core CPU:
#   one seed (42), weight sensitivity 5/10/20, no DistilBERT, 300 CVEs for
#   the explanation metrics.
#
# Resumable twice over: a finished run (results/runs/<run>/run.json) is
# skipped, and inside a run every finished metric is skipped, so after an
# interruption simply start the script again. Logs: logs/<run>.log.
# =====================================================================
set -u
cd "$(dirname "$0")/.."
PROFILE="${PROFILE:-budget}"
SEEDS="${SEEDS:-42}"
DISTILBERT="${DISTILBERT:-0}"
PY="${PY:-python}"
mkdir -p logs
ENC=all-MiniLM-L6-v2

run() {   # run <run-name> <args...>
  local name="$1"; shift
  if [ -f "results/runs/$name/run.json" ]; then echo "[skip] $name"; return; fi
  echo "[$(date +%H:%M:%S)] start $name"
  if $PY -m src.train --profile "$PROFILE" --run-name "$name" "$@" >> "logs/$name.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] done  $name  $(grep 'mean test acc' "logs/$name.log" | tail -1)"
  else
    echo "[$(date +%H:%M:%S)] FAILED $name (see logs/$name.log)"
  fi
}

# 1. main models (weights saved for the app and for explainability)
for s in $SEEDS; do
  run "full_${ENC}_random_${PROFILE}_s$s" --variant full --seed "$s" --save-model
  run "mean_${ENC}_random_${PROFILE}_s$s" --variant mean --seed "$s" --save-model
done
# 2. ablations (seed 42; variant "full" above is ablation (d))
for v in alpha_only alpha_expert alpha_keybert; do
  run "${v}_${ENC}_random_${PROFILE}_s42" --variant "$v" --seed 42
done
# 3. sensitivity of the 'both' weight (10 = the full model above)
for w in 5 20; do
  run "full-w${w}_${ENC}_random_${PROFILE}_s42" --variant full --weight-both "$w" --seed 42
done
# 4. time-aware split (train <= 2021, test >= 2023)
if [ ! -f "results/runs/tfidf-lr_temporal_${PROFILE}_s42/run.json" ]; then
  $PY -m src.baselines tfidf --profile "$PROFILE" --split temporal > logs/tfidf_temporal.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] done tfidf temporal $(tail -1 logs/tfidf_temporal.log)"
fi
run "full_${ENC}_temporal_${PROFILE}_s42" --variant full --split temporal --seed 42
# 5. optional DistilBERT baseline (about 3x slower than MiniLM on CPU)
if [ "$DISTILBERT" = "1" ]; then
  run "cls_distilbert-base-uncased_random_${PROFILE}_s42" --variant cls --encoder distilbert-base-uncased --seed 42 --save-model
fi
# 6. explainability: ThreatWeigh vs plain SBERT
if [ ! -f results/explain_summary.csv ]; then
  echo "[$(date +%H:%M:%S)] start explain"
  $PY -m src.explain --tw "full_${ENC}_random_${PROFILE}_s42" --base "mean_${ENC}_random_${PROFILE}_s42" \
      > logs/explain.log 2>&1 && echo "[$(date +%H:%M:%S)] done explain" || echo "[$(date +%H:%M:%S)] FAILED explain"
fi
$PY -m src.evaluate --profile "$PROFILE" > logs/evaluate.log 2>&1 && echo "[$(date +%H:%M:%S)] ALL DONE"
