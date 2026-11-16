#!/usr/bin/env bash
# =====================================================================
# ThreatWeigh - run every experiment of the report, in priority order.
#
#   bash scripts/run_all.sh            # profile "budget" (what the README reports)
#   PROFILE=full bash scripts/run_all.sh   # paper-scale settings (use a GPU!)
#
# Resumable: a run whose results/runs/<run>/run.json already exists is skipped,
# so the script can simply be started again after an interruption.
# Each run logs to logs/<run>.log.
# =====================================================================
set -u
cd "$(dirname "$0")/.."
PROFILE="${PROFILE:-budget}"
PY="${PY:-python}"
mkdir -p logs
ENC=all-MiniLM-L6-v2

run() {   # run <run-name> <args...>
  local name="$1"; shift
  if [ -f "results/runs/$name/run.json" ]; then echo "[skip] $name"; return; fi
  echo "[$(date +%H:%M:%S)] start $name"
  if $PY -m src.train --profile "$PROFILE" --run-name "$name" "$@" > "logs/$name.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] done  $name  $(grep 'mean test acc' "logs/$name.log")"
  else
    echo "[$(date +%H:%M:%S)] FAILED $name (see logs/$name.log)"
  fi
}

# 1. main models, seed 42 (weights saved for the app and for explainability)
run "full_${ENC}_random_${PROFILE}_s42" --variant full --seed 42 --save-model
run "mean_${ENC}_random_${PROFILE}_s42" --variant mean --seed 42 --save-model
# 2. ablations (seed 42; variant "full" above is ablation (d))
for v in alpha_only alpha_expert alpha_keybert; do
  run "${v}_${ENC}_random_${PROFILE}_s42" --variant "$v" --seed 42
done
# 3. extra seeds for the main comparison
for s in 43 44; do
  run "full_${ENC}_random_${PROFILE}_s$s" --variant full --seed "$s"
  run "mean_${ENC}_random_${PROFILE}_s$s" --variant mean --seed "$s"
done
# 4. sensitivity of the 'both' weight (10 = the full model above)
for w in 2 5 20; do
  run "full-w${w}_${ENC}_random_${PROFILE}_s42" --variant full --weight-both "$w" --seed 42
done
# 5. time-aware split (train <= 2021, test >= 2023)
if [ ! -f "results/runs/tfidf-lr_temporal_${PROFILE}_s42/run.json" ]; then
  $PY -m src.baselines tfidf --profile "$PROFILE" --split temporal > logs/tfidf_temporal.log 2>&1 \
    && echo "done tfidf temporal $(tail -1 logs/tfidf_temporal.log)"
fi
run "full_${ENC}_temporal_${PROFILE}_s42" --variant full --split temporal --seed 42
run "mean_${ENC}_temporal_${PROFILE}_s42" --variant mean --split temporal --seed 42
# 6. DistilBERT baseline (about 3x slower than MiniLM on CPU)
run "cls_distilbert-base-uncased_random_${PROFILE}_s42" --variant cls --encoder distilbert-base-uncased --seed 42 --save-model
# 7. explainability: ThreatWeigh vs plain SBERT (and DistilBERT on a smaller sample)
if [ ! -f results/explain_summary.csv ]; then
  echo "[$(date +%H:%M:%S)] start explain"
  $PY -m src.explain --tw "full_${ENC}_random_${PROFILE}_s42" --base "mean_${ENC}_random_${PROFILE}_s42" \
      > logs/explain.log 2>&1 && echo "[$(date +%H:%M:%S)] done explain" || echo "FAILED explain"
fi
if [ ! -f results/explain_summary_distilbert.csv ]; then
  $PY -m src.explain --tw "full_${ENC}_random_${PROFILE}_s42" --base "cls_distilbert-base-uncased_random_${PROFILE}_s42" \
      --n-samples 100 --n-sensitivity 0 --tag _distilbert > logs/explain_distilbert.log 2>&1 \
      && echo "[$(date +%H:%M:%S)] done explain distilbert" || echo "FAILED explain distilbert"
fi
$PY -m src.evaluate --profile "$PROFILE" > logs/evaluate.log 2>&1 && echo "ALL DONE"
