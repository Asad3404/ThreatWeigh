# ThreatWeigh - TODO

Status legend: [x] done and run, [~] in progress, [ ] not started.

## Stages
- [x] 1. Setup, data download (WSymBERT repo, `Nvd/Dataset/final.zip`), cleaning, strict dedup, 80/10/10 stratified split, temporal split, EDA (`python -m src.data`)
- [x] 2. `src/cvss.py` + `tests/test_cvss.py` (matches NVD's score on all 137k rows)
- [x] 3. Expert lexicons (`python -m src.lexicon`, coverage) + KeyBERT cache (`python -m src.keybert_cache`)
- [x] 4. Baselines: TF-IDF+LR, SBERT mean pooling. DistilBERT dropped from the run plan (CPU budget; `DISTILBERT=1 bash scripts/run_all.sh`)
- [x] 5. ThreatWeigh full model, 8 metrics, CVSS score MSE/MAE (seed 42: mean acc 88.28%)
- [x] 6. Ablations (a-d), bootstrap CI + McNemar, weight sensitivity (5/10/20), temporal split. Seeds reduced to 1 (user chose the faster plan)
- [x] 7. Explainability on 300 test CVEs: IG (pad / ones baselines), top-10 words, noise rate, infidelity, sensitivity, deletion-AUC
- [x] 8. Streamlit app (3 tabs) + `predict.py` CLI (tested headless + in a browser with the trained model)
- [x] 9. README with equations, results, limitations, credits

## Known issues / decisions
- Label leakage: 9,199 descriptions contained the CVSS score/vector; stripped in `src/data.py` (tested).
- The build container restarted once mid-run; the queue and `src.train` now resume per metric.
- The brief's expected score 4.6 for `AV:P/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L` is wrong; the official value (and NVD's) is 4.3. The test asserts 4.3.
- No GPU in the build machine: results use the `budget` profile (4k-row train subsample, up to 3 epochs, 1 seed). Paper-scale settings are the `full` profile.
- Lexicon lift filter cannot keep cues for very frequent labels (AC:L 94%, S:U 82%): by design, cues are informative for minority labels.
- The reference WSymBERT code differs from the paper equations (weights capped at 5, no padding mask, 80/20 split without test set, 3 epochs). ThreatWeigh follows the paper equations.

## Nice to have (not done)
- 3 seeds (43, 44) for mean ± std: `SEEDS="42 43 44" bash scripts/run_all.sh`.
- DistilBERT baseline: `DISTILBERT=1 bash scripts/run_all.sh`.
- Weight-sensitivity point w=2; plain-SBERT run on the temporal split.
- `fetch_nvd.py` NVD API 2.0 fallback (not needed: the clone worked).
- Paper-scale GPU run with `all-mpnet-base-v2`.
