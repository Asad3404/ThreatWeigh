# ThreatWeigh - TODO

Status legend: [x] done and run, [~] in progress, [ ] not started.

## Stages
- [x] 1. Setup, data download (WSymBERT repo, `Nvd/Dataset/final.zip`), cleaning, strict dedup, 80/10/10 stratified split, temporal split, EDA (`python -m src.data`)
- [x] 2. `src/cvss.py` + `tests/test_cvss.py` (matches NVD's score on all 137k rows)
- [x] 3. Expert lexicons (`python -m src.lexicon`, coverage) + KeyBERT cache (`python -m src.keybert_cache`)
- [~] 4. Baselines: TF-IDF+LR (done), SBERT mean pooling (running). DistilBERT dropped from the run plan (CPU budget; `DISTILBERT=1 bash scripts/run_all.sh`)
- [x] 5. ThreatWeigh full model, 8 metrics, CVSS score MSE/MAE (seed 42: mean acc 88.28%)
- [ ] 6. Ablations (a-d), bootstrap CI + McNemar, weight sensitivity (5/10/20), temporal split. Seeds reduced to 1 (user chose the faster plan)
- [ ] 7. Explainability on 300 test CVEs: IG (pad / ones baselines), top-10 words, noise rate, infidelity, sensitivity, deletion-AUC
- [x] 8. Streamlit app (3 tabs) + `predict.py` CLI (tested headless + in a browser with the trained model)
- [ ] 9. README with equations, results, limitations, credits

## Known issues / decisions
- Label leakage: 9,199 descriptions contained the CVSS score/vector; stripped in `src/data.py` (tested).
- The build container restarted once mid-run; the queue and `src.train` now resume per metric.
- The brief's expected score 4.6 for `AV:P/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L` is wrong; the official value (and NVD's) is 4.3. The test asserts 4.3.
- No GPU in the build machine: results use the `budget` profile (6k-row train subsample, 4 epochs). Paper-scale settings are the `full` profile.
- Lexicon lift filter cannot keep cues for very frequent labels (AC:L 94%, S:U 82%): by design, cues are informative for minority labels.
- The reference WSymBERT code differs from the paper equations (weights capped at 5, no padding mask, 80/20 split without test set, 3 epochs). ThreatWeigh follows the paper equations.

## Nice to have (not done)
- `fetch_nvd.py` NVD API 2.0 fallback (not needed: the clone worked).
- Paper-scale GPU run with `all-mpnet-base-v2`.
