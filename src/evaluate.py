"""
Evaluation: turns saved test predictions into the result tables in results/.

    python -m src.evaluate                 # rebuild every table from results/runs/*

For each run (= one model type, one seed, all 8 metrics) we compute
  * per metric: accuracy, macro-precision, macro-recall, macro-F1
  * mean accuracy and mean macro-F1 over the 8 metrics
  * the CVSS v3.1 Base Score from the 8 PREDICTED labels, compared with
      - the score recomputed from the TRUE labels with the v3.1 formula, and
      - NVD's own score column (27% of rows were scored with CVSS v3.0)
    as MSE and MAE, plus the % of CVEs whose severity band is correct.

Runs with the same name apart from the seed (`..._s42`, `..._s43`, ...) are
grouped and reported as mean ± std. For ThreatWeigh vs plain SBERT we add a
paired bootstrap 95% confidence interval and McNemar tests on the test CVEs.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.cvss import base_score, severity
from src.utils import LABELS, load_config, load_json, project_path

METRICS = list(LABELS)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def runs_dir(cfg) -> Path:
    return project_path(cfg["paths"]["results_dir"]) / "runs"


def load_run(run: str, cfg) -> dict:
    """{metric: {'ids','y_true','probs','labels'}} for one run (all available metrics)."""
    d = runs_dir(cfg) / run
    out = {}
    for m in METRICS:
        p = d / f"preds_{m}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            out[m] = {k: z[k] for k in z.files}
    return out


def strip_seed(run: str) -> str:
    return re.sub(r"_s\d+$", "", run)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def classification_table(preds: dict) -> pd.DataFrame:
    rows = []
    for m, p in preds.items():
        y, yhat = p["y_true"], p["probs"].argmax(1)
        P, R, F, _ = precision_recall_fscore_support(y, yhat, average="macro",
                                                     labels=range(len(p["labels"])), zero_division=0)
        rows.append({"metric": m.upper(), "accuracy": accuracy_score(y, yhat),
                     "macro_precision": P, "macro_recall": R, "macro_f1": F})
    return pd.DataFrame(rows)


def predicted_labels(preds: dict) -> pd.DataFrame:
    """One row per test CVE with the predicted and the true letter for each metric."""
    first = next(iter(preds.values()))
    df = pd.DataFrame({"ID": first["ids"]})
    for m, p in preds.items():
        assert (p["ids"] == first["ids"]).all(), "test rows must be in the same order"
        labs = np.asarray(p["labels"]).astype(str)
        df[f"pred_{m}"] = labs[p["probs"].argmax(1)]
        df[f"true_{m}"] = labs[p["y_true"]]
    return df


def score_errors(preds: dict, test_df: pd.DataFrame | None = None) -> dict:
    """MSE / MAE between the Base Score from predicted labels and the true score."""
    if set(preds) != set(METRICS):
        return {}
    lab = predicted_labels(preds)
    pred_s = lab.apply(lambda r: base_score({m: r[f"pred_{m}"] for m in METRICS}), axis=1).to_numpy()
    true_s = lab.apply(lambda r: base_score({m: r[f"true_{m}"] for m in METRICS}), axis=1).to_numpy()
    out = {"score_mse": float(np.mean((pred_s - true_s) ** 2)),
           "score_mae": float(np.mean(np.abs(pred_s - true_s))),
           "severity_acc": float(np.mean([severity(a) == severity(b) for a, b in zip(pred_s, true_s)])),
           "exact_vector_acc": float(np.mean(np.all([lab[f"pred_{m}"] == lab[f"true_{m}"] for m in METRICS], 0)))}
    if test_df is not None:
        nvd = test_df.set_index("ID").loc[lab.ID, "score"].to_numpy(dtype=float)
        out["score_mse_vs_nvd"] = float(np.mean((pred_s - nvd) ** 2))
        out["score_mae_vs_nvd"] = float(np.mean(np.abs(pred_s - nvd)))
    return out


def run_summary(run: str, cfg, test_df=None) -> dict:
    preds = load_run(run, cfg)
    tab = classification_table(preds)
    s = {"run": run, "group": strip_seed(run), "n_metrics": len(preds),
         "mean_accuracy": tab.accuracy.mean(), "mean_macro_f1": tab.macro_f1.mean()}
    for _, r in tab.iterrows():
        s[f"acc_{r.metric}"] = r.accuracy
        s[f"f1_{r.metric}"] = r.macro_f1
    s.update(score_errors(preds, test_df))
    info = runs_dir(cfg) / run / "run.json"
    if info.exists():
        j = load_json(info)
        s.update({"n_train": j.get("n_train"), "minutes": j.get("total_minutes")})
    return s


def mean_std(df: pd.DataFrame, cols) -> pd.DataFrame:
    g = df.groupby("group")
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


# ---------------------------------------------------------------------------
# Significance: paired bootstrap + McNemar
# ---------------------------------------------------------------------------
def correctness(runs: list[str], cfg) -> dict[str, np.ndarray]:
    """{metric: [n_seeds, n_test] boolean matrix of correct predictions}."""
    out = {}
    for m in METRICS:
        rows = []
        for r in runs:
            p = load_run(r, cfg)[m]
            rows.append(p["probs"].argmax(1) == p["y_true"])
        out[m] = np.stack(rows)
    return out


def paired_bootstrap(runs_a: list[str], runs_b: list[str], cfg, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """95% CI of accuracy(A) - accuracy(B) by resampling the test CVEs (paired).
    Each model's correctness is first averaged over its seeds."""
    ca, cb = correctness(runs_a, cfg), correctness(runs_b, cfg)
    n = next(iter(ca.values())).shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rows, diffs_all = [], []
    for m in METRICS:
        d = ca[m].mean(0) - cb[m].mean(0)            # per-CVE difference in (seed-averaged) correctness
        boots = d[idx].mean(1)
        diffs_all.append(boots)
        # McNemar exact test on the first seed of each model (discordant pairs)
        a0, b0 = ca[m][0], cb[m][0]
        n10, n01 = int((a0 & ~b0).sum()), int((~a0 & b0).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        rows.append({"metric": m.upper(), "acc_diff": d.mean(), "ci_low": np.percentile(boots, 2.5),
                     "ci_high": np.percentile(boots, 97.5), "A_only_correct": n10, "B_only_correct": n01,
                     "mcnemar_p": p})
    mean_boot = np.mean(diffs_all, 0)
    rows.append({"metric": "MEAN", "acc_diff": float(np.mean([r["acc_diff"] for r in rows])),
                 "ci_low": np.percentile(mean_boot, 2.5), "ci_high": np.percentile(mean_boot, 97.5),
                 "A_only_correct": np.nan, "B_only_correct": np.nan, "mcnemar_p": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Building the report tables
# ---------------------------------------------------------------------------
DISPLAY = {   # group name prefix -> pretty name, in table order
    "tfidf-lr": "TF-IDF + LogReg",
    "cls_distilbert-base-uncased": "DistilBERT ([CLS])",
    "mean_all-MiniLM-L6-v2": "SBERT (mean pooling)",
    "alpha_only_all-MiniLM-L6-v2": "ThreatWeigh (a) alpha only",
    "alpha_expert_all-MiniLM-L6-v2": "ThreatWeigh (b) alpha + expert",
    "alpha_keybert_all-MiniLM-L6-v2": "ThreatWeigh (c) alpha + KeyBERT",
    "full_all-MiniLM-L6-v2": "ThreatWeigh (full)",
}


def pretty(group: str) -> str:
    for k, v in DISPLAY.items():
        if group.startswith(k + "_"):
            rest = group[len(k) + 1:]
            return f"{v} [{rest}]"
    m = re.match(r"full-w([\d.]+)_", group)
    if m:
        return f"ThreatWeigh (weight_both={m.group(1)}) [{group.split('_', 1)[1]}]"
    return group


def fmt(mean, std, pct=True, nd=2):
    k = 100 if pct else 1
    if std is None or np.isnan(std):
        return f"{mean * k:.{nd}f}"
    return f"{mean * k:.{nd}f} ± {std * k:.{nd}f}"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_tables(cfg, profile_filter: str = "budget") -> dict[str, pd.DataFrame]:
    from src.data import load_split

    rd = runs_dir(cfg)
    runs = sorted(p.name for p in rd.iterdir() if (p / "preds_av.npz").exists()) if rd.exists() else []
    test_random = load_split("test", cfg)
    try:
        test_temporal = load_split("test", cfg, "temporal")
    except FileNotFoundError:
        test_temporal = None
    summaries = []
    for r in runs:
        test_df = test_temporal if "_temporal_" in r else test_random
        summaries.append(run_summary(r, cfg, test_df))
    all_runs = pd.DataFrame(summaries)
    res = project_path(cfg["paths"]["results_dir"])
    all_runs.to_csv(res / "all_runs.csv", index=False)
    tables = {"all_runs": all_runs}
    if all_runs.empty:
        return tables

    num_cols = [c for c in all_runs.columns if c.startswith(("acc_", "f1_", "mean_", "score_", "severity", "exact"))]
    agg = mean_std(all_runs[all_runs.n_metrics == 8], num_cols)
    agg["model"] = agg.group.map(pretty)

    # ---- main comparison (random split, chosen profile, weight 10) -------------------
    def main_table(groups):
        rows = []
        for g in groups:
            r = agg[agg.group == g]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {"model": r.model, "seeds": int(r.n_seeds)}
            for m in METRICS:
                row[m.upper()] = fmt(r[f"acc_{m.upper()}_mean"], r[f"acc_{m.upper()}_std"], nd=1)
            row["mean acc %"] = fmt(r["mean_accuracy_mean"], r["mean_accuracy_std"])
            row["mean macro-F1 %"] = fmt(r["mean_macro_f1_mean"], r["mean_macro_f1_std"])
            if "score_mse_mean" in r and not np.isnan(r["score_mse_mean"]):
                row["score MSE"] = fmt(r["score_mse_mean"], r["score_mse_std"], pct=False, nd=3)
                row["score MAE"] = fmt(r["score_mae_mean"], r["score_mae_std"], pct=False, nd=3)
                row["severity acc %"] = fmt(r["severity_acc_mean"], r["severity_acc_std"])
            rows.append(row)
        return pd.DataFrame(rows)

    sfx = f"random_{profile_filter}"
    main_groups = [f"tfidf-lr_{sfx}", f"tfidf-lr_random_full", f"cls_distilbert-base-uncased_{sfx}",
                   f"mean_all-MiniLM-L6-v2_{sfx}", f"full_all-MiniLM-L6-v2_{sfx}"]
    tables["main_comparison"] = main_table(main_groups)
    tables["ablation"] = main_table([f"{v}_all-MiniLM-L6-v2_{sfx}"
                                     for v in ["mean", "alpha_only", "alpha_expert", "alpha_keybert", "full"]])
    sens = [f"full-w{w:g}_all-MiniLM-L6-v2_{sfx}" for w in cfg["experiments"]["weight_sensitivity"]]
    sens = [g if g in set(agg.group) else f"full_all-MiniLM-L6-v2_{sfx}" if "w10_" in g else g for g in sens]
    tables["weight_sensitivity"] = main_table(sens)
    tables["temporal"] = main_table([f"tfidf-lr_temporal_{profile_filter}", f"mean_all-MiniLM-L6-v2_temporal_{profile_filter}",
                                     f"full_all-MiniLM-L6-v2_temporal_{profile_filter}"])

    # ---- per-metric P/R/F1 tables for every model group (first seed + seed mean) ----
    per_metric = []
    for g in agg.group:
        members = [r for r in runs if strip_seed(r) == g]
        tabs = [classification_table(load_run(r, cfg)) for r in members]
        cat = pd.concat(tabs).groupby("metric", sort=False).agg(["mean", "std"])
        for metric, r in cat.iterrows():
            per_metric.append({"model": pretty(g), "metric": metric,
                               **{k: fmt(r[(k, "mean")], r[(k, "std")]) for k in
                                  ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}})
    tables["per_metric"] = pd.DataFrame(per_metric)

    # ---- per-seed table (variance across seeds) -------------------------------------
    seeds = all_runs[all_runs.group.isin([f"mean_all-MiniLM-L6-v2_{sfx}", f"full_all-MiniLM-L6-v2_{sfx}"])]
    if not seeds.empty:
        tables["seeds"] = seeds.assign(model=seeds.group.map(pretty))[
            ["model", "run", "mean_accuracy", "mean_macro_f1", "score_mse", "score_mae"]].round(4)

    # ---- significance: ThreatWeigh (full) vs plain SBERT --------------------------
    full_runs = sorted(r for r in runs if strip_seed(r) == f"full_all-MiniLM-L6-v2_{sfx}")
    mean_runs = sorted(r for r in runs if strip_seed(r) == f"mean_all-MiniLM-L6-v2_{sfx}")
    if full_runs and mean_runs:
        sig = paired_bootstrap(full_runs, mean_runs, cfg, cfg["experiments"]["bootstrap_samples"])
        for c in ["acc_diff", "ci_low", "ci_high"]:
            sig[c] = (100 * sig[c]).round(2)
        sig["mcnemar_p"] = sig["mcnemar_p"].map(lambda p: "" if pd.isna(p) else f"{p:.3g}")
        tables["significance_full_vs_sbert"] = sig.rename(columns={
            "acc_diff": "acc diff (pp)", "ci_low": "95% CI low", "ci_high": "95% CI high",
            "A_only_correct": "only ThreatWeigh correct", "B_only_correct": "only SBERT correct"})

    for name, t in tables.items():
        if name == "all_runs" or t.empty:
            continue
        t.to_csv(res / f"{name}.csv", index=False)
        (res / f"{name}.md").write_text(to_markdown(t) + "\n")
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="budget", help="which training profile's runs to tabulate")
    args = ap.parse_args()
    cfg = load_config()
    tables = build_tables(cfg, args.profile)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, t in tables.items():
        if name in ("all_runs", "per_metric") or t.empty:
            continue
        print(f"\n=== {name} ===")
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
