"""
Stage 4 - baselines.

1. Classic baseline: TF-IDF (word 1-2 grams) + logistic regression, one per metric.
   Run on the SAME training rows as the neural models (--profile budget uses the
   same 6,000-row stratified subsample) and, as a reference, on all training rows.

       python -m src.baselines tfidf --profile budget      # same rows as neural runs
       python -m src.baselines tfidf --profile full        # all 110k training rows

2. Neural baselines use the shared training loop in src/train.py so that the
   splits, optimiser, learning rate, batch size and epochs are identical:

       python -m src.baselines sbert      [--profile budget --seed 42]
           == python -m src.train --variant mean
       python -m src.baselines distilbert [--profile budget --seed 42]
           == python -m src.train --variant cls --encoder distilbert-base-uncased
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_split, subsample
from src.utils import LABELS, load_config, project_path, save_json, set_seed


def run_tfidf(profile: str, split: str = "random"):
    cfg = load_config(profile)
    set_seed(cfg["seed"])
    t = cfg["train"]
    train = subsample(load_split("train", cfg, split), t["max_train"], cfg["seed"])
    test = subsample(load_split("test", cfg, split), t["max_test"], cfg["seed"])
    tc = cfg["tfidf"]
    run = f"tfidf-lr_{split}_{profile}_s{cfg['seed']}"
    run_dir = project_path(cfg["paths"]["results_dir"]) / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run '{run}': train={len(train)} test={len(test)}")

    # The vocabulary is fitted on the training texts only.
    vec = TfidfVectorizer(ngram_range=tuple(tc["ngram_range"]), max_features=tc["max_features"],
                          sublinear_tf=True, min_df=2, lowercase=True)
    X_train = vec.fit_transform(train.text)
    X_test = vec.transform(test.text)
    summary = []
    for m in cfg["metrics"]:
        t0 = time.time()
        labels = LABELS[m]
        lab2id = {l: k for k, l in enumerate(labels)}
        y_train = train[m].map(lab2id).to_numpy()
        y_test = test[m].map(lab2id).to_numpy()
        clf = LogisticRegression(C=tc["C"], max_iter=tc["max_iter"])   # no class weighting
        clf.fit(X_train, y_train)
        # predict_proba only has columns for classes seen in training -> place them correctly
        probs = np.zeros((X_test.shape[0], len(labels)))
        probs[:, clf.classes_] = clf.predict_proba(X_test)
        np.savez_compressed(run_dir / f"preds_{m}.npz", ids=test.ID.to_numpy(), y_true=y_test,
                            probs=probs, labels=np.array(labels))
        acc = float((probs.argmax(1) == y_test).mean())
        summary.append({"metric": m, "test_acc": acc})
        print(f"  [{m.upper()}] test acc = {acc:.4f}  ({time.time() - t0:.0f}s)")
    save_json({"run": run, "variant": "tfidf-lr", "profile": profile, "n_train": len(train),
               "n_test": len(test), "metrics": summary}, run_dir / "run.json")
    print(f"mean test acc = {np.mean([s['test_acc'] for s in summary]):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["tfidf", "sbert", "distilbert"])
    ap.add_argument("--profile", default="budget")
    ap.add_argument("--split", default="random", choices=["random", "temporal"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()
    if args.which == "tfidf":
        return run_tfidf(args.profile, args.split)
    cmd = [sys.executable, "-m", "src.train", "--profile", args.profile, "--split", args.split]
    if args.which == "sbert":
        cmd += ["--variant", "mean"]
    else:
        cmd += ["--variant", "cls", "--encoder", load_config()["model"]["distilbert"]]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.save_model:
        cmd += ["--save-model"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
