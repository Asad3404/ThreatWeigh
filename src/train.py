"""
Stage 5 - train one independent model per CVSS metric.

Examples
--------
    # ThreatWeigh (full model), smoke test on CPU (~minutes):
    python -m src.train --variant full --quick

    # the configuration used for the reported results on a CPU box:
    python -m src.train --variant full --profile budget --seed 42 --save-model

    # paper-scale run (all 110k training rows, 15 epochs) - needs a GPU:
    python -m src.train --variant full --profile full --save-model

    # baselines / ablations use the very same loop:
    python -m src.train --variant mean                          # plain SBERT, mean pooling
    python -m src.train --variant cls --encoder distilbert-base-uncased   # DistilBERT
    python -m src.train --variant alpha_only | alpha_expert | alpha_keybert
    python -m src.train --variant full --weight-both 20         # sensitivity test
    python -m src.train --variant full --split temporal         # time-aware split

Variants
--------
    full / alpha_only / alpha_expert / alpha_keybert   weighted pooling (see src/model.py)
    mean                                               mean pooling, no symbolic guidance
    cls                                                [CLS] pooling (DistilBERT baseline)

What gets saved (run name printed at the start):
    results/runs/<run>/preds_<metric>.npz     test-set probabilities + labels
    results/runs/<run>/history_<metric>.csv   per-epoch train loss / val accuracy
    results/runs/<run>/run.json               all settings + timings
    models/<run>/<metric>.pt                  best weights (only with --save-model)

The test split is touched exactly once per metric, AFTER model selection on
the validation split.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data import load_split, subsample
from src.keybert_cache import load_keywords
from src.lexicon import load_lexicon, metric_phrases
from src.model import VARIANT_SOURCES, ThreatWeighModel, load_tokenizer, symbolic_weights
from src.utils import LABELS, get_device, load_config, project_path, save_json, set_seed

POOLING = {"mean": "mean", "cls": "cls", **{v: "weighted" for v in VARIANT_SOURCES}}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class EncodedSplit:
    """Tokenises a split ONCE (with character offsets) so it can be reused for all 8 metrics."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        enc = tokenizer(self.df.text.tolist(), truncation=True, max_length=max_length,
                        return_offsets_mapping=True)
        self.input_ids = enc["input_ids"]
        self.offsets = enc["offset_mapping"]

    def weights(self, variant: str, metric: str, lex: dict, keywords: dict, sym_cfg: dict) -> list[np.ndarray]:
        """Symbolic weight vector for every row (all ones for non-weighted variants)."""
        if variant not in VARIANT_SOURCES:
            return [np.ones(len(ids), np.float32) for ids in self.input_ids]
        use_exp, use_kb = VARIANT_SOURCES[variant]
        exp_phrases = metric_phrases(lex, metric)
        out = []
        for text, offs, cve in zip(self.df.text, self.offsets, self.df.ID):
            out.append(symbolic_weights(text, offs, exp_phrases, keywords.get(cve, []), sym_cfg,
                                        use_exp=use_exp, use_kb=use_kb))
        return out


class MetricDataset(Dataset):
    def __init__(self, input_ids, weights, labels):
        self.input_ids, self.weights, self.labels = input_ids, weights, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.weights[i], self.labels[i]


def make_collate(pad_id: int):
    """Pad each batch only to its own longest text (much faster than always 128)."""

    def collate(batch):
        T = max(len(ids) for ids, _, _ in batch)
        ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        w = torch.ones((len(batch), T), dtype=torch.float)
        for k, (i, wt, _) in enumerate(batch):
            ids[k, :len(i)] = torch.tensor(i)
            mask[k, :len(i)] = 1
            w[k, :len(i)] = torch.from_numpy(np.asarray(wt, dtype=np.float32))
        y = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return ids, mask, w, y

    return collate


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for ids, mask, w, _ in loader:
        logits = model(ids.to(device), mask.to(device), w.to(device))
        out.append(torch.softmax(logits, -1).float().cpu().numpy())
    return np.concatenate(out)


def train_one_metric(metric, splits_enc, args, cfg, lex, keywords, device, run_dir, models_dir):
    """Train, select on val, test once. Returns a small summary dict."""
    tcfg = cfg["train"]
    labels = LABELS[metric]
    lab2id = {l: k for k, l in enumerate(labels)}
    tok_pad = splits_enc["pad_id"]
    collate = make_collate(tok_pad)
    loaders = {}
    for name in ("train", "val", "test"):
        enc: EncodedSplit = splits_enc[name]
        y = [lab2id[v] for v in enc.df[metric]]
        w = enc.weights(args.variant, metric, lex, keywords, cfg["symbolic"])
        ds = MetricDataset(enc.input_ids, w, y)
        g = torch.Generator().manual_seed(args.seed)
        loaders[name] = DataLoader(ds, batch_size=tcfg["batch_size"] if name == "train" else tcfg["eval_batch_size"],
                                   shuffle=(name == "train"), collate_fn=collate, generator=g)

    set_seed(args.seed)   # same initialisation for the pooling/classifier layers in every variant
    model = ThreatWeighModel(args.encoder, len(labels), pooling=POOLING[args.variant],
                             dropout=cfg["model"]["dropout"], eps=cfg["symbolic"]["eps"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()           # no class weighting (as in the paper)
    y_val = np.array([lab2id[v] for v in splits_enc["val"].df[metric]])

    best_acc, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for ids, mask, w, y in loaders["train"]:
            ids, mask, w, y = ids.to(device), mask.to(device), w.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, mask, w), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        val_acc = float((predict_proba(model, loaders["val"], device).argmax(1) == y_val).mean())
        history.append({"epoch": epoch, "train_loss": total / n, "val_acc": val_acc, "sec": time.time() - t0})
        print(f"  [{metric.upper()}] epoch {epoch}/{tcfg['epochs']}  loss={total / n:.4f}  "
              f"val_acc={val_acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc, best_epoch, bad = val_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                print(f"  [{metric.upper()}] early stopping (no val improvement for {bad} epochs)")
                break

    model.load_state_dict(best_state)
    probs = predict_proba(model, loaders["test"], device)       # the ONLY look at the test split
    test_df = splits_enc["test"].df
    y_test = np.array([lab2id[v] for v in test_df[metric]])
    np.savez_compressed(run_dir / f"preds_{metric}.npz", ids=test_df.ID.to_numpy(), y_true=y_test,
                        probs=probs, labels=np.array(labels))
    pd.DataFrame(history).to_csv(run_dir / f"history_{metric}.csv", index=False)
    if args.save_model:
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, models_dir / f"{metric}.pt")
    test_acc = float((probs.argmax(1) == y_test).mean())
    print(f"  [{metric.upper()}] best epoch {best_epoch}  val_acc={best_acc:.4f}  TEST acc={test_acc:.4f}")
    return {"metric": metric, "best_epoch": best_epoch, "val_acc": best_acc, "test_acc": test_acc}


def default_run_name(args, cfg) -> str:
    enc = args.encoder.split("/")[-1]
    name = f"{args.variant}"
    if args.weight_both is not None:
        name += f"-w{args.weight_both:g}"
    name += f"_{enc}_{args.split}_{cfg['profile']}_s{args.seed}"
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="full", choices=sorted(POOLING))
    ap.add_argument("--encoder", default=None, help="HF checkpoint (default: config model.encoder)")
    ap.add_argument("--metrics", nargs="+", default=None, help="subset of metrics, e.g. av ac")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--profile", default=None, help="quick | budget | full (default full)")
    ap.add_argument("--quick", action="store_true", help="alias for --profile quick")
    ap.add_argument("--epochs", type=int, default=None, help="override number of epochs")
    ap.add_argument("--split", default="random", choices=["random", "temporal"])
    ap.add_argument("--weight-both", type=float, default=None, help="override symbolic weight_both (10)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    profile = "quick" if args.quick else (args.profile or "full")
    cfg = load_config(profile)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.weight_both is not None:
        cfg["symbolic"]["weight_both"] = args.weight_both
    args.encoder = args.encoder or cfg["model"]["encoder"]
    args.seed = cfg["seed"] if args.seed is None else args.seed
    metrics = args.metrics or cfg["metrics"]
    run = args.run_name or default_run_name(args, cfg)
    run_dir = project_path(cfg["paths"]["results_dir"]) / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = project_path(cfg["paths"]["models_dir"]) / run

    if cfg["train"].get("num_threads"):
        torch.set_num_threads(cfg["train"]["num_threads"])
    device = get_device()
    print(f"Run '{run}'  device={device}  profile={profile}  encoder={args.encoder}  variant={args.variant}")

    # The data subsample is drawn with the MASTER seed so that every variant and
    # every training seed sees exactly the same rows; --seed only changes the
    # model initialisation, dropout and batch order.
    t = cfg["train"]
    train = subsample(load_split("train", cfg, args.split), t["max_train"], cfg["seed"])
    val = subsample(load_split("val", cfg, args.split), t["max_val"], cfg["seed"])
    test = subsample(load_split("test", cfg, args.split), t["max_test"], cfg["seed"])
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}")

    tok = load_tokenizer(args.encoder)
    ml = cfg["model"]["max_length"]
    splits_enc = {"train": EncodedSplit(train, tok, ml), "val": EncodedSplit(val, tok, ml),
                  "test": EncodedSplit(test, tok, ml), "pad_id": tok.pad_token_id}
    lex = load_lexicon(cfg)
    keywords = load_keywords(cfg) if args.variant in ("full", "alpha_keybert") else {}

    summaries, t_start = [], time.time()
    for m in metrics:
        t0 = time.time()
        s = train_one_metric(m, splits_enc, args, cfg, lex, keywords, device, run_dir, models_dir)
        s["minutes"] = round((time.time() - t0) / 60, 2)
        summaries.append(s)
    info = {"run": run, "variant": args.variant, "encoder": args.encoder, "seed": args.seed,
            "split": args.split, "profile": profile, "train_cfg": cfg["train"], "symbolic": cfg["symbolic"],
            "n_train": len(train), "n_val": len(val), "n_test": len(test), "device": str(device),
            "metrics": summaries, "total_minutes": round((time.time() - t_start) / 60, 2)}
    save_json(info, run_dir / "run.json")
    print(json.dumps({s["metric"]: round(s["test_acc"], 4) for s in summaries}))
    print(f"mean test acc = {np.mean([s['test_acc'] for s in summaries]):.4f}   "
          f"({info['total_minutes']} min)")


if __name__ == "__main__":
    main()
