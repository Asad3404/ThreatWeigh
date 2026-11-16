"""
Stage 7 - inference + explainability.

This module has two jobs:

1. INFERENCE: `ThreatWeighPredictor` loads the 8 metric-specific models of a
   trained run and turns a CVE description into 8 labels (+ confidence), the
   CVSS vector, the Base Score and the severity. Used by predict.py and app.py.

2. EXPLANATION: Integrated Gradients (Captum) on the token *embeddings* of each
   metric-specific model, with two reference baselines:
     * "pad"  - every position replaced by the [PAD] token embedding
     * "ones" - an all-ones embedding
   and quantitative checks on a sample of test CVEs:
     * noise rate  - % of the top-10 attributed words that are stop-words,
                     pure numbers / version strings, or <= 2 characters
     * infidelity  - Captum `infidelity` (lower = attributions predict the effect
                     of small input perturbations better)
     * sensitivity - Captum `sensitivity_max` (lower = more stable explanation)
     * deletion-AUC- remove words from most to least attributed and track the
                     predicted-class probability; lower area = more faithful.
                     A random deletion order is reported as a reference.

Run:
    python -m src.explain --tw <ThreatWeigh run> --base <SBERT-mean run>

IMPORTANT: attributions DESCRIBE what the model's output is sensitive to; they
are not proof that the model "reasons" like a security analyst.
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.cvss import base_score, severity, vector_string
from src.lexicon import load_lexicon, metric_phrases, phrase_to_labels
from src.model import VARIANT_SOURCES, ThreatWeighModel, char_mask, load_tokenizer, symbolic_weights
from src.utils import LABELS, get_device, load_config, load_json, project_path, set_seed

POOLING = {"mean": "mean", "cls": "cls", **{v: "weighted" for v in VARIANT_SOURCES}}
NUMBER_RE = re.compile(r"^(v|version)?[\d._\-x]+[a-z]?$", re.IGNORECASE)   # 2.4.1, 10, v1.2, 5.x
SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"}


# ---------------------------------------------------------------------------
# Loading one metric model
# ---------------------------------------------------------------------------
class MetricModel:
    """A trained model for ONE metric plus everything needed to feed it."""

    def __init__(self, run: str, metric: str, cfg: dict, device=None, lex=None):
        info = load_json(project_path(cfg["paths"]["results_dir"]) / "runs" / run / "run.json")
        self.run, self.metric, self.cfg = run, metric, cfg
        self.variant, self.encoder = info["variant"], info["encoder"]
        self.sym_cfg = info.get("symbolic", cfg["symbolic"])
        self.labels = LABELS[metric]
        self.device = device or get_device()
        self.tok = load_tokenizer(self.encoder)
        self.model = ThreatWeighModel(self.encoder, len(self.labels), POOLING[self.variant],
                                      cfg["model"]["dropout"], cfg["symbolic"]["eps"])
        state = torch.load(project_path(cfg["paths"]["models_dir"]) / run / f"{metric}.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.lex = lex or load_lexicon(cfg)
        self.exp_phrases = metric_phrases(self.lex, metric)

    # -- encoding ---------------------------------------------------------------
    def encode(self, text: str, kb_words=()) -> dict:
        enc = self.tok(text, truncation=True, max_length=self.cfg["model"]["max_length"],
                       return_offsets_mapping=True)
        if self.variant in VARIANT_SOURCES:
            use_exp, use_kb = VARIANT_SOURCES[self.variant]
            w = symbolic_weights(text, enc["offset_mapping"], self.exp_phrases, list(kb_words), self.sym_cfg,
                                 use_exp, use_kb)
        else:
            w = np.ones(len(enc["input_ids"]), np.float32)
        d = self.device
        return {"text": text, "offsets": enc["offset_mapping"], "word_ids": enc.word_ids(),
                "tokens": self.tok.convert_ids_to_tokens(enc["input_ids"]),
                "ids": torch.tensor([enc["input_ids"]], device=d),
                "mask": torch.ones(1, len(enc["input_ids"]), dtype=torch.long, device=d),
                "w": torch.tensor(w, device=d).unsqueeze(0)}

    def embed(self, ids):
        return self.model.word_embeddings()(ids)

    def forward_embeds(self, emb, mask, w):
        """Class probabilities from word embeddings (what IG differentiates)."""
        n = emb.shape[0]
        logits = self.model(inputs_embeds=emb, attention_mask=mask.expand(n, -1), sym_weights=w.expand(n, -1))
        return torch.softmax(logits, -1)

    @torch.no_grad()
    def predict(self, enc) -> np.ndarray:
        return self.forward_embeds(self.embed(enc["ids"]), enc["mask"], enc["w"])[0].cpu().numpy()

    # -- attribution ----------------------------------------------------------------
    def baseline(self, emb, kind: str):
        if kind == "pad":
            pad = self.embed(torch.tensor([[self.tok.pad_token_id]], device=emb.device))
            return pad.expand_as(emb).detach().clone()
        if kind == "ones":
            return torch.ones_like(emb)
        raise ValueError(kind)

    def integrated_gradients(self, enc, target: int, kind: str = "pad", n_steps: int = 32):
        """Token-level IG attributions (summed over the embedding dimension)."""
        from captum.attr import IntegratedGradients

        emb = self.embed(enc["ids"]).detach()
        ig = IntegratedGradients(lambda e, m, w: self.forward_embeds(e, m, w))
        attr = ig.attribute(emb, baselines=self.baseline(emb, kind), target=target,
                            additional_forward_args=(enc["mask"], enc["w"]), n_steps=n_steps,
                            internal_batch_size=n_steps)
        return attr, emb

    def word_attributions(self, enc, tok_attr: np.ndarray) -> list[dict]:
        """Merge sub-word tokens into words (sum of their attributions)."""
        words: dict[int, dict] = {}
        for t, wid in enumerate(enc["word_ids"]):
            if wid is None:
                continue
            a, b = enc["offsets"][t]
            d = words.setdefault(wid, {"start": a, "end": b, "attr": 0.0, "tokens": []})
            d["start"], d["end"] = min(d["start"], a), max(d["end"], b)
            d["attr"] += float(tok_attr[t])
            d["tokens"].append(t)
        out = []
        lexmask = char_mask(enc["text"], self.exp_phrases, True)
        for wid, d in words.items():
            d["word"] = enc["text"][d["start"]:d["end"]]
            d["in_lexicon"] = bool(lexmask[d["start"]:d["end"]].any())
            d["sym_weight"] = float(enc["w"][0, d["tokens"]].max())
            out.append(d)
        return out


def top_words(words: list[dict], k: int = 10) -> list[dict]:
    """The k words with the largest positive attribution towards the prediction."""
    return sorted(words, key=lambda d: -d["attr"])[:k]


def is_noise(word: str) -> bool:
    w = word.lower().strip()
    return len(w) <= 2 or w in ENGLISH_STOP_WORDS or bool(NUMBER_RE.match(w))


# ---------------------------------------------------------------------------
# Inference for all 8 metrics
# ---------------------------------------------------------------------------
class ThreatWeighPredictor:
    """Loads the 8 models of a run. `predict(text)` returns everything the app shows."""

    def __init__(self, run: str, cfg: dict | None = None, device=None):
        self.cfg = cfg or load_config()
        self.lex = load_lexicon(self.cfg)
        self.models = {m: MetricModel(run, m, self.cfg, device, self.lex) for m in self.cfg["metrics"]}
        self.variant = next(iter(self.models.values())).variant
        self._kw = None

    def keywords(self, text: str) -> list[str]:
        """KeyBERT top-5 unigrams for a NEW text (same settings as the training cache)."""
        if not any(VARIANT_SOURCES.get(self.variant, (0, 0))[1:]):
            return []
        if self._kw is None:
            from keybert import KeyBERT
            from sentence_transformers import SentenceTransformer

            self._kw = KeyBERT(model=SentenceTransformer(self.cfg["model"]["encoder"], device="cpu"))
        kb = self.cfg["keybert"]
        return [w for w, _ in self._kw.extract_keywords(text, keyphrase_ngram_range=tuple(kb["ngram_range"]),
                                                          stop_words=kb["stop_words"], top_n=kb["top_n"],
                                                          use_mmr=kb["use_mmr"])]

    def predict(self, text: str, explain: bool = False, baseline: str = "pad", n_steps: int = 32) -> dict:
        kws = self.keywords(text)
        out = {"text": text, "keywords": kws, "metrics": {}}
        for m, mm in self.models.items():
            enc = mm.encode(text, kws)
            probs = mm.predict(enc)
            k = int(probs.argmax())
            r = {"label": mm.labels[k], "confidence": float(probs[k]),
                 "probs": {l: float(p) for l, p in zip(mm.labels, probs)}}
            if explain:
                attr, _ = mm.integrated_gradients(enc, k, baseline, n_steps)
                words = mm.word_attributions(enc, attr.sum(-1)[0].detach().cpu().numpy())
                p2l = phrase_to_labels(self.lex, m)
                for w in words:
                    w["lexicon_labels"] = sorted({l for p, ls in p2l.items()
                                                  if char_mask(w["word"], [p], True).any() or p == w["word"].lower()
                                                  for l in ls}) if w["in_lexicon"] else []
                r["words"] = words
            out["metrics"][m] = r
        labs = {m: r["label"] for m, r in out["metrics"].items()}
        out["vector"] = vector_string(labs)
        out["score"] = base_score(labs)
        out["severity"] = severity(out["score"])
        return out


# ---------------------------------------------------------------------------
# Quantitative checks
# ---------------------------------------------------------------------------
def deletion_curve(mm: MetricModel, enc, order: list[list[int]], target: int, steps: int) -> np.ndarray:
    """Probability of `target` after removing the first 0, 1/steps, ..., 100% of the words in
    `order` (each word = list of its token positions). Removal = attention mask set to 0."""
    n = len(order)
    masks = []
    for s in range(steps + 1):
        k = int(round(n * s / steps))
        m = enc["mask"].clone()
        for toks in order[:k]:
            m[0, toks] = 0
        masks.append(m)
    M = torch.cat(masks, 0)
    with torch.no_grad():
        emb = mm.embed(enc["ids"]).expand(len(masks), -1, -1)
        logits = mm.model(inputs_embeds=emb, attention_mask=M, sym_weights=enc["w"].expand(len(masks), -1))
        return torch.softmax(logits, -1)[:, target].cpu().numpy()


def auc(y: np.ndarray) -> float:
    x = np.linspace(0, 1, len(y))
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def quantitative(run: str, label: str, sample: pd.DataFrame, keywords: dict, cfg: dict, n_sens: int,
                 rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    """Compute the explanation metrics of one run on a sample of test CVEs."""
    from captum.metrics import infidelity, sensitivity_max

    ec = cfg["explain"]
    rows, examples = [], []
    for m in cfg["metrics"]:
        mm = MetricModel(run, m, cfg)
        t0 = time.time()
        for j, (_, r) in enumerate(sample.iterrows()):
            enc = mm.encode(r.text, keywords.get(r.ID, []))
            probs = mm.predict(enc)
            target = int(probs.argmax())
            rec = {"model": label, "metric": m.upper(), "ID": r.ID,
                   "correct": mm.labels[target] == r[m]}
            for kind in ec["baselines"]:
                attr, emb = mm.integrated_gradients(enc, target, kind, ec["ig_steps"])
                tok_attr = attr.sum(-1)[0].detach().cpu().numpy()
                words = mm.word_attributions(enc, tok_attr)
                top = top_words(words, ec["top_k"])
                rec[f"noise_rate_{kind}"] = np.mean([is_noise(w["word"]) for w in top])
                rec[f"top_in_lexicon_{kind}"] = np.mean([w["in_lexicon"] for w in top])
                order = [w["tokens"] for w in sorted(words, key=lambda d: -d["attr"])]
                rec[f"deletion_auc_{kind}"] = auc(deletion_curve(mm, enc, order, target, ec["deletion_steps"]))
                if kind == ec["baselines"][0]:
                    rnd = [order[i] for i in rng.permutation(len(order))]
                    rec["deletion_auc_random"] = auc(deletion_curve(mm, enc, rnd, target, ec["deletion_steps"]))

                    def fwd(e, mask=enc["mask"], w=enc["w"]):
                        return mm.forward_embeds(e, mask, w)

                    def perturb(e):
                        noise = torch.randn_like(e) * 0.01
                        return noise, e - noise

                    with torch.no_grad():
                        rec["infidelity"] = float(infidelity(fwd, perturb, emb, attr, target=target,
                                                             n_perturb_samples=ec["infidelity_perturbations"]))
                    if j < n_sens:
                        from captum.attr import IntegratedGradients

                        ig = IntegratedGradients(fwd)
                        rec["sensitivity"] = float(sensitivity_max(
                            ig.attribute, emb, perturb_radius=ec["sensitivity_radius"],
                            n_perturb_samples=ec["sensitivity_perturbations"], target=target,
                            baselines=mm.baseline(emb, kind), n_steps=ec["ig_steps"] // 2,
                            internal_batch_size=ec["ig_steps"] // 2))
                    if j < 5:
                        examples.append({"model": label, "metric": m.upper(), "ID": r.ID, "text": r.text,
                                         "true": r[m], "pred": mm.labels[target], "baseline": kind,
                                         "top10": [(w["word"], round(w["attr"], 4), w["in_lexicon"]) for w in top]})
            rows.append(rec)
        print(f"  {label} [{m.upper()}] {len(sample)} CVEs in {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), examples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tw", required=True, help="ThreatWeigh run name (needs --save-model weights)")
    ap.add_argument("--base", nargs="+", required=True, help="baseline run name(s)")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--n-sensitivity", type=int, default=50, help="sensitivity is costly: first N CVEs only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    torch.set_grad_enabled(True)
    from src.data import load_split
    from src.keybert_cache import load_keywords

    n = args.n_samples or cfg["explain"]["n_samples"]
    sample = load_split("test", cfg).sample(n=n, random_state=cfg["seed"]).reset_index(drop=True)
    keywords = load_keywords(cfg)
    rng = np.random.default_rng(cfg["seed"])
    frames, examples = [], []
    for run, label in [(args.tw, "ThreatWeigh")] + [(b, "baseline:" + b.split("_")[0]) for b in args.base]:
        df, ex = quantitative(run, label, sample, keywords, cfg, args.n_sensitivity, rng)
        frames.append(df)
        examples += ex
    per = pd.concat(frames, ignore_index=True)
    res = project_path(cfg["paths"]["results_dir"])
    per.to_csv(res / f"explain_per_cve{args.tag}.csv", index=False)
    cols = [c for c in per.columns if c.startswith(("noise_rate", "top_in_lexicon", "deletion_auc", "infidelity",
                                                     "sensitivity"))]
    summary = per.groupby("model")[cols].mean().T
    by_metric = per.groupby(["model", "metric"])[cols].mean().round(4)
    summary.to_csv(res / f"explain_summary{args.tag}.csv")
    by_metric.to_csv(res / f"explain_by_metric{args.tag}.csv")
    pd.DataFrame(examples).to_json(res / f"explain_examples{args.tag}.json", orient="records", indent=1)
    print("\nMean over metrics and CVEs (lower is better for every row except top_in_lexicon):")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
