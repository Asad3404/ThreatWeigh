"""
Stage 3b - KeyBERT keywords (K_mdl), computed once and cached to disk.

Run:
    python -m src.keybert_cache            # every CVE in the splits (~137k)
    python -m src.keybert_cache --limit 1000   # quick benchmark

Settings follow the paper:
  * the SAME Sentence-BERT checkpoint as the encoder (config model.encoder),
    used as-is (not fine-tuned) because keywords are computed BEFORE training
  * keyphrase_ngram_range = (1, 1)   -> single words only
  * top_n = 5
  * stop_words = None                -> no stop-word filtering
  * use_mmr = False                  -> plain cosine ranking, no diversification

Output: data/keybert/keywords_<encoder>.json  {CVE-ID: [kw1, ..., kw5]}

Note on equivalence: KeyBERT ranks the candidate words OF EACH DOCUMENT by
cosine similarity to that document's embedding. Passing documents in chunks
only changes which vocabulary is embedded at once, not the result per doc.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from tqdm import tqdm

from src.utils import get_device, load_config, load_json, project_path, save_json, set_seed


def cache_path(cfg: dict):
    name = cfg["model"]["encoder"].split("/")[-1]
    return project_path(cfg["paths"]["keybert_dir"]) / f"keywords_{name}.json"


def load_keywords(cfg: dict) -> dict[str, list[str]]:
    p = cache_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m src.keybert_cache` first.")
    return load_json(p)


def extract(texts: list[str], cfg: dict, chunk: int = 2000) -> list[list[str]]:
    """Run KeyBERT on a list of texts, returning top-n words per text."""
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    kb = cfg["keybert"]
    st = SentenceTransformer(cfg["model"]["encoder"], device=str(get_device()))
    kw_model = KeyBERT(model=st)
    out: list[list[str]] = []
    for start in tqdm(range(0, len(texts), chunk), desc="KeyBERT chunks"):
        docs = texts[start:start + chunk]
        res = kw_model.extract_keywords(
            docs,
            keyphrase_ngram_range=tuple(kb["ngram_range"]),
            stop_words=kb["stop_words"],
            top_n=kb["top_n"],
            use_mmr=kb["use_mmr"],
        )
        if len(docs) == 1:          # KeyBERT returns a flat list for a single doc
            res = [res]
        out.extend([w for w, _score in r] for r in res)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N rows (benchmark)")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    from src.data import load_split

    df = pd.concat([load_split(s, cfg) for s in ("train", "val", "test")], ignore_index=True)
    path = cache_path(cfg)
    cache = load_json(path) if path.exists() else {}
    todo = df[~df.ID.isin(cache)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(cache)} cached, {len(todo)} to compute with {cfg['model']['encoder']}")
    t0 = time.time()
    if len(todo):
        kws = extract(todo.text.tolist(), cfg)
        cache.update(dict(zip(todo.ID, kws)))
        save_json(cache, path)
    dt = time.time() - t0
    print(f"done in {dt:.0f}s ({len(todo) / max(dt, 1e-9):.0f} docs/s) -> {path}")
    for _, r in todo.head(3).iterrows():
        print(f"\n{r.ID}: {r.text[:150]}...\n  keywords: {cache[r.ID]}")


if __name__ == "__main__":
    main()
