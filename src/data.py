"""
Stage 1 - data download, cleaning, splitting and a small EDA.

Run:
    python -m src.data            # download (if needed), clean, split, print EDA

What it does
------------
1. Gets `final.csv` (137,521 NVD CVEs with CVSS v3.x vectors) from the
   WSymBERT repository (Nvd/Dataset/final.zip). Note: the repo's
   combine/Dataset/final.zip is byte-identical, so there is only ONE dataset.
2. Drops rows with missing text.
3. Strict de-duplication on the text: texts are lower-cased and whitespace
   is collapsed before comparing, and only the first copy is kept.
4. Stratified 80/10/10 train/val/test split on the Attack Vector label with
   a fixed seed, saved to data/splits/ so EVERY model uses identical splits.
5. A time-aware split (older CVEs -> train, newer -> test) for the
   temporal-generalisation experiment, using the year in the CVE ID.
6. Prints label distributions per metric and split.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import LABELS, load_config, project_path, set_seed

COLUMNS = ["ID", "text", "vectorString", "av", "ac", "pr", "ui", "s", "c", "i", "a", "score", "baseSeverity"]


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def ensure_raw_csv(cfg: dict) -> None:
    """Make sure data/raw/final.csv exists. Clone the WSymBERT repo if needed."""
    raw_csv = project_path(cfg["paths"]["raw_csv"])
    raw_zip = project_path(cfg["paths"]["raw_zip"])
    if raw_csv.exists():
        return
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if not raw_zip.exists():
        print(f"Cloning {cfg['paths']['wsymbert_repo']} (about 94 MB) ...")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "clone", "--depth", "1", cfg["paths"]["wsymbert_repo"], tmp + "/repo"], check=True)
            shutil.copy(tmp + "/repo/Nvd/Dataset/final.zip", raw_zip)
    with zipfile.ZipFile(raw_zip) as z:
        # Only extract the one file we need (never trust paths inside a zip).
        with z.open("final.csv") as src, open(raw_csv, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"Extracted {raw_csv}")


# ---------------------------------------------------------------------------
# 2-3. Cleaning
# ---------------------------------------------------------------------------
def normalise_text(s: pd.Series) -> pd.Series:
    """Lower-case and collapse whitespace; used only to detect duplicates."""
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0]
    n1 = len(df)
    key = normalise_text(df["text"])
    df = df[~key.duplicated(keep="first")].reset_index(drop=True)
    n2 = len(df)
    # The year in the CVE ID is used for the temporal split.
    df["year"] = df["ID"].str.extract(r"CVE-(\d{4})-")[0].astype(int)
    print(f"Rows: {n0} raw -> {n1} after dropping empty text -> {n2} after strict de-duplication "
          f"({n1 - n2} duplicates removed)")
    # sanity: every label must be one of the expected letters
    for m, labs in LABELS.items():
        bad = ~df[m].isin(labs)
        assert not bad.any(), f"unexpected labels in {m}: {df.loc[bad, m].unique()}"
    return df


# ---------------------------------------------------------------------------
# 4. Random stratified split
# ---------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, cfg: dict):
    seed = cfg["seed"]
    strat = cfg["data"]["stratify_on"]
    test_frac, val_frac = cfg["data"]["test_frac"], cfg["data"]["val_frac"]
    train_val, test = train_test_split(df, test_size=test_frac, stratify=df[strat], random_state=seed)
    # val is val_frac of the WHOLE dataset -> rescale for the remaining part
    val_rel = val_frac / (1.0 - test_frac)
    train, val = train_test_split(train_val, test_size=val_rel, stratify=train_val[strat], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Temporal split
# ---------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, cfg: dict):
    t = cfg["data"]["temporal"]
    train = df[df.year <= t["train_until_year"]]
    val = df[df.year == t["val_year"]]
    test = df[df.year >= t["test_from_year"]]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loading helpers used by every other module
# ---------------------------------------------------------------------------
def load_split(name: str, cfg: dict, split_kind: str = "random") -> pd.DataFrame:
    """Load 'train' / 'val' / 'test'. split_kind='temporal' loads temporal_*.csv."""
    prefix = "" if split_kind == "random" else "temporal_"
    path = project_path(cfg["paths"]["splits_dir"]) / f"{prefix}{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data` first.")
    return pd.read_csv(path, keep_default_na=False)


def subsample(df: pd.DataFrame, n: int | None, seed: int, stratify_col: str | None = "av") -> pd.DataFrame:
    """Return a stratified random subsample of n rows (or df unchanged if n is None)."""
    if n is None or n >= len(df):
        return df
    if stratify_col:
        sub, _ = train_test_split(df, train_size=n, stratify=df[stratify_col], random_state=seed)
    else:
        sub = df.sample(n=n, random_state=seed)
    return sub.reset_index(drop=True)


def label_distribution(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Table with one row per (metric, label) and the % in each split."""
    rows = []
    for m, labs in LABELS.items():
        for lab in labs:
            row = {"metric": m.upper(), "label": lab}
            for name, d in splits.items():
                row[f"{name}_n"] = int((d[m] == lab).sum())
                row[f"{name}_%"] = round(100 * (d[m] == lab).mean(), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_raw_csv(cfg)
    df = pd.read_csv(project_path(cfg["paths"]["raw_csv"]), keep_default_na=False)
    assert list(df.columns) == COLUMNS, df.columns
    df = clean(df)

    out = project_path(cfg["paths"]["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = stratified_split(df, cfg)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_csv(out / f"{name}.csv", index=False)
    print(f"Random split   : train={len(train)}  val={len(val)}  test={len(test)}")

    ttrain, tval, ttest = temporal_split(df, cfg)
    for name, d in [("train", ttrain), ("val", tval), ("test", ttest)]:
        d.to_csv(out / f"temporal_{name}.csv", index=False)
    print(f"Temporal split : train={len(ttrain)} (<= {cfg['data']['temporal']['train_until_year']})  "
          f"val={len(tval)}  test={len(ttest)}")

    # ---- EDA -------------------------------------------------------------
    dist = label_distribution({"train": train, "val": val, "test": test})
    res = project_path(cfg["paths"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    dist.to_csv(res / "label_distribution.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nLabel distribution per metric (% of each split):")
    print(dist[["metric", "label", "train_n", "train_%", "val_%", "test_%"]].to_string(index=False))
    words = df.text.str.split().str.len()
    print(f"\nWords per description: mean={words.mean():.1f} median={words.median():.0f} "
          f"p95={words.quantile(.95):.0f} max={words.max()}")
    print("CVEs per year:", df.year.value_counts().sort_index().to_dict())
    print("CVSS version in vectorString:", df.vectorString.str.slice(0, 8).value_counts().to_dict())


if __name__ == "__main__":
    main()
