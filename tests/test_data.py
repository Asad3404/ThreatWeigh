"""Tests for the saved data splits: no leakage, correct sizes, stratification."""
import pandas as pd
import pytest

from src.data import load_split, normalise_text
from src.utils import LABELS, load_config

cfg = load_config()


@pytest.fixture(scope="module")
def splits():
    try:
        return {n: load_split(n, cfg) for n in ("train", "val", "test")}
    except FileNotFoundError:
        pytest.skip("splits not created - run `python -m src.data`")


def test_no_id_overlap(splits):
    ids = {n: set(d.ID) for n, d in splits.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_no_text_overlap_after_normalisation(splits):
    """Strict dedup means no (normalised) description appears in two splits."""
    txt = {n: set(normalise_text(d.text)) for n, d in splits.items()}
    assert not txt["train"] & txt["test"]
    assert not txt["train"] & txt["val"]
    assert not txt["val"] & txt["test"]


def test_no_duplicates_within_split(splits):
    for d in splits.values():
        assert not normalise_text(d.text).duplicated().any()


def test_split_proportions(splits):
    n = sum(len(d) for d in splits.values())
    assert abs(len(splits["test"]) / n - cfg["data"]["test_frac"]) < 0.005
    assert abs(len(splits["val"]) / n - cfg["data"]["val_frac"]) < 0.005


def test_stratified_on_av(splits):
    p = {k: d.av.value_counts(normalize=True) for k, d in splits.items()}
    for lab in LABELS["av"]:
        assert abs(p["train"][lab] - p["test"][lab]) < 0.002
        assert abs(p["train"][lab] - p["val"][lab]) < 0.002


def test_temporal_split_is_ordered():
    try:
        tr, te = load_split("train", cfg, "temporal"), load_split("test", cfg, "temporal")
    except FileNotFoundError:
        pytest.skip("temporal splits not created")
    assert tr.year.max() < te.year.min()
    assert not set(tr.ID) & set(te.ID)
