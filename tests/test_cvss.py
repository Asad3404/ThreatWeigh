"""Unit tests for the CVSS v3.1 calculator (src/cvss.py)."""
import pandas as pd
import pytest

from src.cvss import base_score, parse_vector, roundup, severity, vector_string
from src.utils import project_path


@pytest.mark.parametrize(
    "vector, expected",
    [
        # The two vectors from the project brief.
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
        # NOTE: the brief expected 4.6 for this vector, but the official
        # value is 4.3: ISS = 1-0.78^3 = 0.5254, Impact = 6.42*ISS = 3.373,
        # Exploitability = 8.22*0.2*0.77*0.85*0.85 = 0.915, sum 4.288 -> 4.3.
        # NVD itself scores this exact vector 4.3 (9 CVEs in final.csv).
        ("CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L", 4.3),
        # The physical-vector case that NVD scores 4.6 (198 CVEs in final.csv)
        ("CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N", 4.6),
        # Extra cases checked against the FIRST online calculator
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),   # typical XSS
        ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8),   # local priv-esc
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5),   # remote DoS
        ("CVSS:3.1/AV:N/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N", 2.0),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0),   # no impact -> 0
    ],
)
def test_known_vectors(vector, expected):
    assert base_score(parse_vector(vector)) == expected


def test_roundup_official_behaviour():
    assert roundup(4.0) == 4.0
    assert roundup(4.02) == 4.1
    assert roundup(4.000000001) == 4.0   # floating-point noise must NOT round up
    assert roundup(9.71) == 9.8


def test_severity_bands():
    assert [severity(s) for s in (0.0, 0.1, 3.9, 4.0, 6.9, 7.0, 8.9, 9.0, 10.0)] == [
        "None", "Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]


def test_vector_roundtrip():
    v = "CVSS:3.1/AV:A/AC:H/PR:L/UI:R/S:C/C:N/I:L/A:H"
    assert vector_string(parse_vector(v)) == v
    assert base_score({"AV": "n", "AC": "l", "PR": "n", "UI": "n", "S": "u", "C": "h", "I": "h", "A": "h"}) == 9.8


def test_matches_nvd_scores_on_dataset():
    """Every CVSS:3.1 row of the NVD dataset must reproduce NVD's own score."""
    path = project_path("data/raw/final.csv")
    if not path.exists():
        pytest.skip("dataset not downloaded")
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df.vectorString.str.startswith("CVSS:3.1")]
    ours = df.vectorString.map(lambda v: base_score(parse_vector(v)))
    mismatch = (ours - df.score).abs() > 1e-9
    assert mismatch.sum() == 0, df[mismatch].head()
