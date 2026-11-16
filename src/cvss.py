"""
CVSS v3.1 Base Score calculator (FIRST specification, section 7 and Appendix A).

    from src.cvss import base_score, severity
    base_score({"av": "N", "ac": "L", "pr": "N", "ui": "N",
                "s": "U", "c": "H", "i": "H", "a": "H"})   # -> 9.8

The formula
-----------
    ISS            = 1 - (1 - C) * (1 - I) * (1 - A)
    Impact (S:U)   = 6.42 * ISS
    Impact (S:C)   = 7.52 * (ISS - 0.029) - 3.25 * (ISS - 0.02) ** 15
    Exploitability = 8.22 * AV * AC * PR * UI
    if Impact <= 0           : Base = 0
    elif Scope is Unchanged  : Base = Roundup(min(Impact + Exploitability, 10))
    else (Scope Changed)     : Base = Roundup(min(1.08 * (Impact + Exploitability), 10))

Roundup is the *official* v3.1 function: it works on integers to avoid
floating-point errors (e.g. 4.000000001 must round to 4.0, not 4.1).
"""
from __future__ import annotations

import math

# Numeric weights for every metric value (CVSS v3.1 spec, Table 16).
AV_W = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
AC_W = {"L": 0.77, "H": 0.44}
PR_W_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
PR_W_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.5}   # PR weighs more if Scope changes
UI_W = {"N": 0.85, "R": 0.62}
CIA_W = {"H": 0.56, "L": 0.22, "N": 0.0}

METRIC_ORDER = ["av", "ac", "pr", "ui", "s", "c", "i", "a"]


def roundup(value: float) -> float:
    """Official CVSS v3.1 Roundup: smallest number with 1 decimal that is >= value.

    Implemented exactly as in Appendix A of the specification:
        int_input = round_to_nearest_integer(input * 100000)
        if int_input % 10000 == 0: return int_input / 100000.0
        else: return (floor(int_input / 10000) + 1) / 10.0
    """
    int_input = int(round(value * 100000))
    if int_input % 10000 == 0:
        return int_input / 100000.0
    return (math.floor(int_input / 10000) + 1) / 10.0


def _norm(metrics: dict) -> dict:
    """Accept keys in any case ('AV' or 'av') and values like 'N'."""
    out = {k.lower(): str(v).upper() for k, v in metrics.items()}
    missing = [m for m in METRIC_ORDER if m not in out]
    if missing:
        raise ValueError(f"missing metrics: {missing}")
    return out


def subscores(metrics: dict) -> dict:
    """Return ISS, Impact and Exploitability (unrounded) for inspection."""
    m = _norm(metrics)
    changed = m["s"] == "C"
    iss = 1 - (1 - CIA_W[m["c"]]) * (1 - CIA_W[m["i"]]) * (1 - CIA_W[m["a"]])
    if changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss
    pr_w = (PR_W_CHANGED if changed else PR_W_UNCHANGED)[m["pr"]]
    exploitability = 8.22 * AV_W[m["av"]] * AC_W[m["ac"]] * pr_w * UI_W[m["ui"]]
    return {"iss": iss, "impact": impact, "exploitability": exploitability}


def base_score(metrics: dict) -> float:
    """CVSS v3.1 Base Score (0.0 - 10.0) from the 8 metric letters."""
    m = _norm(metrics)
    s = subscores(m)
    if s["impact"] <= 0:
        return 0.0
    if m["s"] == "U":
        return roundup(min(s["impact"] + s["exploitability"], 10))
    return roundup(min(1.08 * (s["impact"] + s["exploitability"]), 10))


def severity(score: float) -> str:
    """Qualitative severity rating (CVSS v3.1 spec, Table 14)."""
    if score == 0.0:
        return "None"
    if score < 4.0:
        return "Low"
    if score < 7.0:
        return "Medium"
    if score < 9.0:
        return "High"
    return "Critical"


def vector_string(metrics: dict) -> str:
    """Build 'CVSS:3.1/AV:N/AC:L/...' from the 8 metric letters."""
    m = _norm(metrics)
    return "CVSS:3.1/" + "/".join(f"{k.upper()}:{m[k]}" for k in METRIC_ORDER)


def parse_vector(vector: str) -> dict:
    """Parse 'CVSS:3.1/AV:N/AC:L/...' (or 3.0) into {'av': 'N', ...}."""
    parts = vector.strip().split("/")
    if parts[0].startswith("CVSS:"):
        parts = parts[1:]
    out = {}
    for p in parts:
        k, v = p.split(":")
        out[k.lower()] = v.upper()
    return {k: out[k] for k in METRIC_ORDER}


if __name__ == "__main__":
    for v in ["CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
              "CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L"]:
        s = base_score(parse_vector(v))
        print(f"{v} -> {s} ({severity(s)})")
