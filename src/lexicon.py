"""
Stage 3a - Expert lexicons (K_exp) built from CVSS v3.1 decision trees.

Run:
    python -m src.lexicon          # build, filter on TRAIN only, freeze, report coverage

Idea
----
A human analyst scoring a CVE with the FIRST CVSS v3.1 specification asks
a short series of questions for each metric, e.g. for Attack Vector:

    Q1 "Is physical contact with the device required?"          -> AV:P
    Q2 "Must the attacker be on the same local/radio network?"  -> AV:A
    Q3 "Does the attacker need a local account / local access?" -> AV:L
    Q4 "Otherwise: can it be exploited remotely?"               -> AV:N

Every answer is associated with typical wording found in CVE descriptions
("physical access", "bluetooth", "local users", "remote attackers", ...).
These wordings are our *seed cues*. They are written by hand from the
specification, NOT mined from the data.

Training-set check (no test data is ever used)
----------------------------------------------
A seed cue is kept for (metric, label) only if, in the TRAINING split,
  * it occurs in >= `MIN_SUPPORT` descriptions, and
  * lift = P(label | cue) / P(label) >= `MIN_LIFT`
i.e. the cue really points towards that label in practice. The resulting
lexicon is written to lexicons/expert_lexicon.json and frozen.

Coverage
--------
The paper reports that 83.6% of 2,000 CVEs had at least one lexicon match.
We report the same statistic (per metric and "any metric") on the test split.
"""
from __future__ import annotations

import argparse
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.utils import LABEL_NAMES, load_config, load_json, project_path, save_json

MIN_SUPPORT = 10   # cue must appear in at least this many TRAIN descriptions
MIN_LIFT = 1.2     # P(label|cue) must be >= 1.2 x the base rate of the label

# ---------------------------------------------------------------------------
# Decision trees. One per metric. Each node = one question from the spec,
# the label a "yes" leads to, and the seed cue phrases (lower-case).
# The order of nodes is the order an analyst would ask the questions.
# ---------------------------------------------------------------------------
TREES: dict[str, list[dict]] = {
    "av": [
        {"q": "Does exploitation require the attacker to physically touch or manipulate the device?",
         "label": "P",
         "cues": ["physical access", "physically proximate", "physical proximity", "physical attacker",
                  "physically present", "usb", "jtag", "uart", "serial console", "debug port",
                  "lock screen", "lockscreen", "lock-screen", "unlocked device", "evil maid",
                  "boot", "bootloader", "sd card", "hardware"]},
        {"q": "Is the attack limited to the same local / radio network segment (logically adjacent)?",
         "label": "A",
         "cues": ["adjacent network", "adjacent attacker", "adjacent", "same network segment", "same subnet",
                  "local network", "local area network", "lan", "bluetooth", "wi-fi", "wifi", "wireless",
                  "wlan", "zigbee", "802.11", "within radio range", "radio range", "same broadcast domain",
                  "layer 2", "arp", "dhcp", "nfc", "proximity"]},
        {"q": "Does the attacker need local access (a local account, a shell, or a user opening a file)?",
         "label": "L",
         "cues": ["local user", "local users", "local attacker", "local attackers", "locally",
                  "local access", "local privilege", "requires local access", "logged-on user",
                  "authenticated local", "local authenticated", "crafted file", "malicious file",
                  "open a crafted", "opening a crafted", "privilege escalation", "escalate privileges",
                  "elevation of privilege", "elevate privileges", "kernel", "driver", "setuid",
                  "symlink", "symbolic link", "local system", "dll", "uncontrolled search path",
                  "physical memory", "ioctl", "android"]},
        {"q": "Otherwise: can it be exploited remotely over a network or the internet?",
         "label": "N",
         "cues": ["remote attacker", "remote attackers", "remotely", "remote", "over the network",
                  "via the network", "network access", "the internet", "unauthenticated remote",
                  "http request", "http requests", "crafted request", "crafted http", "web request",
                  "crafted url", "url", "crafted packet", "crafted packets", "crafted web site",
                  "web page", "web interface", "website", "cross-site scripting", "xss",
                  "sql injection", "cross-site request forgery", "csrf", "server-side request forgery",
                  "ssrf", "open redirect", "api", "parameter", "plugin", "wordpress", "php"]},
    ],
    "ac": [
        {"q": "Does success depend on conditions beyond the attacker's control "
              "(race, man-in-the-middle, special configuration, information gathering)?",
         "label": "H",
         "cues": ["race condition", "man-in-the-middle", "man in the middle", "mitm", "timing attack",
                  "side-channel", "side channel", "brute force", "brute-force", "non-default",
                  "specific configuration", "certain configurations", "certain conditions",
                  "specific conditions", "in a position to", "intercept", "predict", "guess",
                  "toctou", "time-of-check", "padding oracle", "cache", "cryptographic",
                  "when configured", "is enabled", "if the", "difficult", "complex", "spoof",
                  "privileged network position", "downgrade"]},
        {"q": "Otherwise: can the attack be repeated at will with no special conditions?",
         "label": "L",
         "cues": ["easily", "trivially", "default configuration", "default installation",
                  "no special", "reliably", "any user", "sql injection", "cross-site scripting", "xss",
                  "unauthenticated", "parameter"]},
    ],
    "pr": [
        {"q": "Must the attacker already hold administrative / high privileges?",
         "label": "H",
         "cues": ["administrator", "administrators", "administrative", "admin", "admin privileges",
                  "administrator privileges", "administrative privileges", "high privileges",
                  "high-privileged", "highly privileged", "privileged user", "privileged users",
                  "superuser", "root privileges", "root access", "site administrator",
                  "system administrator", "elevated privileges", "privileged account", "admin panel",
                  "admin account", "authenticated admin", "authenticated administrator"]},
        {"q": "Must the attacker be an ordinary authenticated / logged-in user?",
         "label": "L",
         "cues": ["authenticated user", "authenticated users", "remote authenticated users",
                  "authenticated attacker", "authenticated attackers", "authenticated", "low-privileged",
                  "low privileged", "low privilege", "low-privilege", "logged-in user", "logged in user",
                  "registered user", "registered users", "local user", "local users", "subscriber",
                  "contributor", "standard user", "unprivileged user", "unprivileged", "normal user",
                  "user account", "valid credentials", "valid account", "with access to", "local attacker"]},
        {"q": "Otherwise: can an anonymous, unauthenticated attacker do it?",
         "label": "N",
         "cues": ["unauthenticated", "unauthenticated attacker", "unauthenticated remote",
                  "without authentication", "does not require authentication", "no authentication",
                  "missing authentication", "pre-authentication", "pre-auth", "anonymous",
                  "authentication bypass", "bypass authentication", "without logging in",
                  "remote attackers", "remote attacker", "cross-site request forgery", "csrf",
                  "cross-site scripting", "xss", "crafted web site", "crafted file", "crafted packet"]},
    ],
    "ui": [
        {"q": "Must a user other than the attacker take some action (click, open, visit)?",
         "label": "R",
         "cues": ["user interaction is needed", "user interaction is required", "requires user interaction",
                  "convince a user", "trick a user", "tricking a user", "victim", "click", "clicks",
                  "clicking", "visit", "visiting", "visits", "open a crafted", "opens a crafted",
                  "malicious link", "crafted link", "crafted web site", "crafted website", "web page",
                  "cross-site scripting", "xss", "cross-site request forgery", "csrf", "phishing",
                  "social engineering", "persuade", "entice", "crafted document", "crafted file",
                  "malicious file", "attachment", "open redirect", "clickjacking", "user to open",
                  "user opens", "stored xss", "reflected", "inject arbitrary web script", "browser"]},
        {"q": "Otherwise: can it be exploited without any action from another user?",
         "label": "N",
         "cues": ["user interaction is not needed", "without user interaction", "no user interaction",
                  "does not require user interaction", "zero-click", "unauthenticated", "sql injection",
                  "command injection", "os command injection", "crafted packet", "crafted request",
                  "http request", "denial of service", "privilege escalation", "local users",
                  "path traversal", "directory traversal", "buffer overflow", "kernel"]},
    ],
    "s": [
        {"q": "Can the impact cross a security boundary into another component "
              "(sandbox/VM/container escape, script running in other users' browsers, SSRF)?",
         "label": "C",
         "cues": ["cross-site scripting", "xss", "inject arbitrary web script", "web script or html",
                  "arbitrary web script", "stored xss", "reflected xss", "sandbox escape",
                  "escape the sandbox", "sandbox", "virtual machine escape", "guest", "hypervisor",
                  "container escape", "escape the container", "guest os", "host os",
                  "server-side request forgery", "ssrf", "other users", "victim's browser",
                  "trusted execution environment", "trustzone", "secure world", "javascript"]},
        {"q": "Otherwise: does the impact stay inside the vulnerable component?",
         "label": "U",
         "cues": ["in the context of the current user", "the affected application", "application crash",
                  "sql injection", "denial of service", "buffer overflow", "command injection",
                  "information disclosure", "privilege escalation", "local users", "kernel"]},
    ],
    "c": [
        {"q": "Can the attacker read all / any restricted data (total loss of confidentiality)?",
         "label": "H",
         "cues": ["execute arbitrary code", "arbitrary code execution", "remote code execution",
                  "code execution", "execute arbitrary commands", "arbitrary commands", "command injection",
                  "sql injection", "read arbitrary files", "arbitrary file read", "full control",
                  "take over", "takeover", "complete compromise", "gain root", "gain privileges",
                  "escalate privileges", "privilege escalation", "elevation of privilege", "credentials",
                  "password", "passwords", "private key", "decrypt", "plaintext", "cleartext",
                  "sensitive data", "database", "hard-coded", "hardcoded", "authentication bypass",
                  "use-after-free", "out-of-bounds write", "memory corruption"]},
        {"q": "Can the attacker read only some restricted data, without control over what?",
         "label": "L",
         "cues": ["information disclosure", "disclose", "obtain sensitive information",
                  "information leak", "information exposure", "leak", "enumerate", "enumeration",
                  "username enumeration", "path disclosure", "partial", "limited information",
                  "obtain information", "cross-site scripting", "xss", "inject arbitrary web script",
                  "version", "metadata", "open redirect", "out-of-bounds read"]},
        {"q": "Otherwise: is there no confidentiality impact (e.g. pure availability bug)?",
         "label": "N",
         "cues": ["denial of service", "crash", "infinite loop", "resource consumption",
                  "memory exhaustion", "resource exhaustion", "hang", "dos", "reboot",
                  "null pointer dereference", "assertion failure", "cpu consumption",
                  "cross-site request forgery", "csrf", "uncontrolled recursion", "panic"]},
    ],
    "i": [
        {"q": "Can the attacker modify any / all data or run code (total loss of integrity)?",
         "label": "H",
         "cues": ["execute arbitrary code", "remote code execution", "code execution", "arbitrary commands",
                  "command injection", "sql injection", "write arbitrary files", "arbitrary file write",
                  "upload arbitrary files", "arbitrary file upload", "overwrite", "delete arbitrary files",
                  "privilege escalation", "escalate privileges", "gain privileges", "full control",
                  "take over", "out-of-bounds write", "buffer overflow", "use-after-free",
                  "memory corruption", "heap overflow", "cross-site request forgery", "csrf",
                  "authentication bypass"]},
        {"q": "Can the attacker modify only some data, with limited consequence?",
         "label": "L",
         "cues": ["cross-site scripting", "xss", "inject arbitrary web script", "html injection",
                  "web script or html", "open redirect", "redirect users", "spoof", "spoofing",
                  "content spoofing", "log injection", "header injection", "crlf injection",
                  "modify some", "tamper"]},
        {"q": "Otherwise: is there no integrity impact (read-only or availability-only bug)?",
         "label": "N",
         "cues": ["denial of service", "crash", "information disclosure", "disclose",
                  "obtain sensitive information", "read arbitrary files", "out-of-bounds read",
                  "information leak", "information exposure", "enumerate", "resource exhaustion",
                  "infinite loop", "null pointer dereference", "memory leak", "hang", "dos"]},
    ],
    "a": [
        {"q": "Can the attacker fully deny access to the component (crash, hang, exhaustion)?",
         "label": "H",
         "cues": ["denial of service", "crash", "reboot", "infinite loop", "resource exhaustion",
                  "memory exhaustion", "cpu consumption", "hang", "out of memory", "execute arbitrary code",
                  "remote code execution", "code execution", "buffer overflow", "use-after-free",
                  "null pointer dereference", "segmentation fault", "assertion failure", "kernel panic",
                  "panic", "unavailable", "stack exhaustion", "uncontrolled recursion", "memory leak",
                  "dos", "command injection", "out-of-bounds write", "memory corruption"]},
        {"q": "Is availability only reduced or interrupted partially?",
         "label": "L",
         "cues": ["performance degradation", "partial denial of service", "degrade", "degradation",
                  "slow down", "slowdown", "temporary", "temporarily", "limited denial",
                  "reduced performance", "some resources", "rate limit", "log flooding"]},
        {"q": "Otherwise: is there no availability impact (read / modify data only)?",
         "label": "N",
         "cues": ["information disclosure", "cross-site scripting", "xss", "obtain sensitive information",
                  "disclose", "open redirect", "spoof", "spoofing", "information leak", "information exposure",
                  "enumerate", "inject arbitrary web script", "csrf", "cross-site request forgery",
                  "out-of-bounds read", "path disclosure"]},
    ],
}


# ---------------------------------------------------------------------------
# Phrase matching
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def phrase_regex(phrase: str) -> re.Pattern:
    """Case-insensitive regex that matches the phrase as whole word(s).
    `(?<![\\w-])` / `(?![\\w-])` act like word boundaries but also treat '-'
    as part of a word, so 'dos' does not match inside 'non-dos-thing'."""
    return re.compile(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", re.IGNORECASE)


def find_spans(text: str, phrases) -> list[tuple[int, int]]:
    """Character spans (start, end) of every occurrence of every phrase in text."""
    spans = []
    for p in phrases:
        spans.extend(m.span() for m in phrase_regex(p).finditer(text))
    return spans


def contains_any(texts: pd.Series, phrases) -> np.ndarray:
    """Boolean array: does each text contain at least one of the phrases?"""
    if not phrases:
        return np.zeros(len(texts), dtype=bool)
    big = re.compile("|".join(f"(?:{phrase_regex(p).pattern})" for p in sorted(phrases, key=len, reverse=True)),
                     re.IGNORECASE)
    return texts.str.contains(big, regex=True).to_numpy()


# ---------------------------------------------------------------------------
# Building / loading
# ---------------------------------------------------------------------------
def build_lexicon(train: pd.DataFrame) -> dict:
    """Filter the seed cues of every tree using the TRAINING split only."""
    lex = {"_meta": {"min_support": MIN_SUPPORT, "min_lift": MIN_LIFT,
                     "built_on": f"train split only ({len(train)} rows)",
                     "source": "hand-written questions from the FIRST CVSS v3.1 specification"},
           "metrics": {}}
    texts = train["text"]
    for metric, tree in TREES.items():
        nodes = []
        for node in tree:
            lab = node["label"]
            base = float((train[metric] == lab).mean())
            kept, dropped = [], []
            for cue in dict.fromkeys(node["cues"]):          # dedupe, keep order
                hit = contains_any(texts, [cue])
                support = int(hit.sum())
                prec = float((train.loc[hit, metric] == lab).mean()) if support else 0.0
                lift = prec / base if base > 0 else 0.0
                rec = {"cue": cue, "support": support, "precision": round(prec, 3), "lift": round(lift, 2)}
                (kept if support >= MIN_SUPPORT and lift >= MIN_LIFT else dropped).append(rec)
            nodes.append({"question": node["q"], "label": lab, "label_name": LABEL_NAMES[metric][lab],
                          "base_rate": round(base, 4), "cues": kept, "dropped": dropped})
        lex["metrics"][metric] = nodes
    return lex


def load_lexicon(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return load_json(cfg["paths"]["lexicon_file"])


def metric_phrases(lex: dict, metric: str) -> list[str]:
    """K_exp for one metric = every kept cue of every label of that metric."""
    out = []
    for node in lex["metrics"][metric]:
        out.extend(c["cue"] for c in node["cues"])
    return list(dict.fromkeys(out))


def phrase_to_labels(lex: dict, metric: str) -> dict[str, list[str]]:
    """Which label(s) each cue of a metric points to (used by the app)."""
    out: dict[str, list[str]] = {}
    for node in lex["metrics"][metric]:
        for c in node["cues"]:
            out.setdefault(c["cue"], []).append(node["label"])
    return out


def coverage(lex: dict, df: pd.DataFrame) -> pd.DataFrame:
    """% of descriptions with >= 1 lexicon match, per metric and for any metric."""
    rows, any_hit = [], np.zeros(len(df), dtype=bool)
    for m in lex["metrics"]:
        hit = contains_any(df["text"], metric_phrases(lex, m))
        any_hit |= hit
        rows.append({"metric": m.upper(), "n_cues": len(metric_phrases(lex, m)),
                     "coverage_%": round(100 * hit.mean(), 2)})
    rows.append({"metric": "ANY", "n_cues": sum(r["n_cues"] for r in rows),
                 "coverage_%": round(100 * any_hit.mean(), 2)})
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from src.data import load_split

    cfg = load_config()
    train, test = load_split("train", cfg), load_split("test", cfg)
    lex = build_lexicon(train)
    save_json(lex, cfg["paths"]["lexicon_file"])
    print(f"Lexicon frozen -> {cfg['paths']['lexicon_file']}\n")
    for m, nodes in lex["metrics"].items():
        for n in nodes:
            print(f"{m.upper()}:{n['label']} kept {len(n['cues']):2d} / dropped {len(n['dropped']):2d}  "
                  f"e.g. {[c['cue'] for c in sorted(n['cues'], key=lambda c: -c['lift'])[:4]]}")
    cov = coverage(lex, test)
    sample = test.sample(n=min(2000, len(test)), random_state=cfg["seed"])
    cov["coverage_2000_sample_%"] = coverage(lex, sample)["coverage_%"]
    res = project_path(cfg["paths"]["results_dir"])
    cov.to_csv(res / "lexicon_coverage.csv", index=False)
    print("\nLexicon coverage on the TEST split (% of descriptions with >= 1 match):")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
