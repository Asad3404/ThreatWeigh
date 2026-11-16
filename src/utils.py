"""
Small helpers shared by every ThreatWeigh script.

* load_config()  - read config.yaml and apply a profile (quick / budget / full)
* set_seed()     - make runs reproducible (python, numpy, torch)
* get_device()   - pick the GPU automatically if there is one, else the CPU
* project_path() - turn a relative path from the config into an absolute one
"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

# The project root is the folder that contains config.yaml (one level above src/).
ROOT = Path(__file__).resolve().parent.parent

# Full names of the 8 CVSS Base metrics and the label letters each can take.
# The order of the letters is the order of the classifier outputs.
METRIC_NAMES = {
    "av": "Attack Vector",
    "ac": "Attack Complexity",
    "pr": "Privileges Required",
    "ui": "User Interaction",
    "s": "Scope",
    "c": "Confidentiality",
    "i": "Integrity",
    "a": "Availability",
}
LABELS = {
    "av": ["N", "A", "L", "P"],
    "ac": ["L", "H"],
    "pr": ["N", "L", "H"],
    "ui": ["N", "R"],
    "s": ["U", "C"],
    "c": ["N", "L", "H"],
    "i": ["N", "L", "H"],
    "a": ["N", "L", "H"],
}
# Human-readable label names (used in the app and in lexicon files).
LABEL_NAMES = {
    "av": {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"},
    "ac": {"L": "Low", "H": "High"},
    "pr": {"N": "None", "L": "Low", "H": "High"},
    "ui": {"N": "None", "R": "Required"},
    "s": {"U": "Unchanged", "C": "Changed"},
    "c": {"N": "None", "L": "Low", "H": "High"},
    "i": {"N": "None", "L": "Low", "H": "High"},
    "a": {"N": "None", "L": "Low", "H": "High"},
}


def project_path(rel: str | os.PathLike) -> Path:
    """Return an absolute path for a path written relative to the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def load_config(profile: str | None = None, path: str | None = None) -> dict:
    """Load config.yaml. If a profile name is given (e.g. 'quick'), the values
    in config['profiles'][profile] overwrite the matching keys in config['train']."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["profile"] = profile or "full"
    if profile and profile != "full":
        if profile not in cfg.get("profiles", {}):
            raise ValueError(f"Unknown profile '{profile}'. Options: {list(cfg['profiles'])} or 'full'")
        cfg["train"].update(cfg["profiles"][profile])
    return cfg


def set_seed(seed: int) -> None:
    """Seed every random number generator we use so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the TF-IDF baseline / cvss tests
        pass


def get_device():
    """Use the first CUDA GPU if available, otherwise Apple MPS, otherwise CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(project_path(path)) as f:
        return json.load(f)
