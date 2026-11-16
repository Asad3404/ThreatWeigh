"""
ThreatWeigh command-line predictor.

    python predict.py "A remote attacker can execute arbitrary code via a crafted HTTP request."
    python predict.py --explain "..."        # also show the top attributed words per metric
    python predict.py --run <run-name> "..."  # use another trained run (default: config serve.run)

Prints the 8 CVSS v3.1 Base metrics with confidence, the vector string,
the Base Score and the severity.
"""
from __future__ import annotations

import argparse
import sys

from src.utils import LABEL_NAMES, METRIC_NAMES, load_config, project_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="+", help="the vulnerability description")
    ap.add_argument("--run", default=None)
    ap.add_argument("--explain", action="store_true", help="Integrated Gradients top words (slower)")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    run = args.run or cfg["serve"]["run"]
    if not (project_path(cfg["paths"]["models_dir"]) / run).exists():
        sys.exit(f"No trained models in models/{run}. Train first, e.g.\n"
                 f"  python -m src.train --variant full --profile budget --save-model")

    from src.explain import ThreatWeighPredictor

    text = " ".join(args.text)
    p = ThreatWeighPredictor(run, cfg).predict(text, explain=args.explain)
    print(f"\nThreatWeigh ({run})\n")
    print(f"{"Metric":<28}{'Prediction':<14}{'Confidence':>10}")
    for m, r in p["metrics"].items():
        name = f"{METRIC_NAMES[m]} ({m.upper()})"
        print(f"{name:<28}{r['label'] + ' ' + LABEL_NAMES[m][r['label']]:<14}{r['confidence']:>10.1%}")
    print(f"\nVector   : {p['vector']}")
    print(f"Base     : {p['score']}  ({p['severity']})")
    if p["keywords"]:
        print(f"KeyBERT  : {', '.join(p['keywords'])}")
    if args.explain:
        print("\nTop attributed words per metric (* = in the expert lexicon):")
        for m, r in p["metrics"].items():
            top = sorted(r["words"], key=lambda w: -w["attr"])[: args.top]
            print(f"  {m.upper():<3}" + ", ".join(w["word"] + ("*" if w["in_lexicon"] else "") for w in top))


if __name__ == "__main__":
    main()
