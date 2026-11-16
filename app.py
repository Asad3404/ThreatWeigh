"""
ThreatWeigh - Streamlit app.

    streamlit run app.py

Tabs
  1. Predict        - type a CVE description, get the 8 metrics, vector, score, severity
                      and the words the model relied on (Integrated Gradients)
  2. Model results  - comparison tables, ablation, seed variance, weight sensitivity
  3. How it works   - plain-language explanation and the pipeline diagram
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.utils import LABEL_NAMES, METRIC_NAMES, load_config, project_path

st.set_page_config(page_title="ThreatWeigh", page_icon="🛡️", layout="wide")
CFG = load_config()
RES = project_path(CFG["paths"]["results_dir"])

EXAMPLES = {
    "Remote code execution": "A buffer overflow in the HTTP request parser of the web management interface allows "
                             "unauthenticated remote attackers to execute arbitrary code via a crafted request.",
    "Stored XSS": "A stored cross-site scripting (XSS) vulnerability in the comment form of the plugin allows "
                  "authenticated users with contributor privileges to inject arbitrary web script or HTML that "
                  "executes when an administrator views the page.",
    "Local privilege escalation": "A race condition in the kernel driver allows local users to gain root privileges "
                                  "via a crafted ioctl call.",
    "Bluetooth DoS": "Improper input validation in the Bluetooth stack allows an attacker within radio range to "
                     "cause a denial of service (device crash) via malformed packets.",
    "Physical access": "An attacker with physical access to the device can bypass the lock screen via the USB "
                       "debug port and read sensitive data stored on the device.",
}
SEV_COLOR = {"None": "#6b7280", "Low": "#2f9e44", "Medium": "#e8a400", "High": "#e8590c", "Critical": "#c92a2a"}


@st.cache_resource(show_spinner="Loading the 8 ThreatWeigh models ...")
def get_predictor(run: str):
    from src.explain import ThreatWeighPredictor

    return ThreatWeighPredictor(run, CFG)


def highlight(text: str, words: list[dict], top_k: int = 8) -> str:
    """HTML with the top attributed words highlighted; lexicon words are underlined + marked ◆."""
    top = sorted([w for w in words if w["attr"] > 0], key=lambda w: -w["attr"])[:top_k]
    if not top:
        return html.escape(text)
    mx = max(w["attr"] for w in top)
    spans = sorted(top, key=lambda w: w["start"])
    out, pos = [], 0
    for w in spans:
        out.append(html.escape(text[pos:w["start"]]))
        alpha = 0.25 + 0.65 * w["attr"] / mx
        style = f"background: rgba(250,82,82,{alpha:.2f}); border-radius:4px; padding:0 2px;"
        mark = ""
        if w["in_lexicon"]:
            style += " text-decoration: underline; text-decoration-thickness: 2px; font-weight:600;"
            mark = "<sup>◆</sup>"
        out.append(f"<span style='{style}' title='attribution {w['attr']:.3f}'>"
                   f"{html.escape(text[w['start']:w['end']])}</span>{mark}")
        pos = w["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out)


# ---------------------------------------------------------------------------
st.title("🛡️ ThreatWeigh")
st.caption("Interpretable CVSS v3.1 prediction from vulnerability text: a simplified, student-friendly "
           "re-implementation of WSymBERT (Mirtaheri et al., 2026).")
tab_pred, tab_res, tab_how = st.tabs(["Predict", "Model results", "How it works"])

# ---------------------------------------------------------------- Predict ---
with tab_pred:
    run = CFG["serve"]["run"]
    if not (project_path(CFG["paths"]["models_dir"]) / run).exists():
        st.warning(f"No trained models found in `models/{run}`. Train them first:\n\n"
                   "`python -m src.train --variant full --profile budget --save-model`")
    else:
        if "text" not in st.session_state:
            st.session_state.text = EXAMPLES["Remote code execution"]
        cols = st.columns(len(EXAMPLES))
        for c, (name, ex) in zip(cols, EXAMPLES.items()):
            if c.button(name, use_container_width=True):
                st.session_state.text = ex
        text = st.text_area("Vulnerability description", key="text", height=120)
        explain = st.toggle("Explain with Integrated Gradients (a few seconds on CPU)", value=True)
        if st.button("Predict", type="primary") and text.strip():
            with st.spinner("Predicting ..."):
                p = get_predictor(run).predict(text.strip(), explain=explain)
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric("Base Score", f"{p['score']:.1f}")
            c2.markdown(f"<div style='margin-top:6px'>Severity<br><span style='font-size:1.6rem;font-weight:700;"
                        f"color:{SEV_COLOR[p['severity']]}'>{p['severity']}</span></div>", unsafe_allow_html=True)
            c3.markdown(f"Vector string\n\n`{p['vector']}`")
            rows = [{"Metric": f"{METRIC_NAMES[m]} ({m.upper()})",
                     "Prediction": f"{r['label']} - {LABEL_NAMES[m][r['label']]}",
                     "Confidence": r["confidence"]} for m, r in p["metrics"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={"Confidence": st.column_config.ProgressColumn(
                             format="%.2f", min_value=0.0, max_value=1.0)})
            if p["keywords"]:
                st.caption("KeyBERT keywords: " + ", ".join(p["keywords"]))
            if explain:
                st.subheader("Which words mattered?")
                st.caption("Red = top words by Integrated Gradients (pad baseline) for that metric's prediction. "
                           "◆ underlined = word is in the expert lexicon. Attributions describe the model's "
                           "sensitivity, not proof of causal reasoning.")
                for m, r in p["metrics"].items():
                    st.markdown(f"**{m.upper()} → {r['label']} ({LABEL_NAMES[m][r['label']]})**  ",
                                unsafe_allow_html=True)
                    st.markdown(f"<div style='line-height:1.9; margin-bottom:18px'>"
                                f"{highlight(text.strip(), r['words'])}</div>",
                                unsafe_allow_html=True)

# ---------------------------------------------------------- Model results ---
with tab_res:
    def show(name: str, title: str, note: str = ""):
        p = RES / f"{name}.csv"
        if p.exists():
            st.subheader(title)
            if note:
                st.caption(note)
            st.dataframe(pd.read_csv(p), hide_index=True, use_container_width=True)

    st.info("All neural models were trained on the same 4,000-row stratified training subsample for up to 3 epochs "
            "on a CPU with one seed (see README → Compute budget) and tested on the full 13,743-row test split. "
            "Numbers are NOT comparable to the paper's (≈96.2% mean accuracy on full data).")
    for fig in ["fig_main_accuracy.png", "fig_ablation.png"]:
        if (RES / fig).exists():
            st.image(str(RES / fig), width=820)
    show("main_comparison", "Main comparison (test accuracy %)", "mean ± std over seeds where >1 seed")
    show("ablation", "Ablation")
    show("significance_full_vs_sbert", "ThreatWeigh vs plain SBERT: paired bootstrap 95% CI + McNemar")
    show("seeds", "Per-seed results")
    show("weight_sensitivity", "Sensitivity to the 'both' weight (10 in the paper)")
    show("temporal", "Time-aware split (train ≤2021, test ≥2023)")
    show("per_metric", "Per-metric accuracy / macro-precision / macro-recall / macro-F1")
    show("explain_summary", "Explanation quality (lower is better except top_in_lexicon)")
    show("lexicon_coverage", "Expert-lexicon coverage on the test split")

# ----------------------------------------------------------- How it works ---
with tab_how:
    st.markdown("""
**The problem.** Every new vulnerability (CVE) gets a CVSS score that says how dangerous it is.
Analysts read the description and pick 8 values (Attack Vector, Attack Complexity, Privileges
Required, User Interaction, Scope, Confidentiality, Integrity, Availability). The official
CVSS v3.1 formula turns those 8 values into a 0-10 Base Score. ThreatWeigh tries to pick the
8 values automatically from the text.

**The idea.** A language model (Sentence-BERT) reads the description and produces one vector per
word-piece. To build one vector for the whole sentence it must decide *how much each word counts*.
ThreatWeigh lets the model learn this (the attention weights **α**), but nudges it with security
knowledge: words that an **expert lexicon** (built from the CVSS rules, e.g. "local users" → AV:Local)
*and* **KeyBERT** (the 5 words most similar to the whole text) both flag get **11×** weight, words
flagged by just one get **2×**, others **1×**.

**One model per metric.** There are 8 completely separate models, so the AV model can focus on
"remote / local / bluetooth" while the UI model focuses on "click / visit / victim".

**Explaining.** Integrated Gradients measures how much each word moved the prediction. This
*describes* the model's behaviour; it does not prove the model reasons like an analyst.
""")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f1f3f5", fontname="Helvetica"];
  text [label="CVE description"];
  enc [label="Sentence-BERT encoder\\n(fine-tuned per metric)"];
  alpha [label="learned scores\\nα = softmax(W·h + b)"];
  lex [label="Expert lexicon\\n(CVSS decision trees)"];
  kb [label="KeyBERT\\ntop-5 words"];
  w [label="symbolic weight\\nw = 11 / 2 / 1"];
  pool [label="fused pooling\\nβ ∝ α·w,  z = Σ β·h"];
  clf [label="linear + softmax\\n→ metric label"];
  x8 [label="× 8 metrics\\nAV AC PR UI S C I A"];
  cvss [label="CVSS v3.1 formula\\n→ Base Score + severity"];
  text -> enc -> alpha -> pool; text -> lex -> w; text -> kb -> w; w -> pool; pool -> clf -> x8 -> cvss;
}""")
    st.markdown("Full details, equations and limitations: see `README.md`.")
