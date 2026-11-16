# ThreatWeigh

**Interpretable CVSS v3.1 prediction from vulnerability text.**

ThreatWeigh reads the English description of a software vulnerability (a CVE) and predicts
the 8 CVSS v3.1 Base metrics. It then feeds them into the official CVSS v3.1 formula to get
the 0–10 Base Score, and shows which words each prediction relied on.

ThreatWeigh is a **simplified, student-friendly re-implementation** of WSymBERT
(Mirtaheri et al., 2026; see [Credits](#credits-and-citation)). It follows the paper's equations,
but it is **not** a reproduction of the paper's numbers. The paper reports about 96.2% mean accuracy
on NVD. That figure is quoted here for reference only: ThreatWeigh's reported runs use a much
smaller compute budget (see [Compute budget](#compute-budget)).

> **Status:** stages 1–3 have run and the TF-IDF baseline has results. The neural results
> (stages 4–7) are being produced and will be filled into the tables below as each run finishes.
> Rows marked *pending* have not been run yet. No number in this README is estimated or invented.

---

## 1. The problem

Every published CVE gets a CVSS vector such as `CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H`.
Each part is one of 8 Base metrics:

| Metric | Name | Values |
|---|---|---|
| AV | Attack Vector | **N**etwork, **A**djacent, **L**ocal, **P**hysical |
| AC | Attack Complexity | **L**ow, **H**igh |
| PR | Privileges Required | **N**one, **L**ow, **H**igh |
| UI | User Interaction | **N**one, **R**equired |
| S | Scope | **U**nchanged, **C**hanged |
| C / I / A | Confidentiality / Integrity / Availability impact | **N**one, **L**ow, **H**igh |

Analysts pick these values by hand after reading the description, which is slow and inconsistent.
ThreatWeigh treats the task as **8 independent text-classification problems**, then computes the
score with the official formula.

## 2. Method

There is **one independent model per metric**: 8 models with no shared parameters and no multi-task loss.
Each model works like this:

```
CVE text ─► Sentence-BERT encoder ─► token vectors h_i ─► learned scores α_i ┐
        ├─► expert lexicon K_exp (CVSS decision trees) ┐                     ├─► fused pooling z ─► linear+softmax ─► label
        └─► KeyBERT top-5 words K_mdl ─────────────────┴─► symbolic weight w_i ┘
8 labels ─► official CVSS v3.1 formula ─► Base Score + severity
```

### 2.1 Encoder
The encoder is a Sentence-BERT checkpoint, loaded with Hugging Face Transformers and fine-tuned separately for each metric.
Inputs are truncated to 128 tokens. The checkpoint is set in `config.yaml → model.encoder`:

* default: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, practical on a CPU or small GPU)
* stronger option: `sentence-transformers/all-mpnet-base-v2` (110M parameters, needs a GPU)

The paper does not name its checkpoint; this choice is ours. (The authors' released code
uses `efederici/sentence-bert-base`.)

### 2.2 Learned pooling weights
For each token vector $h_i$:

$$s_i = W_{pool}\,h_i + b, \qquad \alpha_i = \frac{\exp(s_i)}{\sum_{j \in \text{real tokens}} \exp(s_j)}$$

Padding tokens are masked out, so they always get $\alpha_i = 0$.

### 2.3 Symbolic knowledge
* **Expert lexicon $K_{exp}$** (`src/lexicon.py` → `lexicons/expert_lexicon.json`). For each metric, a
  question-style decision tree written from the FIRST CVSS v3.1 specification. For example, AV asks:
  *"Is physical contact required?"* → P, *"Same local or radio network?"* → A, *"Local account or local
  access?"* → L, *"Otherwise remotely exploitable?"* → N. Each answer has hand-written cue phrases
  ("physical access", "bluetooth", "local users", "remote attackers", …).
  Cues are then checked on the **training split only**. A cue is kept if it occurs in at least 10 training
  descriptions and lift = P(label | cue) / P(label) ≥ 1.2. The lexicon is then frozen.
  No words are mined from the data, and test data is never used.
* **KeyBERT keywords $K_{mdl}$** (`src/keybert_cache.py`). The top 5 single words per description, scored by
  the same SBERT checkpoint (not fine-tuned) with `keyphrase_ngram_range=(1,1)`, no stop-word
  filtering and no MMR. They are cached to disk before training.

### 2.4 Symbolic weight per token
Multi-word lexicon phrases are mapped onto all of their sub-word tokens through the tokenizer's
character offsets:

$$w_i = \begin{cases} 10 & \text{token in } K_{exp} \text{ and } K_{mdl} \\ 1 & \text{in exactly one} \\ 0 & \text{otherwise}\end{cases} \qquad w_i \leftarrow w_i + 1 \;\in \{11, 2, 1\}$$

### 2.5 Fused pooling and classifier

$$\beta_i = \frac{\alpha_i\, w_i}{\sum_j \alpha_j w_j + \varepsilon}, \qquad z = \sum_i \beta_i h_i, \qquad p(y \mid x) = \mathrm{softmax}(W z + b)$$

Training uses cross-entropy with no class weighting, AdamW, learning rate 2e-5 and batch size 16. The best epoch
is chosen by validation accuracy, with early stopping. The test split is used **once**, at the end.

### 2.6 CVSS v3.1 Base Score (`src/cvss.py`)

$$ISS = 1-(1-C)(1-I)(1-A)$$
$$\text{Impact} = \begin{cases}6.42\cdot ISS & S{=}U\\ 7.52\,(ISS-0.029) - 3.25\,(ISS-0.02)^{15} & S{=}C\end{cases}$$
$$\text{Exploitability} = 8.22 \cdot AV \cdot AC \cdot PR \cdot UI$$

Base = 0 if Impact ≤ 0. Otherwise it is Roundup(min(Impact + Exploitability, 10)) when S is Unchanged,
or Roundup(min(1.08 · (Impact + Exploitability), 10)) when S is Changed. Roundup is the official integer-based
function from Appendix A of the spec. The calculator reproduces NVD's own score for **all 137,521 rows** of
the dataset (`tests/test_cvss.py`).

> Note on the project brief: it expected `AV:P/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L` → 4.6. The official value is
> **4.3** (ISS = 0.5254, Impact = 3.373, Exploitability = 0.915, sum 4.288 → 4.3), and NVD scores the 9 CVEs with this
> vector 4.3. The 4.6 belongs to `AV:P/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N`. The test checks both vectors.

### 2.7 Differences from the authors' released code
The released WSymBERT scripts differ from the paper's equations in a few places. ThreatWeigh follows the
**paper**:

| | Paper equations / ThreatWeigh | Released code (`Nvd/AV.py`) |
|---|---|---|
| token weight | 11 / 2 / 1 | `min(1 + w, 5)`, so capped at 5 |
| padding | masked out of the softmax | not masked |
| split | train / val / test | 80/20 train/val, no held-out test |
| epochs | 15 | 3 |

## 3. Data

* Source: `Nvd/Dataset/final.zip` from https://github.com/ShahbazianR/WSymBert, with 137,521 NVD CVEs from 2016–2024.
  Columns: `ID, text, vectorString, av … a, score, baseSeverity`. 100,122 rows have CVSS 3.1 vectors and 37,399
  have CVSS 3.0 vectors. Both versions define the 8 metrics identically, and our v3.1 calculator also matches NVD's
  score on the 3.0 rows.
* `combine/Dataset/final.zip` in the same repo is **byte-identical** (same MD5) to the NVD file, so it is not a
  separate dataset. ThreatWeigh uses one dataset and makes no claim about a "Combine" experiment.
* **Label leakage removed.** 9,199 descriptions (6.7%), mostly Oracle advisories, contain the answer, e.g.
  *"CVSS 3.1 Base Score 7.5 (Integrity impacts). CVSS Vector: (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N)."*
  On these rows any model can read the labels straight off the input. `src/data.py` strips such score and vector
  sentences before anything else, and `tests/test_data.py` checks that no vector remains in any split.
  This differs from the original dataset, so results are not directly comparable to models trained on the raw text.
* Preprocessing (`python -m src.data`): strip the leaked CVSS text, drop empty text, then strict de-duplication on
  lower-cased, whitespace-collapsed text (91 removed). That leaves **137,430 rows**. Then a stratified 80/10/10 split on AV
  with seed 42: **train 109,943 / val 13,744 / test 13,743**. A time-aware split by CVE ID year:
  train ≤ 2021 (90,267), val 2022 (21,821), test ≥ 2023 (25,342).

Label distribution of the training split (full table: `results/label_distribution.csv`). Note the heavy imbalance:

| AV | AC | PR | UI | S | C | I | A |
|---|---|---|---|---|---|---|---|
| N 73.9, L 23.0, A 2.1, P 1.0 | L 94.0, H 6.0 | N 62.4, L 29.4, H 8.2 | N 65.4, R 34.6 | U 82.3, C 17.7 | H 57.5, N 21.8, L 20.7 | H 49.5, N 31.1, L 19.3 | H 57.5, N 40.3, L 2.2 |

Because of this imbalance, accuracy flatters the rare classes, so macro-F1 is reported too.

## 4. Setup

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

A GPU is detected and used automatically if one is present. Everything also runs on a CPU.
The first run downloads `all-MiniLM-L6-v2` (~90 MB) and `distilbert-base-uncased` (~260 MB) from Hugging Face.

## 5. How to run each stage

All settings are in **`config.yaml`**. The `--profile` flag selects `quick` (smoke test), `budget` (what was run
for the results below) or `full` (paper settings: all training rows, 15 epochs).

| Stage | Command |
|---|---|
| 1. Data + EDA | `python -m src.data` |
| 2. CVSS tests | `python -m pytest` |
| 3a. Expert lexicon + coverage | `python -m src.lexicon` |
| 3b. KeyBERT cache (~25 min on 4 CPU cores) | `python -m src.keybert_cache` |
| 4. TF-IDF baseline | `python -m src.baselines tfidf --profile budget` |
| 4. SBERT mean-pooling baseline | `python -m src.baselines sbert --profile budget --seed 42` |
| 4. DistilBERT baseline | `python -m src.baselines distilbert --profile budget --seed 42` |
| 5. ThreatWeigh | `python -m src.train --variant full --profile budget --seed 42 --save-model` |
| 6. Ablations | `python -m src.train --variant alpha_only` (also `alpha_expert`, `alpha_keybert`) |
| 6. Weight sensitivity | `python -m src.train --variant full --weight-both 2` (also 5, 20) |
| 6. Temporal split | `python -m src.train --variant full --split temporal` |
| 6. Tables, seeds, bootstrap | `python -m src.evaluate` |
| 7. Explainability | `python -m src.explain --tw <threatweigh run> --base <sbert run>` |
| 8. CLI | `python predict.py "A remote attacker can execute arbitrary code via a crafted request."` |
| 8. App | `streamlit run app.py` |

Smoke test of the whole pipeline on any CPU: `python -m src.train --variant full --quick`.

Each training run writes `results/runs/<run>/preds_<metric>.npz` (test probabilities),
per-epoch histories and `run.json`. With `--save-model`, it also writes the weights to `models/<run>/`.

## 6. Results

### Compute budget
The results were produced on a machine with **no GPU (4 CPU cores, 15 GB RAM)**. Training 8 models × many variants ×
3 seeds on all 110k training rows for 15 epochs was not feasible there, so the reported runs use the `budget` profile:

* the **same** 4,000-row stratified training subsample for every model (drawn once with seed 42)
* a 1,000-row validation subsample for model selection, up to 3 epochs, early-stopping patience 2
* length-bucketed batches (texts of similar length are batched together; ~1.8× faster on CPU, same model and loss)
* the **full 13,745-row test split** for every reported number
* the same learning rate (2e-5) and batch size (16) as the paper

Absolute accuracies are therefore lower than a full-data GPU run would give. **The comparisons between models are
the meaningful part**, because all models share the same rows, splits and hyperparameters. To reproduce at paper
scale, run with `--profile full` on a GPU.

### 6.1 Expert-lexicon coverage (test split)
Coverage is the % of descriptions with at least one cue match. The paper reports 83.6% on 2,000 CVEs with its own lexicon.

| Metric | AV | AC | PR | UI | S | C | I | A | **Any** |
|---|---|---|---|---|---|---|---|---|---|
| cues | 94 | 29 | 68 | 54 | 20 | 64 | 55 | 47 | 431 |
| coverage % (13,743 test CVEs) | 58.0 | 6.3 | 39.5 | 38.0 | 20.1 | 58.2 | 59.6 | 48.4 | **88.9** |
| coverage % (random 2,000) | 60.3 | 5.5 | 39.3 | 37.5 | 19.8 | 57.8 | 59.2 | 47.8 | **89.9** |

The lift filter cannot keep cues for very frequent labels. AC:L is 94% of training rows, so its lift can be at most ≈1.06,
and it kept no cues. S:U kept only 3. The lexicon is informative mainly for the minority labels.

### 6.2 Main comparison (test accuracy %, budget profile)
*Generated by `python -m src.evaluate` → `results/main_comparison.md`.*

| Model | AV | AC | PR | UI | S | C | I | A | mean acc | mean macro-F1 | score MSE | score MAE | severity acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 88.9 | 95.6 | 80.9 | 91.0 | 95.3 | 83.5 | 84.5 | 87.2 | 88.38 | 75.82 | 2.515 | 0.919 | 66.88 |
| DistilBERT ([CLS]) | *pending* | | | | | | | | | | | | |
| SBERT (mean pooling) | *pending* | | | | | | | | | | | | |
| **ThreatWeigh** | *pending* | | | | | | | | | | | | |

The score MSE and MAE compare the Base Score computed from the 8 *predicted* labels with the score from the true labels.
Severity accuracy is the % of CVEs whose None/Low/Medium/High/Critical band is correct.

### 6.3 Ablation, seeds, significance, weight sensitivity, temporal split
*Pending*. These will appear in `results/ablation.md`, `seeds.md`, `significance_full_vs_sbert.md`,
`weight_sensitivity.md` and `temporal.md`.

### 6.4 Explainability
*Pending*. Will appear in `results/explain_summary.csv` and `explain_examples.json`.

Integrated Gradients (Captum) is computed on the token embeddings of every metric model, with two reference baselines:
an all-[PAD]-token embedding and an all-ones embedding. Reported measures:

* the top-10 attributed words for sample CVEs (ThreatWeigh vs plain SBERT)
* the **noise rate**: the % of top-10 words that are stop-words, pure numbers or version strings, or ≤ 2 characters
* **infidelity** and **sensitivity** (Captum metrics)
* **deletion-AUC**: words are removed from most to least attributed while tracking the predicted-class probability.
  A lower AUC means the attribution found the words that matter. A random deletion order is reported as a reference.

**Attributions are descriptive.** They show what the model's output is sensitive to. They do not prove that the model
reasons causally, or the way a security analyst would.

## 7. Interface

* **CLI**: `python predict.py "description…"` prints the 8 metrics with confidence, the vector, the score and the severity.
  `--explain` adds the top words for each metric, with lexicon words marked `*`.
* **Streamlit app** (`streamlit run app.py`), titled "ThreatWeigh", has three tabs:
  1. *Predict*: a text box and example buttons. It shows the 8 metrics with confidence, the vector, the Base Score and
     severity, and highlights the top-attributed words, marking those in the expert lexicon.
  2. *Model results*: the comparison tables, ablation, seed variance, sensitivity and temporal results.
  3. *How it works*: a plain-language explanation and the pipeline diagram.

## 8. Project structure

```
README.md  TODO.md  requirements.txt  config.yaml  pytest.ini
data/        raw/final.zip (CC-BY-4.0, from WSymBERT), splits/ and keybert/ are generated
lexicons/    expert_lexicon.json   (frozen; per-cue train support / precision / lift)
models/      trained weights (generated, not committed)
results/     tables (.csv/.md), figures, runs/<run>/preds_*.npz
src/         utils.py data.py cvss.py lexicon.py keybert_cache.py model.py
             train.py baselines.py evaluate.py explain.py
tests/       test_cvss.py test_weights.py test_data.py
app.py  predict.py
```

Tests: `python -m pytest`. They cover the CVSS calculator (including agreement with NVD on the whole dataset), the
11/2/1 weighting rule and phrase-to-sub-word mapping, pooling weights that sum to 1 and ignore padding, and data splits
with no ID or text leakage, correct proportions and AV stratification. Seeds are set for Python, NumPy and PyTorch.

## 9. Limitations

* **Compute.** The reported neural results come from a CPU-only budget (6k training rows, ≤ 4 epochs, MiniLM). They are
  not comparable to the paper's full-data GPU numbers, and are likely an underestimate of what the method can do.
* **Our lexicon is not the authors' lexicon.** It was written from the CVSS specification for this project and filtered
  on training data. Different cue lists will give different results.
* **Labels are noisy.** NVD vectors are assigned by people; 17 duplicate descriptions in the raw data carry conflicting
  vectors. A 27% share of the rows were scored under CVSS v3.0.
* **Description-only.** Vendor advisories, CWE and CPE are not used, and some metrics (AC, S) are often not determinable from
  the text alone.
* **Rare classes** (AV:P 1%, AV:A 2%, A:L 2%, AC:H 6%) are hard. Without class weighting (as in the paper), macro-F1 is much lower
  than accuracy.
* **Associations, not causes.** Any accuracy differences between variants are observations on this data and budget. Small
  differences may not survive other seeds, encoders or data sizes; see the seed variance and bootstrap CIs.
* **Explanations** are descriptive (see 6.4).

## 10. Credits and citation

ThreatWeigh re-implements the method of:

> Mirtaheri et al. "WSymBERT: a weighted neuro-symbolic model for interpretable CVSS prediction."
> *International Journal of Information Security*, 2026.

Code and data: https://github.com/ShahbazianR/WSymBert. The code is MIT licensed and the **datasets are CC-BY-4.0**. The
`data/raw/final.zip` file in this repository is redistributed unchanged from that repository under CC-BY-4.0, with credit to the
WSymBERT authors. The vulnerability descriptions and CVSS vectors originate from the **NIST National Vulnerability
Database (NVD)**.

Other tools used: Sentence-Transformers (`all-MiniLM-L6-v2`), Hugging Face Transformers (`distilbert-base-uncased`),
KeyBERT, Captum, scikit-learn, Streamlit. CVSS v3.1 is © FIRST (https://www.first.org/cvss/).

ThreatWeigh is an educational project and must not be used as the sole basis for real vulnerability-management decisions.
