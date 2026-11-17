# YouTube upload template — ThreatWeigh demo

Files to upload:
* **Video:** `video/threatweigh_demo.mp4` (1280×720, 3:09, no audio)
* **Thumbnail:** `video/youtube/thumbnail.png` (1280×720 PNG; source `thumbnail.html`)

Replace the `[…]` placeholders before publishing.

---

## Title (pick one, all ≤ 100 characters)

1. ThreatWeigh: Predicting CVSS Scores from Vulnerability Text (Explainable NLP Demo)
2. From CVE Description to CVSS Score — a Neuro-Symbolic NLP Model, Explained
3. Can a Language Model Score Vulnerabilities? ThreatWeigh Demo (Sentence-BERT + Expert Rules)

---

## Description (copy everything between the lines)

---8<---
ThreatWeigh reads the text of a software vulnerability (a CVE) and predicts the 8 CVSS v3.1 Base metrics — Attack Vector, Attack Complexity, Privileges Required, User Interaction, Scope, Confidentiality, Integrity and Availability — then computes the official 0–10 Base Score. It also shows which words each prediction relied on.

It is a student-friendly re-implementation of WSymBERT: a Sentence-BERT model whose attention is nudged by expert knowledge. Words that both a CVSS-based expert lexicon and KeyBERT flag get 11× weight in the pooling step.

In this demo:
• the data pipeline (137,521 NVD CVEs, and the label leakage we found and removed)
• how the model weighs words (with the real weights for an example sentence)
• the command-line tool with real output
• the live Streamlit web app: predictions, word-level explanations and an honest failure case
• the results

Key results (one training seed, 4,000 training rows on a CPU, 13,743 test CVEs):
• ThreatWeigh 88.28% mean accuracy, plain Sentence-BERT 88.22%, TF-IDF + logistic regression 88.24% — practically tied
• On newer CVEs (trained up to 2021, tested 2023+): ThreatWeigh 88.33% vs TF-IDF 87.15%
• Expert word lists can also mislead: "administrator" pushed one prediction to Privileges Required = High
These are observations at a small compute budget, not the paper's full-scale setting.

⏱️ Chapters
0:00 Intro
0:06 The problem: CVE text → CVSS score
0:17 Data pipeline and label leakage
0:29 Model architecture
0:46 How words get weighed
0:59 Command-line demo
1:18 Live app: remote code execution example
1:47 Live app: stored XSS and a failure case
2:14 Model results tab
2:37 How it works tab
2:50 Results summary and how to run it

🔗 Code: [GitHub link]

📚 Credits
Method: Mirtaheri et al., "WSymBERT: a weighted neuro-symbolic model for interpretable CVSS prediction", International Journal of Information Security (2026). ThreatWeigh is an independent, simplified re-implementation and does not reproduce the paper's numbers.
Data: CVE descriptions and CVSS vectors from the NIST National Vulnerability Database (NVD), via the WSymBERT repository (github.com/ShahbazianR/WSymBert), licensed CC-BY-4.0.
CVSS is a standard of FIRST (first.org/cvss).
Tools: Sentence-Transformers (all-MiniLM-L6-v2), Hugging Face Transformers, KeyBERT, Captum, scikit-learn, Streamlit.

This is an educational project and should not be the only basis for real vulnerability-management decisions.

#cybersecurity #NLP #ExplainableAI
--->8---

---

## Tags (≤ 500 characters in total)

```
CVSS, CVSS 3.1, CVE, vulnerability, vulnerability scoring, cybersecurity, NVD, NLP, natural language processing, machine learning, deep learning, Sentence-BERT, SBERT, BERT, transformers, KeyBERT, explainable AI, XAI, interpretability, Integrated Gradients, Captum, neuro-symbolic, Streamlit, Python, PyTorch, security, threat intelligence, WSymBERT
```

---

## Upload settings checklist

- [ ] Title and description pasted, placeholders filled in (`[GitHub link]`)
- [ ] Thumbnail: `thumbnail.png`
- [ ] Audience: "No, it's not made for kids"
- [ ] Category: Science & Technology
- [ ] Language: English · Caption certification: "None" (the video has on-screen captions, no speech)
- [ ] License: Standard YouTube License (the data credit above satisfies CC-BY-4.0 attribution)
- [ ] Chapters: keep "Allow automatic chapters" on; the timestamps above define them (first must be 0:00, each ≥ 10 s)
- [ ] Visibility: Unlisted first to check it, then Public
