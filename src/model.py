"""
The ThreatWeigh model: Sentence-BERT encoder + symbolic-guided weighted pooling.

For ONE metric (we train 8 independent copies, one per CVSS metric):

    h_i     = encoder(text)_i                    contextual vector of token i
    s_i     = W_pool h_i + b                     learned relevance score (scalar)
    alpha_i = softmax_i(s_i)                     over real tokens only (padding masked)
    w_i     = 10 if token i in K_exp AND K_mdl   (expert lexicon AND KeyBERT)
              1  if in exactly one of them
              0  otherwise
    w_i     = w_i + 1                            -> final values 11, 2 or 1
    beta_i  = alpha_i * w_i / (sum_j alpha_j * w_j + eps)
    z       = sum_i beta_i h_i                   the sentence vector
    p(y|x)  = softmax(W_cls z + b_cls)           trained with cross-entropy

So the network still *learns* where to look (alpha), but tokens that the
security expert lexicon and KeyBERT both flag get up to 11x more weight.

Pooling variants (used for baselines and ablations):
    "weighted"  ThreatWeigh pooling above (the symbolic sources decide w_i)
    "mean"      plain mean of token vectors   (plain SBERT baseline)
    "cls"       first ([CLS]) token vector     (DistilBERT baseline)
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Which symbolic sources each ablation variant uses.
VARIANT_SOURCES = {
    "alpha_only": (False, False),     # (a) learned alpha only, w_i = 1 everywhere
    "alpha_expert": (True, False),    # (b) alpha + expert lexicon
    "alpha_keybert": (False, True),   # (c) alpha + KeyBERT
    "full": (True, True),             # (d) alpha + expert + KeyBERT = ThreatWeigh
}


# ---------------------------------------------------------------------------
# 1. Symbolic weights per token
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _alternation(phrases: tuple[str, ...], hyphen_is_word: bool) -> re.Pattern:
    """One regex that finds every phrase at EVERY start position (via a lookahead),
    so overlapping phrases ('remote', 'remote attackers') are all found."""
    if not phrases:
        return re.compile(r"(?!x)x")                     # matches nothing
    alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    if hyphen_is_word:   # expert phrases: 'dos' must not match inside 'anti-dos'
        left, right = r"(?<![\w-])", r"(?![\w-])"
    else:                # KeyBERT words follow sklearn's \b\w\w+\b tokenisation
        left, right = r"\b", r"\b"
    return re.compile(rf"(?={left}({alts}){right})", re.IGNORECASE)


def char_mask(text: str, phrases, hyphen_is_word: bool) -> np.ndarray:
    """Boolean array over characters: True where any phrase occurs."""
    mask = np.zeros(len(text) + 1, dtype=bool)
    for m in _alternation(tuple(sorted(set(phrases))), hyphen_is_word).finditer(text):
        mask[m.start(1):m.end(1)] = True
    return mask


def token_hits(offsets, cmask: np.ndarray) -> np.ndarray:
    """A token is 'in' a source if any of its characters is covered by a match.
    Special tokens ([CLS], [SEP]) have offsets (0, 0) and never match."""
    out = np.zeros(len(offsets), dtype=bool)
    for t, (a, b) in enumerate(offsets):
        if b > a and cmask[a:b].any():
            out[t] = True
    return out


def combine_weights(in_exp: np.ndarray, in_kb: np.ndarray, weight_both: float = 10.0,
                    weight_one: float = 1.0, weight_none: float = 0.0, offset: float = 1.0) -> np.ndarray:
    """The paper's rule: both -> 10, exactly one -> 1, none -> 0, then +1 (-> 11/2/1)."""
    both = in_exp & in_kb
    one = in_exp ^ in_kb
    w = np.where(both, weight_both, np.where(one, weight_one, weight_none)).astype(np.float32)
    return w + offset


def symbolic_weights(text: str, offsets, exp_phrases, kb_words, sym_cfg: dict,
                     use_exp: bool = True, use_kb: bool = True) -> np.ndarray:
    """Weights w_i for every token of one text (length = number of tokens)."""
    n = len(offsets)
    in_exp = token_hits(offsets, char_mask(text, exp_phrases, True)) if use_exp else np.zeros(n, bool)
    in_kb = token_hits(offsets, char_mask(text, kb_words, False)) if use_kb else np.zeros(n, bool)
    return combine_weights(in_exp, in_kb, sym_cfg["weight_both"], sym_cfg["weight_one"],
                           sym_cfg["weight_none"], sym_cfg["offset"])


# ---------------------------------------------------------------------------
# 2. Pooling
# ---------------------------------------------------------------------------
def fused_pooling(h: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor,
                  sym_w: torch.Tensor, eps: float = 1e-8):
    """beta = alpha * w / (sum alpha * w + eps);  z = sum beta * h.

    h      [B, T, H] token vectors      scores [B, T] raw scores s_i
    mask   [B, T] 1 = real token        sym_w  [B, T] symbolic weights w_i
    Returns z [B, H], alpha [B, T], beta [B, T].
    """
    scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)  # padding gets alpha = 0
    alpha = torch.softmax(scores, dim=-1) * mask                           # exactly 0 on padding
    aw = alpha * sym_w * mask
    beta = aw / (aw.sum(dim=-1, keepdim=True) + eps)
    z = torch.einsum("bt,bth->bh", beta, h)
    return z, alpha, beta


class ThreatWeighModel(nn.Module):
    """Encoder + pooling + linear classifier for ONE CVSS metric."""

    def __init__(self, encoder_name: str, n_labels: int, pooling: str = "weighted",
                 dropout: float = 0.1, eps: float = 1e-8):
        super().__init__()
        assert pooling in {"weighted", "mean", "cls"}
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.eps = eps
        self.pool_score = nn.Linear(hidden, 1)          # W_pool, b  -> s_i
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_labels)   # linear layer; softmax is in the loss

    def word_embeddings(self) -> nn.Embedding:
        return self.encoder.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, sym_weights=None, inputs_embeds=None,
                return_details: bool = False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        h = out.last_hidden_state                                   # [B, T, H]
        mask = attention_mask.to(h.dtype)
        alpha = beta = None
        if self.pooling == "mean":
            z = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        elif self.pooling == "cls":
            z = h[:, 0]
        else:
            if sym_weights is None:
                sym_weights = torch.ones_like(mask)
            scores = self.pool_score(h).squeeze(-1)                 # s_i = W_pool h_i + b
            z, alpha, beta = fused_pooling(h, scores, mask, sym_weights.to(h.dtype), self.eps)
        logits = self.classifier(self.dropout(z))
        if return_details:
            return logits, {"alpha": alpha, "beta": beta}
        return logits


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    assert tok.is_fast, "a fast tokenizer is needed for character offsets"
    return tok
