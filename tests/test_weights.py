"""Tests for the symbolic weighting (11/2/1 rule) and the fused pooling."""
import numpy as np
import torch

from src.model import combine_weights, fused_pooling, symbolic_weights, token_hits, char_mask

SYM = {"weight_both": 10, "weight_one": 1, "weight_none": 0, "offset": 1}


def test_11_2_1_rule():
    exp = np.array([True, True, False, False])
    kb = np.array([True, False, True, False])
    assert combine_weights(exp, kb).tolist() == [11, 2, 2, 1]


def test_weight_value_is_configurable():
    exp, kb = np.array([True, False]), np.array([True, False])
    assert combine_weights(exp, kb, weight_both=20).tolist() == [21, 1]


def test_multiword_phrase_maps_to_all_subword_tokens():
    # 'remote attackers' is one lexicon phrase; tokens are 'remote','attack','##ers'
    text = "allows remote attackers to crash"
    offsets = [(0, 0), (0, 6), (7, 13), (14, 20), (20, 23), (24, 26), (27, 32), (0, 0)]
    hits = token_hits(offsets, char_mask(text, ["remote attackers"], True))
    assert hits.tolist() == [False, False, True, True, True, False, False, False]


def test_symbolic_weights_end_to_end():
    text = "allows remote attackers to crash"
    offsets = [(0, 0), (0, 6), (7, 13), (14, 20), (20, 23), (24, 26), (27, 32), (0, 0)]
    w = symbolic_weights(text, offsets, exp_phrases=["remote attackers", "crash"],
                         kb_words=["crash", "allows"], sym_cfg=SYM)
    #           [CLS] allows remote attack ##ers  to  crash [SEP]
    assert w.tolist() == [1, 2, 2, 2, 2, 1, 11, 1]


def test_whole_word_matching():
    # 'dos' must not match inside 'windows'; 'admin' must not match 'administrator'
    m = char_mask("windows administrator", ["dos", "admin"], True)
    assert not m.any()


def test_pooling_weights_sum_to_one_and_ignore_padding():
    torch.manual_seed(0)
    B, T, H = 3, 6, 4
    h = torch.randn(B, T, H)
    scores = torch.randn(B, T)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float)
    w = torch.tensor([[11, 2, 1, 1, 1, 1], [1, 11, 2, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.float)
    z, alpha, beta = fused_pooling(h, scores, mask, w)
    assert torch.allclose(alpha.sum(-1), torch.ones(B), atol=1e-6)
    assert torch.allclose(beta.sum(-1), torch.ones(B), atol=1e-6)
    assert (beta[mask == 0] == 0).all() and (alpha[mask == 0] == 0).all()
    # with all weights 1, beta must equal alpha
    _, a2, b2 = fused_pooling(h, scores, mask, torch.ones(B, T))
    assert torch.allclose(a2, b2, atol=1e-6)
    # a token with weight 11 gets a larger share than with weight 1
    _, _, b_flat = fused_pooling(h, scores, mask, torch.ones(B, T))
    assert beta[0, 0] > b_flat[0, 0]
    # z is the beta-weighted average of the token vectors
    assert torch.allclose(z[1], (beta[1, :, None] * h[1]).sum(0), atol=1e-6)
