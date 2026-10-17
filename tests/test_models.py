import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from affinity import (
    AffinityDataset,
    BilinearAffinityModel,
    CrossAttentionRegressor,
    HybridAffinityModel,
    MLPPredictor,
    collate_padded,
    kd_to_pkd,
    make_weighted_sampler,
)

B = 4


@pytest.fixture
def token_batch():
    """A padded batch of random token-level embeddings with uneven lengths."""
    torch.manual_seed(0)
    df = pd.DataFrame(
        {
            "Drug": [f"d{i}" for i in range(B)],
            "Target": [f"p{i % 2}" for i in range(B)],
            "pKd": np.linspace(5, 9, B),
            "Protein_Family": ["kinase", "GPCR", "kinase", "GPCR"],
        }
    )
    drugs = {f"d{i}": torch.randn(10 + i, 768) for i in range(B)}
    prots = {"p0": torch.randn(30, 1024), "p1": torch.randn(45, 1024)}
    loader = DataLoader(AffinityDataset(df, drugs, prots), batch_size=B, collate_fn=collate_padded)
    return next(iter(loader))


def test_kd_to_pkd():
    assert kd_to_pkd(1.0) == pytest.approx(9.0)  # 1 nM
    assert kd_to_pkd(10_000.0) == pytest.approx(5.0)  # 10 uM


def test_collate_padded_masks(token_batch):
    drug_pad, drug_mask, prot_pad, prot_mask, y, families = token_batch
    assert drug_pad.shape == (B, 13, 768) and prot_pad.shape == (B, 45, 1024)
    assert drug_mask.sum(1).tolist() == [10, 11, 12, 13]
    assert prot_mask.sum(1).tolist() == [30, 45, 30, 45]
    assert y.shape == (B,) and families == ["kinase", "GPCR", "kinase", "GPCR"]


@pytest.mark.parametrize(
    "model",
    [
        MLPPredictor(),
        MLPPredictor(hidden_dim=[1024, 512, 256]),
        MLPPredictor(norm_after_activation=True),
        BilinearAffinityModel(),
        BilinearAffinityModel(dropout=0.4, post_norm="layer"),
    ],
)
def test_pooled_models_output_shape(model):
    model.eval()
    out = model(torch.randn(B, 768), torch.randn(B, 1024))
    assert out.shape == (B,)


def test_mlp_gaussian_head():
    out = MLPPredictor(output_dim=2).eval()(torch.randn(B, 768), torch.randn(B, 1024))
    assert out.shape == (B, 2)


@pytest.mark.parametrize("model", [CrossAttentionRegressor(), HybridAffinityModel()])
@pytest.mark.parametrize("training", [True, False])
def test_token_models_output_shape(model, training, token_batch):
    drug_pad, drug_mask, prot_pad, prot_mask, y, _ = token_batch
    model.train(training)
    out = model(drug_pad, drug_mask, prot_pad, prot_mask)
    assert out.shape == (B,)
    if training:
        ((out - y) ** 2).mean().backward()


@pytest.mark.parametrize("model", [CrossAttentionRegressor(), HybridAffinityModel()])
def test_token_models_ignore_padding(model, token_batch):
    """Changing values in padded positions must not change predictions."""
    drug_pad, drug_mask, prot_pad, prot_mask, _, _ = token_batch
    model.eval()
    before = model(drug_pad, drug_mask, prot_pad, prot_mask)
    noisy_prot = prot_pad + 100 * (1 - prot_mask).unsqueeze(-1) * torch.randn_like(prot_pad)
    after = model(drug_pad, drug_mask, noisy_prot, prot_mask)
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-4)


def test_weighted_sampler_favours_rare_proteins():
    df = pd.DataFrame({"Target": ["common"] * 9 + ["rare"]})
    weights = make_weighted_sampler(df).weights
    assert weights[-1] == pytest.approx(3 * weights[0])  # 1/sqrt(1) vs 1/sqrt(9)
