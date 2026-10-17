"""Datasets, collation, and sampling for (drug, protein, pKd) pairs."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def kd_to_pkd(kd_nm):
    """Convert Kd in nM to pKd = -log10(Kd [M])."""
    return -np.log10(np.asarray(kd_nm) / 1e9)


class AffinityDataset(Dataset):
    """Look up cached embeddings for each row of a split dataframe.

    ``drug_cache`` / ``prot_cache`` map SMILES / amino-acid sequence to a tensor:
    either a pooled ``(H,)`` vector or a token-level ``(L, H)`` sequence.
    Items are ``(drug_emb, prot_emb, pKd, protein_family)``.
    """

    def __init__(self, df: pd.DataFrame, drug_cache: dict, prot_cache: dict):
        self.rows = df.reset_index(drop=True)
        self.drug_cache = drug_cache
        self.prot_cache = prot_cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        family = row["Protein_Family"] if "Protein_Family" in row else "other"
        y = torch.tensor(row["pKd"], dtype=torch.float)
        return self.drug_cache[row["Drug"]], self.prot_cache[row["Target"]], y, family


def collate_padded(batch):
    """Pad variable-length token embeddings to the batch maximum.

    Returns ``(drug_pad, drug_mask, prot_pad, prot_mask, pKd, families)`` where
    masks are 1 for real tokens and 0 for padding.
    """
    drug_embs, prot_embs, pkds, families = zip(*batch)
    max_d = max(e.shape[0] for e in drug_embs)
    max_p = max(e.shape[0] for e in prot_embs)
    B, D, P = len(batch), drug_embs[0].shape[1], prot_embs[0].shape[1]

    drug_pad, drug_mask = torch.zeros(B, max_d, D), torch.zeros(B, max_d)
    prot_pad, prot_mask = torch.zeros(B, max_p, P), torch.zeros(B, max_p)
    for i, (d, p) in enumerate(zip(drug_embs, prot_embs)):
        drug_pad[i, : d.shape[0]] = d
        drug_mask[i, : d.shape[0]] = 1
        prot_pad[i, : p.shape[0]] = p
        prot_mask[i, : p.shape[0]] = 1

    return drug_pad, drug_mask, prot_pad, prot_mask, torch.stack(pkds), list(families)


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Oversample pairs from rarely-measured proteins (weight = 1/sqrt(count))."""
    counts = df["Target"].value_counts()
    weights = df["Target"].map(lambda p: 1.0 / np.sqrt(counts[p])).values
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(df),
        replacement=True,
    )
