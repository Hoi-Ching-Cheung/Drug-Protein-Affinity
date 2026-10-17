"""Fusion heads that predict pKd from frozen ChemBERTa / ProtBERT embeddings.

Pooled-embedding models take ``(drug_emb, prot_emb)`` of shape ``(B, 768)`` and
``(B, 1024)``. Token-level models take padded sequences plus masks as produced
by :func:`affinity.data.collate_padded`. All models return ``(B,)`` predictions
(or ``(B, 2)`` mean/log-variance for ``MLPPredictor(output_dim=2)``).

Module and attribute names match the notebook so its checkpoints load directly.
"""

import random

import torch
import torch.nn as nn

DRUG_DIM = 768  # ChemBERTa-100M-MLM hidden size
PROT_DIM = 1024  # ProtBERT-BFD hidden size


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the sequence dimension of ``x`` (B, L, H), ignoring padding."""
    mask_f = mask.unsqueeze(-1).float()
    summed = (x * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp(min=1e-8)
    return summed / denom


class MLPPredictor(nn.Module):
    """Concatenate per-modality LayerNorm'd embeddings and regress with an MLP.

    Each hidden block is Linear -> BatchNorm -> ReLU -> Dropout. The baseline
    checkpoint was trained with ReLU before BatchNorm; load it with
    ``norm_after_activation=True``.
    """

    def __init__(
        self,
        drug_dim: int = DRUG_DIM,
        protein_dim: int = PROT_DIM,
        hidden_dim: list[int] = (512, 256),
        output_dim: int = 1,  # 1: MSE, 2: Gaussian NLL (mu, log_var)
        dropout: float = 0.2,
        norm_after_activation: bool = False,
    ):
        super().__init__()
        self.drug_norm = nn.LayerNorm(drug_dim)
        self.prot_norm = nn.LayerNorm(protein_dim)
        self.output_dim = output_dim

        layers = []
        in_dim = drug_dim + protein_dim
        for h in hidden_dim:
            norm_act = [nn.ReLU(), nn.BatchNorm1d(h)] if norm_after_activation else [nn.BatchNorm1d(h), nn.ReLU()]
            layers += [nn.Linear(in_dim, h), *norm_act, nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, drug_emb, prot_emb):
        x = torch.cat([self.drug_norm(drug_emb), self.prot_norm(prot_emb)], dim=1)
        out = self.mlp(x)
        if self.output_dim == 1:
            return out.squeeze(dim=-1)
        return out


class BilinearAffinityModel(nn.Module):
    """Project each modality to a shared space and fuse with a bilinear layer.

    ``post_norm="batch"`` is the baseline configuration; ``post_norm="layer"``
    with ``dropout=0.4`` is the tuned configuration reported in the results.
    """

    def __init__(
        self,
        drug_dim: int = DRUG_DIM,
        protein_dim: int = PROT_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        post_norm: str = "batch",
    ):
        super().__init__()
        if post_norm not in ("batch", "layer"):
            raise ValueError("post_norm must be 'batch' or 'layer'")

        self.drug_norm = nn.LayerNorm(drug_dim)
        self.prot_norm = nn.LayerNorm(protein_dim)
        self.drug_proj = nn.Sequential(nn.Linear(drug_dim, hidden_dim), nn.ReLU())
        self.prot_proj = nn.Sequential(nn.Linear(protein_dim, hidden_dim), nn.ReLU())
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        norm = nn.BatchNorm1d(hidden_dim) if post_norm == "batch" else nn.LayerNorm(hidden_dim)
        self.post_bilinear = nn.Sequential(norm, nn.ReLU(), nn.Dropout(dropout))
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, drug_emb, prot_emb):
        drug_h = self.drug_proj(self.drug_norm(drug_emb))
        prot_h = self.prot_proj(self.prot_norm(prot_emb))
        interaction = self.post_bilinear(self.bilinear(drug_h, prot_h))
        return self.output(interaction).squeeze(-1)


class CrossAttentionRegressor(nn.Module):
    """Bidirectional cross-attention between drug and protein token sequences.

    Drug tokens attend to protein tokens, then protein tokens attend to the
    updated drug tokens. Each attention block is skipped with probability
    ``attention_layer_dropout`` during training. The two sequences are
    mean-pooled, concatenated, and regressed with a small MLP.
    """

    def __init__(
        self,
        drug_input_dim: int = DRUG_DIM,
        prot_input_dim: int = PROT_DIM,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        regressor_hidden_dim: int = 128,
        attention_layer_dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        def proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        def post():
            return nn.Sequential(nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        self.drug_proj = proj(drug_input_dim)
        self.prot_proj = proj(prot_input_dim)
        self.drug_to_prot_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.prot_to_drug_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drug_post = post()
        self.prot_post = post()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, regressor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden_dim, regressor_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden_dim // 2, 1),
        )
        self.attention_layer_dropout = attention_layer_dropout

    def _skip_block(self) -> bool:
        return self.training and random.random() < self.attention_layer_dropout

    def forward(self, drug_tokens, drug_mask, prot_tokens, prot_mask):
        d = self.drug_proj(drug_tokens)
        p = self.prot_proj(prot_tokens)

        if not self._skip_block():
            d_att, _ = self.drug_to_prot_attn(d, p, p, key_padding_mask=prot_mask == 0, need_weights=False)
            d = self.drug_post(d + d_att)
        if not self._skip_block():
            p_att, _ = self.prot_to_drug_attn(p, d, d, key_padding_mask=drug_mask == 0, need_weights=False)
            p = self.prot_post(p + p_att)

        joint = torch.cat([masked_mean(d, drug_mask.bool()), masked_mean(p, prot_mask.bool())], dim=-1)
        return self.regressor(joint).squeeze(-1)


class HybridAffinityModel(nn.Module):
    """Parallel cross-attention and bilinear pathways, fused by a shared regressor.

    - Path A: bidirectional cross-attention + FFN blocks over token sequences,
      mean-pooled to ``2 * attn_dim``.
    - Path B: bilinear interaction of mean-pooled projections (``bilinear_dim``).
    - Each pathway is LayerNorm'd before concatenation. During training,
      optional Gaussian input noise and *pathway dropout* (zero one pathway and
      scale the other by 2) regularise the model.
    """

    def __init__(
        self,
        drug_input_dim: int = DRUG_DIM,
        prot_input_dim: int = PROT_DIM,
        attn_dim: int = 128,
        bilinear_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        embedding_noise_std: float = 0.0,
        pathway_drop_prob: float = 0.15,
    ):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        self.embedding_noise_std = embedding_noise_std
        self.pathway_drop_prob = pathway_drop_prob

        def proj(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(), nn.Dropout(dropout))

        def ffn():
            return nn.Sequential(
                nn.Linear(attn_dim, attn_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(attn_dim * 4, attn_dim),
                nn.Dropout(dropout),
            )

        self.drug_proj_attn = proj(drug_input_dim, attn_dim)
        self.prot_proj_attn = proj(prot_input_dim, attn_dim)
        self.drug_proj_bil = proj(drug_input_dim, bilinear_dim)
        self.prot_proj_bil = proj(prot_input_dim, bilinear_dim)

        self.drug_to_prot_attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.prot_to_drug_attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.drug_attn_norm = nn.LayerNorm(attn_dim)
        self.prot_attn_norm = nn.LayerNorm(attn_dim)
        self.drug_ffn = ffn()
        self.drug_ffn_norm = nn.LayerNorm(attn_dim)
        self.prot_ffn = ffn()
        self.prot_ffn_norm = nn.LayerNorm(attn_dim)

        self.bilinear = nn.Bilinear(bilinear_dim, bilinear_dim, bilinear_dim)
        self.bilinear_post = nn.Sequential(nn.BatchNorm1d(bilinear_dim), nn.ReLU(), nn.Dropout(dropout))

        self.attn_path_norm = nn.LayerNorm(attn_dim * 2)
        self.bil_path_norm = nn.LayerNorm(bilinear_dim)
        self.regressor = nn.Sequential(
            nn.Linear(attn_dim * 2 + bilinear_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, drug_tokens, drug_mask, prot_tokens, prot_mask):
        if self.training and self.embedding_noise_std > 0:
            drug_tokens = drug_tokens + torch.randn_like(drug_tokens) * self.embedding_noise_std
            prot_tokens = prot_tokens + torch.randn_like(prot_tokens) * self.embedding_noise_std

        d_attn = self.drug_proj_attn(drug_tokens)
        p_attn = self.prot_proj_attn(prot_tokens)
        d_bil = self.drug_proj_bil(drug_tokens)
        p_bil = self.prot_proj_bil(prot_tokens)
        drug_pad, prot_pad = drug_mask == 0, prot_mask == 0

        # Path A: cross-attention transformer blocks
        d_att, _ = self.drug_to_prot_attn(d_attn, p_attn, p_attn, key_padding_mask=prot_pad, need_weights=False)
        d_attn = self.drug_attn_norm(d_attn + d_att)
        d_attn = self.drug_ffn_norm(d_attn + self.drug_ffn(d_attn))
        p_att, _ = self.prot_to_drug_attn(p_attn, d_attn, d_attn, key_padding_mask=drug_pad, need_weights=False)
        p_attn = self.prot_attn_norm(p_attn + p_att)
        p_attn = self.prot_ffn_norm(p_attn + self.prot_ffn(p_attn))
        attn_signal = self.attn_path_norm(
            torch.cat([masked_mean(d_attn, drug_mask.bool()), masked_mean(p_attn, prot_mask.bool())], dim=-1)
        )

        # Path B: bilinear fusion on pooled projections
        bil = self.bilinear(masked_mean(d_bil, drug_mask.bool()), masked_mean(p_bil, prot_mask.bool()))
        bil_signal = self.bil_path_norm(self.bilinear_post(bil))

        if self.training and self.pathway_drop_prob > 0:
            r = torch.rand(1).item()
            if r < self.pathway_drop_prob:
                attn_signal, bil_signal = torch.zeros_like(attn_signal), bil_signal * 2.0
            elif r < 2 * self.pathway_drop_prob:
                attn_signal, bil_signal = attn_signal * 2.0, torch.zeros_like(bil_signal)

        return self.regressor(torch.cat([attn_signal, bil_signal], dim=-1)).squeeze(-1)
