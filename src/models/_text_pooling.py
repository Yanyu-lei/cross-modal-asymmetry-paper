"""Alternative text-pooling helpers for the EOT-bottleneck mechanistic probe.

Given a text-encoder hidden state h: (B, T, D) and the model's tokenized input
ids: (B, T), produce four candidate pooled vectors:

  argmax  — h[b, argmax(ids[b])]    (CLIP/PE-Core convention; EOT position)
  mean    — masked mean over the *sentence* tokens [0..eot_pos] inclusive
  last    — h[:, -1]                 (SigLIP convention)
  first   — h[:, 0]

The "mean" path needs to know where the sentence ends. For CLIP-family
tokenizers the EOT id is the highest-value token in the sequence (its position
is `argmax(ids)`). For SigLIP / SigLIP 2 the sequence is right-padded with the
pad token (id 1 for SigLIP 2 SentencePiece tokenizer); the *last non-pad*
position is the meaningful end. We pass `eot_strategy` to control which
end-of-sentence rule to apply.
"""
from __future__ import annotations

from typing import Literal

import torch


def alt_pool(
    h: torch.Tensor,                # (B, T, D)
    tokens: torch.Tensor,           # (B, T)
    pool_type: str,
    *,
    eot_strategy: Literal["argmax", "last_non_pad", "last"] = "argmax",
    pad_id: int = 0,
) -> torch.Tensor:
    """Return (B, D) pooled embedding using `pool_type`.

    eot_strategy controls how the end-of-sentence position is determined for
    the "mean" pool. CLIP-family uses "argmax" (EOT is the highest token id),
    SigLIP-family uses "last_non_pad" (PAD id is fixed, sentence ends at the
    last non-pad position), and "last" just uses (T-1).
    """
    B, T, D = h.shape
    arange_b = torch.arange(B, device=h.device)

    if pool_type == "argmax":
        idx = tokens.argmax(dim=-1)       # (B,)
        return h[arange_b, idx]

    if pool_type == "last":
        return h[:, -1, :]

    if pool_type == "first":
        return h[:, 0, :]

    if pool_type == "mean":
        if eot_strategy == "argmax":
            end = tokens.argmax(dim=-1)   # inclusive end position
        elif eot_strategy == "last_non_pad":
            non_pad = (tokens != pad_id).long()
            # last index where non_pad == 1; if none, use 0
            arange_t = torch.arange(T, device=h.device).unsqueeze(0)  # (1, T)
            end = (non_pad * (arange_t + 1)).max(dim=-1).values - 1   # (B,)
            end = end.clamp(min=0)
        elif eot_strategy == "last":
            end = torch.full((B,), T - 1, device=h.device, dtype=torch.long)
        else:
            raise ValueError(f"unknown eot_strategy: {eot_strategy}")

        arange_t = torch.arange(T, device=h.device).unsqueeze(0)  # (1, T)
        mask = arange_t <= end.unsqueeze(1)                       # (B, T)
        h_masked = h * mask.unsqueeze(-1).to(h.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)  # (B, 1)
        return h_masked.sum(dim=1) / count

    raise ValueError(f"unknown pool_type: {pool_type}")
