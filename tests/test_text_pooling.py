"""Mean pooling stops at the end of the caption and skips the padding after it."""
from __future__ import annotations

import torch

from src.load_models import pool_text_tokens


def test_mean_pooling_stops_at_the_last_token_that_is_not_padding():
    # SigLIP 2 pads with id 0. The first caption has three real tokens and the second has five.
    tokens = torch.tensor([[7, 8, 9, 0, 0], [4, 5, 6, 7, 8]])
    h = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    pooled = pool_text_tokens(h, tokens, "mean", end="last_non_pad", pad_id=0)
    torch.testing.assert_close(pooled[0], h[0, :3].mean(dim=0))
    torch.testing.assert_close(pooled[1], h[1].mean(dim=0))


def test_mean_pooling_stops_at_the_end_of_text_token():
    # In CLIP's vocabulary the end-of-text token has the highest id, here 99.
    tokens = torch.tensor([[1, 5, 99, 0, 0]])
    h = torch.randn(1, 5, 4)
    pooled = pool_text_tokens(h, tokens, "mean", end="argmax")
    torch.testing.assert_close(pooled[0], h[0, :3].mean(dim=0))
