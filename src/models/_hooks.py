"""Forward-hook helper: capture outputs of selected blocks during one forward pass.

Used by both backends to pull intermediate hidden states without
relying on framework-specific `forward_intermediates` APIs (which differ between
open_clip versions and aren't available in PE-Core).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


@contextmanager
def capture_block_outputs(blocks: nn.ModuleList, indices: Iterable[int]):
    """Yield a dict {idx: tensor} that gets populated when the model runs forward.

    Block outputs are detached and kept on whatever device the model produced them on.
    Caller is responsible for moving to CPU if needed (we keep on-device by default
    so cosine ops can run on GPU/MPS).
    """
    captures: Dict[int, torch.Tensor] = {}
    handles = []
    indices = list(indices)

    def _make_hook(idx: int):
        def _hook(_module, _inp, out):
            # open_clip ResidualAttentionBlock returns Tensor; timm Block returns Tensor;
            # PE-Core ResidualAttentionBlock returns Tensor. If a tuple is ever returned
            # we take the first element.
            if isinstance(out, tuple):
                out = out[0]
            captures[idx] = out.detach()
        return _hook

    for idx in indices:
        if idx < 0 or idx >= len(blocks):
            raise IndexError(f"Block index {idx} out of range for ModuleList of length {len(blocks)}")
        handles.append(blocks[idx].register_forward_hook(_make_hook(idx)))

    try:
        yield captures
    finally:
        for h in handles:
            h.remove()
