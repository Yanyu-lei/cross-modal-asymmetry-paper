"""Each of the five models runs on the GPU, when there is one, so a bad install shows up before the run."""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from src import settings
from src.load_models import load_model

CAPTIONS = ["a dog runs on the grass", "a man rides a bicycle along a narrow road"]


def encode_everything(model: str) -> None:
    """Encode a picture and captions at the model's own depths, under both text poolings."""
    adapter = load_model(model)
    cfg = settings.MODEL_SETTINGS[model]
    image_depths = adapter.proportional_depths(cfg["image_depth_fractions"], "image")
    text_depths = adapter.proportional_depths(cfg["text_depth_fractions"], "text")
    image = adapter.encode_image([Image.new("RGB", (640, 480), "gray")], image_depths)
    text = adapter.encode_text(CAPTIONS, text_depths)
    poolings = adapter.encode_text_both_poolings(CAPTIONS)
    for tensor in (image.pooled, *image.patch_states, text.pooled, *text.pooled_states, *poolings.values()):
        assert torch.isfinite(tensor).all(), model
    # Standard pooling must give the model's own caption embedding.
    assert torch.allclose(poolings["standard"].float(), text.pooled.float(), atol=1e-3), model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("model", settings.MODELS)
def test_model_runs_on_the_gpu(model):
    encode_everything(model)
