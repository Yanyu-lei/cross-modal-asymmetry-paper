# Synthetic data with known asymmetry; matched-quintile calibration must recover it.
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.calibration_test import matched_quintile_diffs


KNOWN_MARGIN = 0.20  # image_value - text_value at every damage level


def _synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    img_dmg = rng.uniform(0, 1, n)
    txt_dmg = rng.uniform(0, 1, n)
    # value decreases linearly with damage, with a constant +0.20 bias on image side
    img_val = 0.7 - 0.3 * img_dmg + KNOWN_MARGIN
    txt_val = 0.7 - 0.3 * txt_dmg
    image_side = pd.DataFrame({"damage_ssim": img_dmg, "value": img_val})
    text_side = pd.DataFrame({"damage_bleu": txt_dmg, "value": txt_val})
    return image_side, text_side


def test_all_five_quintile_diffs_are_positive():
    img, txt = _synthetic()
    out = matched_quintile_diffs(img, txt, image_damage_col="damage_ssim", text_damage_col="damage_bleu")
    assert (out["diff_image_minus_text"] > 0).all()


def test_diffs_close_to_known_margin():
    img, txt = _synthetic(n=2000, seed=1)
    out = matched_quintile_diffs(img, txt, image_damage_col="damage_ssim", text_damage_col="damage_bleu")
    np.testing.assert_allclose(out["diff_image_minus_text"].mean(), KNOWN_MARGIN, atol=0.02)


def test_no_asymmetry_yields_diffs_near_zero():
    rng = np.random.default_rng(2)
    n = 2000
    dmg_a = rng.uniform(0, 1, n); dmg_b = rng.uniform(0, 1, n)
    val_a = 0.7 - 0.3 * dmg_a; val_b = 0.7 - 0.3 * dmg_b
    img = pd.DataFrame({"damage_ssim": dmg_a, "value": val_a})
    txt = pd.DataFrame({"damage_bleu": dmg_b, "value": val_b})
    out = matched_quintile_diffs(img, txt, image_damage_col="damage_ssim", text_damage_col="damage_bleu")
    np.testing.assert_allclose(out["diff_image_minus_text"].mean(), 0.0, atol=0.02)
