# Wrapper for the matched-quintile calibration test. One function, testable
# on synthetic data with a known cross-modal asymmetry.
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .quintile import assign_quintile, QUINTILE_LABELS


def matched_quintile_diffs(
    image_side: pd.DataFrame,
    text_side: pd.DataFrame,
    *,
    image_damage_col: str,
    text_damage_col: str,
    value_col: str = "value",
    quintile_labels: Sequence[str] = QUINTILE_LABELS,
) -> pd.DataFrame:
    """Within each modality, assign quintiles by damage; per matched quintile
    return image-side mean, text-side mean, and (image - text) diff.
    Returns a DataFrame indexed by quintile."""
    img = image_side.copy()
    txt = text_side.copy()
    img["q"] = assign_quintile(img[image_damage_col], labels=quintile_labels)
    txt["q"] = assign_quintile(txt[text_damage_col], labels=quintile_labels)

    rows = []
    for q in quintile_labels:
        i_v = img.loc[img["q"] == q, value_col].to_numpy(dtype=float)
        t_v = txt.loc[txt["q"] == q, value_col].to_numpy(dtype=float)
        rows.append({
            "quintile": q,
            "image_n": len(i_v),
            "text_n": len(t_v),
            "image_mean": float(i_v.mean()) if i_v.size else float("nan"),
            "text_mean": float(t_v.mean()) if t_v.size else float("nan"),
            "diff_image_minus_text": (float(i_v.mean()) - float(t_v.mean())) if i_v.size and t_v.size else float("nan"),
        })
    return pd.DataFrame(rows).set_index("quintile")
