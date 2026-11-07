"""The paper's numerical summaries keep their definitions and scope."""
from __future__ import annotations

import re
from functools import partial

import numpy as np
import pandas as pd
import pytest

from src import file_paths, load_results, paper_appendix_tables, paper_figures, paper_numbers, settings
from src.bootstrap_and_quintiles import bootstrap_ci, bootstrap_group_statistics
from src.paper_figures import MODELS


def _column_count(tex: str) -> int:
    """Count the columns of a tabular specification, treating each p{...} as one column."""
    spec = re.search(r"\\begin\{tabular\}\{(.*)\}", tex).group(1)
    return len(re.findall(r"p\{[^}]*\}|[lrc]", re.sub(r"@\{[^}]*\}", "", spec)))


def _assert_rows_fit(tex: str) -> None:
    """Check that every data row of every table in a file fills all of its columns."""
    tables = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", tex, flags=re.S)
    assert tables
    for table in tables:
        n = _column_count(table)
        for line in table.splitlines():
            line = line.strip()
            if line.endswith((r"\\", r"\tabularnewline")) and "multicolumn" not in line:
                assert line.count("&") + 1 == n, line


def _rows(values: dict[tuple[int, int], list[float]]) -> pd.DataFrame:
    return pd.DataFrame([{"seed": seed, "pair_id": pair, "value": value}
                         for (seed, pair), group in values.items() for value in group])


def test_one_group_statistic_matches_the_single_interval():
    rng = np.random.default_rng(3)
    rows = pd.DataFrame({"seed": np.repeat([0, 1, 2], 40), "pair_id": np.tile(np.arange(20).repeat(2), 3), "value": rng.integers(0, 2, 120)})
    expected = bootstrap_ci(rows, n_boot=300)
    assert bootstrap_group_statistics({"a": rows}, {"mean": lambda g: g["a"]}, n_boot=300)["mean"] == pytest.approx(expected)


def test_group_statistics_resample_the_same_pairs_in_every_group():
    first = _rows({(seed, pair): [float(pair + 1)] for seed in (0, 1) for pair in range(8)})
    second = first.assign(value=first["value"] / 2)
    point, lo, hi = bootstrap_group_statistics({"a": first, "b": second}, {"ratio": lambda g: g["a"] / g["b"]}, n_boot=200)["ratio"]
    assert (point, lo, hi) == pytest.approx((2.0, 2.0, 2.0))


def test_group_statistics_weight_pairs_by_their_rows():
    rows = _rows({(0, 0): [1.0, 1.0, 1.0], (0, 1): [0.0]})
    point, lo, hi = bootstrap_group_statistics({"a": rows}, {"mean": lambda g: g["a"]}, n_boot=500)["mean"]
    assert point == pytest.approx(0.75)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.0)


def test_damage_columns_rank_lower_psnr_as_more_damage():
    table = load_results._with_damage(pd.DataFrame({"ssim": [0.9], "psnr": [30.0], "bleu": [0.25]}))
    assert table.loc[0, ["damage_ssim", "damage_psnr", "damage_bleu"]].tolist() == pytest.approx([0.1, -30.0, 0.75])


def test_pooling_summary_names_each_reference_level(monkeypatch):
    """Clean and severity-1 margins lead to different drops, and each relative drop uses its own start."""
    margins = {("standard", 0): 0.40, ("standard", 1): 0.30, ("standard", 5): 0.10, ("mean", 0): 0.20, ("mean", 1): 0.16, ("mean", 5): 0.08}
    rows = [{"model": m, "seed": seed, "pair_id": pair, "text_pooling": pooling, "text_severity": level, "value": value}
            for m in MODELS for seed in (0, 1) for pair in range(4) for (pooling, level), value in margins.items()]
    monkeypatch.setattr(paper_figures, "text_pooling_ablation", lambda dataset: pd.DataFrame(rows))
    monkeypatch.setattr(paper_figures, "bootstrap_ci", partial(bootstrap_ci, n_boot=20))
    monkeypatch.setattr(paper_figures, "bootstrap_group_statistics", partial(bootstrap_group_statistics, n_boot=20))
    paper_figures.pooling_summary.cache_clear()
    paper_figures._pooling_margins_by_level.cache_clear()
    try:
        row = paper_figures.pooling_summary("coco").iloc[0]
    finally:
        paper_figures.pooling_summary.cache_clear()
        paper_figures._pooling_margins_by_level.cache_clear()
    assert row["standard_drop"] == pytest.approx(0.20) and row["mean_drop"] == pytest.approx(0.08)
    assert row["drop_reduction_pct"] == pytest.approx(60.0)
    assert row["standard_rel_drop_pct"] == pytest.approx(100 * 0.20 / 0.30)
    assert row["mean_rel_drop_pct"] == pytest.approx(50.0)
    assert row["standard_drop_from_clean"] == pytest.approx(0.30)
    assert row["standard_rel_drop_from_clean_pct"] == pytest.approx(75.0)
    assert row["rel_drop_mean_minus_standard"] == pytest.approx(50.0 - 100 * 0.20 / 0.30)


def _protocol_inputs():
    """Build hits and damage for ten pairs, where two captions tie at the Q5 edge and fail while the Q5 caption succeeds."""
    hits, damage = [], []
    caption_damage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9]
    for pair, amount in enumerate(caption_damage):
        for m in MODELS:
            hits.append({"model": m, "seed": 0, "pair_id": pair, "vision_corruption": "gaussian_noise", "image_severity": 1,
                         "text_corruption": "none", "text_severity": 0, "metric": "per_pair_recall_at_1_i2t", "value": 1})
            hits.append({"model": m, "seed": 0, "pair_id": pair, "vision_corruption": "none", "image_severity": 0,
                         "text_corruption": "mask", "text_severity": 1, "metric": "per_pair_recall_at_1_t2i", "value": 0 if pair in (7, 8) else 1})
        damage.append({"seed": 0, "pair_id": pair, "modality": "image", "corruption_type": "gaussian_noise", "severity": 1,
                       "ssim": 1 - amount, "psnr": 40 - 10 * amount, "norm_edit_distance": np.nan, "bleu": np.nan})
        damage.append({"seed": 0, "pair_id": pair, "modality": "text", "corruption_type": "mask", "severity": 1,
                       "ssim": np.nan, "psnr": np.nan, "norm_edit_distance": amount, "bleu": 1 - amount})
    return pd.DataFrame(hits), load_results._with_damage(pd.DataFrame(damage))


def test_protocol_table_moves_only_the_tied_captions(monkeypatch, tmp_path):
    hits, damage = _protocol_inputs()
    monkeypatch.setattr(paper_numbers, "DATASETS", ("coco",))
    monkeypatch.setattr(paper_numbers, "per_pair_retrieval", lambda dataset: hits)
    monkeypatch.setattr(paper_numbers, "headline_damage", lambda dataset: damage)
    monkeypatch.setattr(paper_numbers, "full_gallery_rows", lambda dataset: hits.assign(spoke="retrieval_per_pair"))
    monkeypatch.setattr(paper_numbers, "full_gallery_damage", lambda dataset: damage)
    monkeypatch.setattr(file_paths, "ROOT", tmp_path)
    monkeypatch.setattr(file_paths, "TAB_Q5_RATIO_BY_PROTOCOL", tmp_path / "protocol.csv")
    paper_numbers.build_q5_ratio_by_protocol()
    table = pd.read_csv(tmp_path / "protocol.csv")
    assert len(table) == (2 * 2 * 2 + 1) * len(MODELS)
    usual = table[(table["gallery_size"] == settings.HEADLINE_GALLERY_SIZE) & (table["image_ruler"] == "ssim")
                  & (table["text_ruler"] == "bleu") & (table["caption_ties_at_q5_edge"] == "lower bin")]
    moved = table[table["caption_ties_at_q5_edge"] == "Q5"]
    assert usual["q5_t2i"].tolist() == pytest.approx([1.0] * len(MODELS))
    assert moved["q5_t2i"].tolist() == pytest.approx([1 / 3] * len(MODELS))
    assert moved["ratio"].tolist() == pytest.approx([3.0] * len(MODELS))
    monkeypatch.setattr(file_paths, "TAB_Q5_RATIO_BY_PROTOCOL_TEX", tmp_path / "protocol.tex")
    tex = paper_appendix_tables.q5_ratio_by_protocol_table()
    _assert_rows_fit(tex)
    assert "ties in Q5" in tex


def test_caption_composition_counts_unchanged_captions(monkeypatch, tmp_path):
    rows = [{"seed": 0, "pair_id": pair, "modality": "text", "corruption_type": "shuffle", "severity": 1, "bleu": 1 - pair / 10,
             "norm_edit_distance": 0.0 if pair == 0 else pair / 10, "ssim": np.nan, "psnr": np.nan} for pair in range(10)]
    damage = load_results._with_damage(pd.DataFrame(rows))
    monkeypatch.setattr(paper_numbers, "datasets_with_headline_rows", lambda: ["coco"])
    monkeypatch.setattr(paper_numbers, "headline_damage", lambda dataset: damage)
    monkeypatch.setattr(file_paths, "ROOT", tmp_path)
    monkeypatch.setattr(file_paths, "NUM_CAPTION_QUINTILE_COMPOSITION", tmp_path / "composition.csv")
    paper_numbers.build_caption_quintile_composition()
    table = pd.read_csv(tmp_path / "composition.csv").set_index("quintile")
    assert table.loc["Q1", "unchanged_rows"] == 1
    assert table.loc["Q1", "unchanged_share_of_quintile"] == pytest.approx(0.5)
    assert table.loc["Q5", "unchanged_rows"] == 0


def test_direction_gap_uses_the_smaller_count_as_each_bins_weight(monkeypatch):
    """One bin has a gap of 1 with weight 3, and the other a gap of 0 with weight 1, so the mean gap is 0.75."""
    monkeypatch.setattr(paper_numbers, "MIN_QUERIES_FOR_CORRUPTION_RATE", 1)
    rows = []
    for m in MODELS:
        for pair in range(3):
            rows.append({"model": m, "seed": 0, "pair_id": pair, "direction": "i2t", "bin": "a", "success": 1})
            rows.append({"model": m, "seed": 0, "pair_id": pair, "direction": "t2i", "bin": "a", "success": 0})
        rows.append({"model": m, "seed": 0, "pair_id": 3, "direction": "i2t", "bin": "b", "success": 1})
        for pair in range(4, 8):
            rows.append({"model": m, "seed": 0, "pair_id": pair, "direction": "t2i", "bin": "b", "success": 1})
    table = paper_numbers._direction_gap_rows(pd.DataFrame(rows), "coco")
    assert table[0]["direction_gap"] == pytest.approx(0.75)
    assert table[0]["bins_with_both_directions"] == 2
    assert table[0]["largest_bin_gap"] == pytest.approx(1.0) and table[0]["smallest_bin_gap"] == pytest.approx(0.0)


def test_saved_object_cutout_areas_stand_in_for_the_image_manifest(monkeypatch, tmp_path):
    saved = pd.DataFrame({"seed": [0], "pair_id": [1], "severity": [2], "area": [0.08639641608391609]})
    saved.to_csv(tmp_path / "areas.csv", index=False)
    monkeypatch.setattr(file_paths, "OBJECT_CUTOUT_AREAS", tmp_path / "areas.csv")
    monkeypatch.setattr(file_paths, "full_manifest", lambda dataset: tmp_path / "absent.json")
    try:
        assert paper_numbers._object_cutout_areas().to_dict("records") == saved.to_dict("records")
        assert not load_results.SKIPPED
        monkeypatch.setattr(file_paths, "OBJECT_CUTOUT_AREAS", tmp_path / "absent.csv")
        assert paper_numbers._object_cutout_areas().empty
        assert any("areas" in message for message in load_results.SKIPPED)
    finally:
        load_results.SKIPPED.clear()


def _built(path) -> pd.DataFrame:
    """Read a built output, or skip a check that waits for the paper build."""
    if not path.exists():
        pytest.skip(f"{path.name} has not been built")
    return pd.read_csv(path)


def test_built_figure_one_agrees_with_the_gallery_1000_file():
    figure = _built(file_paths.FIG_RECALL_BY_DAMAGE_QUINTILE_DATA)
    table = _built(file_paths.TAB_QUINTILES_GALLERY1000)
    table = table[table["text_ruler"] == "bleu"]
    joined = figure.merge(table, on=["dataset", "model", "quintile"], suffixes=("", "_table"))
    assert len(joined) == len(file_paths.DATASETS) * len(MODELS) * 5
    assert joined["i2t"].to_numpy() == pytest.approx(joined["i2t_table"].to_numpy())
    assert joined["t2i"].to_numpy() == pytest.approx(joined["t2i_table"].to_numpy())


def test_built_protocol_table_agrees_with_the_quintile_files():
    protocol = _built(file_paths.TAB_Q5_RATIO_BY_PROTOCOL)
    usual = protocol[(protocol["image_ruler"] == "ssim") & (protocol["caption_ties_at_q5_edge"] == "lower bin")]
    for gallery, path in ((settings.HEADLINE_GALLERY_SIZE, file_paths.TAB_QUINTILES_GALLERY1000), (max(settings.GALLERY_SIZES), file_paths.TAB_QUINTILES_GALLERY2000)):
        table = _built(path)
        table = table[table["quintile"] == "Q5"]
        joined = usual[usual["gallery_size"] == gallery].merge(table, on=["dataset", "text_ruler", "model"], suffixes=("", "_table"))
        assert len(joined) == len(file_paths.DATASETS) * len(MODELS) * 2
        assert joined["ratio"].to_numpy() == pytest.approx(joined["ratio_table"].to_numpy())


def test_built_headline_table_agrees_with_figure_one():
    headline = _built(file_paths.TAB_Q1_Q5_VS_CLEAN_DATA)
    figure = _built(file_paths.FIG_RECALL_BY_DAMAGE_QUINTILE_DATA)
    q5 = figure[figure["quintile"] == "Q5"].merge(headline, on=["dataset", "model"])
    assert (q5["i2t"] / q5["t2i"]).to_numpy() == pytest.approx(q5["q5_ratio"].to_numpy())
    assert ((q5["q5_ratio_lo"] < q5["q5_ratio"]) & (q5["q5_ratio"] < q5["q5_ratio_hi"])).all()


def test_built_severity_5_file_states_its_scope():
    table = _built(file_paths.FIG_RECALL_BY_CORRUPTION_SEV5_DATA)
    assert set(table["dataset"]) == {"coco"}
    assert set(table["gallery_size"]) == {settings.HEADLINE_GALLERY_SIZE}
    corrupted = table[table["kind"] == "corrupted"]
    assert (corrupted["n"] == settings.HEADLINE_GALLERY_SIZE * len(settings.SEEDS)).all()


def test_built_files_hold_different_values_for_each_dataset():
    """A step that read one dataset twice would print the same numbers under both names."""
    compared = 0
    for path in sorted(file_paths.PAPER.rglob("*.csv")):
        table = pd.read_csv(path)
        if "dataset" not in table.columns or table["dataset"].nunique() < 2:
            continue
        values = table.select_dtypes("float")
        coco = np.sort(values[table["dataset"] == "coco"].to_numpy().ravel())
        flickr = np.sort(values[table["dataset"] == "flickr30k"].to_numpy().ravel())
        assert coco.shape != flickr.shape or not np.array_equal(coco, flickr, equal_nan=True), path.name
        compared += 1
    if not compared:
        pytest.skip("no built file covers both datasets")
