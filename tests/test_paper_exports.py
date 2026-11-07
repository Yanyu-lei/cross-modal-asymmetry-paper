"""The clarified exports retain their gallery scope, statistics and bin membership."""

import numpy as np
import pandas as pd
import pytest

from src import file_paths, load_results, paper_numbers, settings


def _write_quintile_inputs(folder, dataset, galleries, image_q5_hits, caption_q5_hits):
    """Write one model's retrieval and damage files with ten pairs, so each quintile holds two pairs."""
    shared = {"model": "openai_clip_b32", "seed": 0, "spoke": "retrieval_per_pair"}
    rows, damage = [], []
    for pair in range(10):
        image_hit = image_q5_hits[pair - 8] if pair >= 8 else 1
        caption_hit = caption_q5_hits[pair - 8] if pair >= 8 else 1
        for gallery in galleries:
            rows.append({**shared, "pair_id": pair, "vision_corruption": "gaussian_noise", "image_severity": 1,
                         "text_corruption": "none", "text_severity": 0, "metric": "per_pair_recall_at_1_i2t",
                         "value": image_hit, "n_eval": gallery})
            rows.append({**shared, "pair_id": pair, "vision_corruption": "none", "image_severity": 0,
                         "text_corruption": "mask", "text_severity": 1, "metric": "per_pair_recall_at_1_t2i",
                         "value": caption_hit, "n_eval": gallery})
        amount = (pair + 1) / 20
        damage.append({"seed": 0, "pair_id": pair, "modality": "image", "corruption_type": "gaussian_noise", "severity": 1,
                       "ssim": 1 - amount, "psnr": 30.0, "norm_edit_distance": None, "bleu": None})
        damage.append({"seed": 0, "pair_id": pair, "modality": "text", "corruption_type": "mask", "severity": 1,
                       "ssim": None, "psnr": None, "norm_edit_distance": amount, "bleu": 1 - amount})
    # These rows fall outside the headline gallery or the bins, so they must not move any quintile edge.
    outside = settings.HEADLINE_GALLERY_SIZE + 500
    damage.append({"seed": 0, "pair_id": outside, "modality": "image", "corruption_type": "gaussian_noise", "severity": 1,
                   "ssim": 0.0, "psnr": 5.0, "norm_edit_distance": None, "bleu": None})
    damage.append({"seed": 0, "pair_id": 0, "modality": "text", "corruption_type": "typo", "severity": 1,
                   "ssim": None, "psnr": None, "norm_edit_distance": 1.0, "bleu": 0.0})
    pd.DataFrame(rows).to_csv(folder / f"{dataset}_2000_pairs_retrieval.csv", index=False)
    pd.DataFrame(damage).to_csv(folder / f"{dataset}_2000_pairs_retrieval_damage.csv", index=False)


def _build_quintiles(monkeypatch, folder) -> pd.DataFrame:
    monkeypatch.setattr(paper_numbers, "DATASETS", ("coco", "flickr30k"))
    monkeypatch.setattr(file_paths, "run_output", lambda run, seed=None: folder / f"{run}.csv")
    monkeypatch.setattr(file_paths, "damage_output", lambda run: folder / f"{run}_damage.csv")
    monkeypatch.setattr(file_paths, "ROOT", folder)
    monkeypatch.setattr(file_paths, "TAB_QUINTILES_GALLERY1000", folder / "quintiles.csv")
    load_results._headline_per_query_rows.cache_clear()
    load_results.headline_damage.cache_clear()
    try:
        paper_numbers.build_quintiles_at_headline_gallery()
    finally:
        load_results._headline_per_query_rows.cache_clear()
        load_results.headline_damage.cache_clear()
    return pd.read_csv(folder / "quintiles.csv")


def test_headline_gallery_quintiles_cover_both_datasets_at_one_gallery(monkeypatch, tmp_path):
    headline, full = settings.HEADLINE_GALLERY_SIZE, settings.HEADLINE_GALLERY_SIZE * 2
    _write_quintile_inputs(tmp_path, "coco", [headline], (1, 1), (1, 0))
    _write_quintile_inputs(tmp_path, "flickr30k", [headline], (1, 0), (1, 1))
    # Full-gallery rows with the opposite hits would change the means if they were read.
    full_rows = pd.read_csv(tmp_path / "coco_2000_pairs_retrieval.csv").assign(n_eval=full, value=0)
    full_rows.to_csv(tmp_path / "coco_2000_pairs_retrieval.csv", mode="a", header=False, index=False)
    table = _build_quintiles(monkeypatch, tmp_path)
    assert list(table.columns) == ["dataset", "text_ruler", "model", "quintile", "i2t", "t2i", "ratio"]
    assert len(table) == 2 * 2 * 5
    q5 = table[table["quintile"] == "Q5"].set_index(["dataset", "text_ruler"])
    for ruler in paper_numbers.TEXT_RULERS:
        assert q5.loc[("coco", ruler), ["i2t", "t2i", "ratio"]].tolist() == [1.0, 0.5, 2.0]
        assert q5.loc[("flickr30k", ruler), ["i2t", "t2i", "ratio"]].tolist() == [0.5, 1.0, 0.5]


def test_headline_gallery_quintiles_report_a_dataset_without_those_rows(monkeypatch, tmp_path):
    _write_quintile_inputs(tmp_path, "coco", [settings.HEADLINE_GALLERY_SIZE], (1, 1), (1, 0))
    _write_quintile_inputs(tmp_path, "flickr30k", [settings.HEADLINE_GALLERY_SIZE * 2], (1, 1), (1, 1))
    try:
        table = _build_quintiles(monkeypatch, tmp_path)
        assert set(table["dataset"]) == {"coco"}
        assert any("flickr30k" in message for message in load_results.SKIPPED)
    finally:
        load_results.SKIPPED.clear()


def test_recall_export_labels_the_gallery_and_seed_deviation(monkeypatch, tmp_path):
    rows = []
    for gallery, values in ((1000, (0.1, 0.2, 0.3)), (2000, (0.9, 0.9, 0.9))):
        for seed, value in enumerate(values):
            rows.append({"model": "openai_clip_b32", "seed": seed, "vision_corruption": "none",
                         "text_corruption": "none", "image_severity": 0, "text_severity": 0,
                         "spoke": "retrieval_gallery_sweep", "metric": f"recall_at_1_i2t_gallery{gallery}",
                         "value": value})
    original_read = pd.read_csv
    monkeypatch.setattr(paper_numbers, "DATASETS", ("coco",))
    monkeypatch.setattr(paper_numbers, "OBJECT_CUTOUT_RUN", "run_that_was_not_saved")
    monkeypatch.setattr(paper_numbers.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame(rows))
    monkeypatch.setattr(file_paths, "ROOT", tmp_path)
    monkeypatch.setattr(file_paths, "TAB_RECALL_AT_K", tmp_path / "recall.csv")
    paper_numbers.build_recall_at_k_table()
    table = original_read(tmp_path / "recall.csv")
    assert len(table) == 1
    assert table.loc[0, "gallery_size"] == 1000
    assert table.loc[0, "metric"] == "recall_at_1_i2t"
    assert table.loc[0, ["seed_0", "seed_1", "seed_2"]].tolist() == [0.1, 0.2, 0.3]
    assert table.loc[0, "mean"] == pytest.approx(0.2)
    assert table.loc[0, "seed_std"] == pytest.approx(0.1)
    assert "std" not in table.columns


@pytest.mark.parametrize("words,edges", [
    ([0, 1, 2, 3, 4, 5], [0, 1.5, 3, 5]),
    ([7, 7, 8, 9, 10, 11, 12, 12], [7, 9, 10.5, 12]),
])
def test_caption_word_bounds_preserve_membership_and_values(words, edges):
    edges = np.asarray(edges, dtype=float)
    table = pd.DataFrame({"words": words, "value": np.arange(len(words), dtype=float) / 7})
    table["len_bin"] = pd.cut(table["words"], edges, include_lowest=True)
    grouped = table.groupby("len_bin", observed=True)["value"].mean().reset_index()
    exported = paper_numbers._caption_word_bounds(grouped, edges)
    np.testing.assert_array_equal(exported["value"], grouped["value"])
    for position, interval in enumerate(grouped["len_bin"]):
        bounds = exported.iloc[position]
        expected = table["len_bin"] == interval
        actual = table["words"].between(bounds["caption_words_min"], bounds["caption_words_max"])
        pd.testing.assert_series_equal(actual, expected, check_names=False)
    assert pd.api.types.is_integer_dtype(exported["caption_words_min"])
    assert pd.api.types.is_integer_dtype(exported["caption_words_max"])
