"""Define the input, configuration and result paths relative to the repository root."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL_OUTPUTS = RESULTS / "model_outputs"
MEASURED_DAMAGE = RESULTS / "measured_damage"
PAPER = RESULTS / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
NUMBERS = PAPER / "numbers"

CONFIGS = ROOT / "configs"
EXPERIMENT_SETTINGS = CONFIGS / "experiment_settings.yaml"
MODEL_SETTINGS = CONFIGS / "model_settings.yaml"
CORRUPTION_SEVERITIES = CONFIGS / "corruption_severities.yaml"
PAPER_RUNS = CONFIGS / "paper_runs.yaml"

# The released pair lists contain IDs, captions and image hashes. Manifests with embedded
# images stay local and are rebuilt by scripts/build_manifests.py.
DATASETS = ("coco", "flickr30k")


def manifest(dataset: str) -> Path:
    return DATA / f"{dataset}_2000_pairs_manifest.json"


def full_manifest(dataset: str) -> Path:
    return DATA / f"{dataset}_2000_pairs_full.json"


COCO_OBJECT_OUTLINES = DATA / "coco_2000_pairs_object_outlines.json"
COCO_ANNOTATIONS = DATA / "coco_annotations"  # Keep the downloaded annotations for rebuilds.


# Split the fidelity and retention outputs by seed to stay below GitHub's 100 MB per-file limit.
def run_output(run: str, seed: int | None = None) -> Path:
    return MODEL_OUTPUTS / (f"{run}_seed{seed}.csv" if seed is not None else f"{run}.csv")


def text_pooling_ablation_output(dataset: str) -> Path:
    return MODEL_OUTPUTS / f"{dataset}_300_pairs_standard_vs_mean_pooling.csv"


def damage_output(run: str) -> Path:
    name = run.replace("_fidelity_retention", "").replace("_retrieval", "")
    return MEASURED_DAMAGE / f"{name}_damage.csv"


# The achieved object cutout areas need the images, so they are saved for rebuilds without them.
OBJECT_CUTOUT_AREAS = MEASURED_DAMAGE / "coco_2000_pairs_object_cutout_area.csv"


def part_dir(run: str) -> Path:
    return MODEL_OUTPUTS / "parts" / run


TEXT_DEPTH_DIAGNOSTIC = MODEL_OUTPUTS / "text_depth_diagnostic.csv"


def fig_retention_by_text_pooling(dataset: str) -> Path:
    return FIGURES / f"retention_by_text_pooling_{dataset}.pdf"


def tab_recall_at_k(dataset: str) -> Path:
    return TABLES / f"recall_at_1_5_10_{dataset}.tex"


def tab_similarity(dataset: str) -> Path:
    return TABLES / f"shared_space_similarity_{dataset}.tex"


FIG_RECALL_BY_DAMAGE_QUINTILE = FIGURES / "recall_at_1_by_damage_quintile.pdf"
FIG_RECALL_BY_DAMAGE_QUINTILE_DATA = FIGURES / "recall_at_1_by_damage_quintile_data.csv"
FIG_RECALL_BY_DAMAGE_QUINTILE_CLEAN = FIGURES / "recall_at_1_by_damage_quintile_clean_overlay.csv"
FIG_JOINT_HEATMAPS = FIGURES / "retention_joint_noise_mask_heatmaps.pdf"
FIG_JOINT_HEATMAPS_DATA = FIGURES / "retention_joint_noise_mask_heatmaps_data.csv"
FIG_FIDELITY_BY_DEPTH = FIGURES / "fidelity_by_layer_depth.pdf"
FIG_FIDELITY_BY_DEPTH_DATA = FIGURES / "fidelity_by_layer_depth_data.csv"
FIG_RETENTION_BY_TEXT_POOLING_DATA = FIGURES / "retention_by_text_pooling_data.csv"
FIG_RECALL_BY_CORRUPTION_SEV5 = FIGURES / "recall_at_1_by_corruption_sev5.pdf"
FIG_RECALL_BY_CORRUPTION_SEV5_DATA = FIGURES / "recall_at_1_by_corruption_sev5_data.csv"
FIG_PSNR_CROSSCHECK = FIGURES / "recall_at_1_psnr_binned_crosscheck.pdf"
FIG_PSNR_CROSSCHECK_DATA = FIGURES / "recall_at_1_psnr_binned_crosscheck_data.csv"
FIG_RECALL_BY_TYPO_SEVERITY = FIGURES / "recall_at_1_by_typo_severity.pdf"
FIG_RECALL_BY_TYPO_SEVERITY_DATA = FIGURES / "recall_at_1_by_typo_severity_data.csv"
FIG_RECALL_BY_OBJECT_CUTOUT_SEVERITY = FIGURES / "recall_at_1_by_object_cutout_severity.pdf"
FIG_RECALL_BY_OBJECT_CUTOUT_SEVERITY_DATA = FIGURES / "recall_at_1_by_object_cutout_severity_data.csv"
FIG_RECALL_BY_DISPLACEMENT_BIN = FIGURES / "recall_at_1_by_displacement_bin.csv"
FIG_RECALL_BY_DISPLACEMENT_BIN_BY_MODEL = FIGURES / "recall_at_1_by_displacement_bin_per_model_and_corruption.csv"

TAB_SPECS = TABLES / "specs_5_models.tex"
TAB_Q1_Q5_VS_CLEAN = TABLES / "recall_at_1_q1_q5_vs_clean.tex"
TAB_Q1_Q5_VS_CLEAN_DATA = TABLES / "recall_at_1_q1_q5_vs_clean_data.csv"
TAB_CORRUPTION_ROLES = TABLES / "corruption_definitions_and_roles.tex"
TAB_SEVERITY_SETTINGS = TABLES / "corruption_severity_settings.tex"
TAB_STANDARD_VS_MEAN_POOLING = TABLES / "retention_standard_vs_mean_pooling.tex"
TAB_STANDARD_VS_MEAN_POOLING_DATA = TABLES / "retention_standard_vs_mean_pooling.csv"
TAB_INPUT_DAMAGE = TABLES / "input_damage_by_corruption_severity.tex"
TAB_INPUT_DAMAGE_DATA = TABLES / "input_damage_by_corruption_severity.csv"
TAB_DOMINANCE = TABLES / "dominance_regression_r2.tex"
TAB_DOMINANCE_DATA = TABLES / "dominance_regression_r2.csv"
TAB_PECORE_TRUNCATION = TABLES / "pecore_caption_truncation.tex"
TAB_PECORE_TRUNCATION_DATA = TABLES / "pecore_caption_truncation.csv"
TAB_PECORE_TRUNCATION_ROBUSTNESS = TABLES / "pecore_q5_without_truncated_captions.csv"
TAB_RECALL_AT_K = TABLES / "recall_at_1_5_10_by_corruption_severity.csv"
TAB_QUINTILES_GALLERY1000 = TABLES / "recall_at_1_by_damage_quintile_gallery1000.csv"
TAB_QUINTILES_GALLERY2000 = TABLES / "recall_at_1_by_damage_quintile_gallery2000.csv"
TAB_QUINTILES_WITH_TYPO = TABLES / "recall_at_1_by_damage_quintile_gallery2000_with_typo.csv"
TAB_TYPO_DAMAGE_PERCENTILE = TABLES / "typo_damage_percentile.csv"
TAB_OBJECT_CUTOUT_DAMAGE_AND_AREA = TABLES / "object_cutout_damage_and_area.csv"
TAB_OBJECT_CUTOUT_DAMAGE_AND_AREA_TEX = TABLES / "object_cutout_damage_and_area.tex"
TAB_GALLERY_SIZE_SWEEP = TABLES / "noise5_mask5_ratio_by_gallery_size.csv"
TAB_GALLERY_SIZE_SWEEP_TEX = TABLES / "noise5_mask5_ratio_by_gallery_size.tex"
TAB_CAPTION_LENGTH = TABLES / "recall_at_1_ratio_by_caption_length.csv"
TAB_CAPTION_LENGTH_PER_MODEL = TABLES / "recall_at_1_ratio_by_caption_length_per_model.csv"
TAB_CAPTION_LENGTH_TEX = TABLES / "recall_at_1_ratio_by_caption_length.tex"
TAB_SIMILARITY_BY_CORRUPTION = TABLES / "shared_space_similarity_by_corruption.csv"
TAB_RECALL_BY_SIMILARITY_BAND = TABLES / "recall_at_1_by_similarity_band.csv"
TAB_RECALL_BY_SIMILARITY_BAND_TEX = TABLES / "recall_at_1_by_similarity_band.tex"
TAB_DISPLACEMENT_BINS_TEX = TABLES / "recall_at_1_by_displacement_bin.tex"
TAB_DISPLACEMENT_BY_CORRUPTION = TABLES / "recall_at_1_by_displacement_bin_per_corruption.csv"
TAB_DISPLACEMENT_BY_CORRUPTION_TEX = TABLES / "recall_at_1_by_displacement_bin_per_corruption.tex"
TAB_DISPLACEMENT_RESIDUAL = TABLES / "recall_at_1_displacement_residual_by_model.csv"
TAB_DISPLACEMENT_RESIDUAL_TEX = TABLES / "recall_at_1_displacement_residual_by_model.tex"
TAB_Q5_RATIO_BY_PROTOCOL = TABLES / "q5_ratio_by_protocol.csv"
TAB_Q5_RATIO_BY_PROTOCOL_TEX = TABLES / "q5_ratio_by_protocol.tex"

NUM_PAIRED_TESTS = NUMBERS / "i2t_vs_t2i_paired_tests_by_quintile.csv"
NUM_CROSS_SEED_SPEARMAN = NUMBERS / "cross_seed_spearman_cell_means.csv"
NUM_RETENTION_BY_DAMAGE_QUINTILE = NUMBERS / "retention_margin_by_damage_quintile.csv"
NUM_SSIM_Q4_Q5_COMPOSITION = NUMBERS / "ssim_q4_q5_composition_by_corruption.csv"
NUM_NOISE5_VS_MASK5_GALLERY2000 = NUMBERS / "recall_at_1_noise5_vs_mask5_gallery2000.csv"
NUM_RETENTION_IMAGE_MINUS_CAPTION = NUMBERS / "retention_image_minus_caption_by_level.csv"
NUM_CAPTION_QUINTILE_COMPOSITION = NUMBERS / "caption_quintile_composition_and_unchanged_share.csv"
NUM_TYPO_FIDELITY_RETENTION = NUMBERS / "typo_fidelity_and_retention.csv"
NUM_OBJECT_CUTOUT_BY_LEVEL = NUMBERS / "object_cutout_against_captions_by_level.csv"
NUM_OBJECT_CUTOUT_ON_QUINTILE_EDGE = NUMBERS / "object_cutout_on_headline_q5_edge.csv"
NUM_UNDRAWN_INTERVAL_SIZES = NUMBERS / "largest_interval_half_widths.csv"
