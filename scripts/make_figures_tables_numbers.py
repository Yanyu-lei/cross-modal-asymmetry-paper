"""Build the paper's figures, tables and numbers from the saved runs without a GPU.

    python -m scripts.make_figures_tables_numbers --require-all
    python -m scripts.make_figures_tables_numbers tables robustness_checks
    python -m scripts.make_figures_tables_numbers --list

Run every step when no names are given. Report the skipped outputs at the end and fail
the build if any were skipped when --require-all is given.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import file_paths, paper_appendix_tables, paper_figures, paper_numbers, paper_tables  # noqa: E402
from src.load_results import SKIPPED  # noqa: E402

STEPS = {
    "recall_at_1_by_damage_quintile": paper_figures.build_recall_at_1_by_damage_quintile,
    "recall_at_1_psnr_binned_crosscheck": paper_figures.build_recall_at_1_psnr_binned_crosscheck,
    "retention_joint_noise_mask_heatmaps": paper_figures.build_retention_joint_noise_mask_heatmaps,
    "fidelity_by_layer_depth": paper_figures.build_fidelity_by_layer_depth,
    "retention_by_text_pooling": paper_figures.build_retention_by_text_pooling,
    "recall_at_1_by_corruption_sev5": paper_figures.build_recall_at_1_by_corruption_sev5,
    "tables": paper_tables.build_tables,
    "i2t_vs_t2i_paired_tests": paper_numbers.build_i2t_vs_t2i_paired_tests,
    "cross_seed_spearman": paper_numbers.build_cross_seed_spearman,
    "retention_by_damage_quintile": paper_numbers.build_retention_by_damage_quintile,
    "retention_image_minus_caption": paper_numbers.build_retention_image_minus_caption_by_level,
    "typo_fidelity_and_retention": paper_numbers.build_typo_fidelity_and_retention,
    "recall_at_k_table": paper_numbers.build_recall_at_k_table,
    "ssim_quintile_composition": paper_numbers.build_ssim_quintile_composition,
    "caption_quintile_composition": paper_numbers.build_caption_quintile_composition,
    "noise5_mask5_at_full_gallery": paper_numbers.build_noise5_mask5_at_full_gallery,
    "robustness_checks": paper_numbers.build_robustness_checks,
    "quintiles_at_headline_gallery": paper_numbers.build_quintiles_at_headline_gallery,
    "q5_ratio_by_protocol": paper_numbers.build_q5_ratio_by_protocol,
    "object_cutout": paper_numbers.build_object_cutout,
    "recall_at_1_by_displacement_bin": paper_numbers.build_recall_at_1_by_displacement_bin,
    "undrawn_interval_sizes": paper_numbers.build_undrawn_interval_sizes,
    "appendix_tables": paper_appendix_tables.build_appendix_tables,  # This reads the files that the steps above write.
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Make every figure, table and number in results/paper/.")
    parser.add_argument("steps", nargs="*", help="steps to run (default: all of them)")
    parser.add_argument("--list", action="store_true", help="list the steps, then stop")
    parser.add_argument("--require-all", action="store_true", help="fail if any part had to be left out")
    args = parser.parse_args()
    if args.list:
        print("\n".join(STEPS))
        return
    unknown = [s for s in args.steps if s not in STEPS]
    if unknown:
        sys.exit(f"unknown step(s): {', '.join(unknown)}. See --list.")
    for folder in (file_paths.FIGURES, file_paths.TABLES, file_paths.NUMBERS):
        folder.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for name in args.steps or STEPS:
        print(f"[{time.time() - start:6.1f}s] {name}", flush=True)
        STEPS[name]()
    print(f"[{time.time() - start:6.1f}s] done")
    if SKIPPED:
        print("Left out of this build:\n  " + "\n  ".join(SKIPPED))
        if args.require_all:
            sys.exit("The build is incomplete, since --require-all was given.")


if __name__ == "__main__":
    main()
