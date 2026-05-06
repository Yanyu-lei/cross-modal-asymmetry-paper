# Text-side fragility in contrastive vision-language retrieval

This repository contains the analysis pipeline and the canonical aggregated artifacts for the paper. Every headline number, table, and figure is materialised under `results/` and `paper/` from a prior run of the pipeline. Re-running the pipeline end-to-end (model inference + post-inference analyses) requires GPU compute and is documented separately below.

## Setup

Python 3.12 (tested). Standard scientific Python stack plus `open_clip_torch` and `timm`:

```bash
pip install -r requirements.txt
```

PE-Core L/14-336 needs a separate install with `--no-deps` to avoid an upstream `decord==0.6.0` pin that has no Python 3.12 wheel:

```bash
pip install --no-deps git+https://github.com/facebookresearch/perception_models.git
```

Model checkpoints (only needed for re-running inference) download automatically on first use via `open_clip` and `timm`; cache lives in `$HF_HOME` or `~/.cache/huggingface`.

## Reproducing paper claims

Every paper claim's headline number is materialised in a shipped artifact. Each row below names the claim, points at the file(s) that hold it, and lists the column or sub-claim that carries the numerical value. **No inference run is required to verify claims from these files.**

| Paper claim | Where to find it | What to read |
|---|---|---|
| 4.6× to 5.8× matched-Q5 i2t/t2i Recall@1 ratio across the five models (Abstract, §4.1, Figure 1, Table 2) | `results/figures/f1_retrieval_asymmetry.pdf`, `results/tables/t2_headline.tex`; raw values in `results/figures/f1_data.csv` | Q5 i2t and Q5 t2i columns; ratio is Q5 i2t / Q5 t2i per model |
| 25 of 25 (model × quintile) cells significant under hierarchical bootstrap 95% CIs (§4.1) | `results/calibration_recall1_quintile.csv` | every row has `i2t > t2i = True` and `diff_significant = True` |
| Per-quintile gap-growth from 1.01×–1.12× at Q1 to 4.63×–5.80× at Q5 (§4.1) | `results/figures/f1_data.csv` | per-(model, quintile) i2t and t2i means |
| Encoder-level fidelity drops at late depth, severity 5 (§4.2, Figure 4) | `results/figures/f4_layer_fidelity.pdf`; raw values aggregated from the (now-untracked) seed CSVs | per-model image patch cosine and text pooled cosine drops |
| 24% to 75% slope-component bottleneck contribution range (§4.2, Figure 5, Table 5) | `results/figures/f5_two_component_decomposition.pdf`, `results/tables/t5_pooling_probe.tex`; raw values in `results/figures/f5_slope_data.csv` and `results/figures/f5_baseline_data.csv` | drop-reduction percent column = (std drop − mean drop) / std drop |
| Approximately 7× shuffle-to-mask Recall@1 ratio at severity 5 (§4.3, Figure 6) | `results/figures/f6_corruption_breakdown.pdf`; raw values in `results/figures/f6_data.csv` | per-model `text/{mask,shuffle}/corrupted` rows; ratio is shuffle / mask |
| Cross-seed reliability ρ = 0.995, 0.995, 0.996 across the three seed pairs (§3.5) | `results/reliability_cell_mean_summary.csv` | three rows, one per seed pair |
| Headline Q5 paired Wilcoxon W = 0, p < 0.001, mean per-cell Cohen's d = 1.15 (§4.1) | `results/paired_tests_summary.csv` | per-(model, quintile) cells plus an `ALL/Q5` headline row |
| Joint 5×5 (image × text severity) corruption grid (Appendix A, Figure 3) | `results/figures/f3_joint_heatmaps.pdf`; raw values in `results/figures/f3_data.csv` | per-model retention margin per (image_severity, text_severity) cell |
| Dominance-regression sign reversal under measured-damage rescaling (§5, Appendix E, Table 4) | `results/tables/t4_dominance.tex` | $R^2$ columns for ordinal-severity vs measured-damage versions |
| PSNR ruler cross-check, Q5 i2t/t2i ratio 3.0× to 3.9× (Appendix D) | `results/figures/f1_psnr.pdf`, `results/figures/f2_psnr.pdf`, `results/tables/t6_damage_calibration.tex` | F1_psnr and F2_psnr produce monotone Q4-to-Q5 i2t drops |
| PE-Core caption truncation (Appendix F, Table 7) | `results/tables/t7_pecore_truncation.tex` | 1 of 1,873 captions exceeds the 30-token content limit |
| Per-(corruption, severity) input-side damage rulers (Appendix D, Table 6) | `results/figures/f5_baseline_data.csv`, `results/calibration_damage_table.csv`, `results/tables/t6_damage_calibration.tex` | SSIM, PSNR for image; edit distance, BLEU for text |

## Re-running model inference (optional, GPU-required)

Raw inference outputs (`results/seed{0,1,2}/`, `results/per_pair_retrieval_v2/`, `results/pooling_probe/`) and the full image-bytes manifest are not shipped because of repository size constraints. They are regenerable end-to-end on a single A100-class GPU in approximately five hours via the following pipeline.

1. **Rebuild the COCO image-bytes manifest.** The manifest (`results/coco_manifest.json`) is not shipped with this repository because of repository size constraints. Regenerating it is the required first step: the build script fetches the same 2,000 images from MS-COCO captions URLs and produces a sha256-keyed manifest used deterministically by every subsequent stage.

   ```bash
   python -m scripts.build_manifest
   ```

2. **Run inference across 5 models × 3 seeds × all corruption types**:

   ```bash
   python -m scripts.run_full_experiment
   ```

   Populates `results/seed{0,1,2}/seed{N}_results.csv` (image fidelity, text fidelity, match retention spokes) and `results/per_pair_retrieval_v2/per_pair_retrieval_v2.csv` (per-pair Recall@k indicators).

3. **Compute input-side damage rulers** for the main spokes and the retrieval pairs:

   ```bash
   python -m scripts.compute_input_damage
   python -m scripts.compute_retrieval_damage
   ```

4. **Run the pooling-substitution probe** (mean-pool vs standard-pool text retention, used by Figure 5):

   ```bash
   python -m scripts.collect_pooling_probe
   ```

5. **Run the clean-baseline retrieval pass** (used as the reference baseline in Figure 1 and Table 2):

   ```bash
   python -m scripts.compute_clean_baseline_retrieval
   ```

6. **Run the post-inference analyses and rebuild paper artifacts**:

   ```bash
   python -m scripts.analyze_recall1_quintile      # Recall@1 quintile calibration; the 25/25 cells claim
   python -m scripts.severity_calibration          # retention-margin quintile + dominance regression
   python -m scripts.diagnostics_psnr_crosscheck   # Appendix D PSNR cross-check
   python -m scripts.analyze_upper_bound           # mean-pool upper-bound test (Appendix)
   python -m scripts.text_depth_diagnostic         # per-model "early" text-encoder depth fractions
   python -m scripts.build_paper_artifacts         # rebuild figures and tables from canonical CSVs
   ```

`scripts/build_paper_artifacts.py` reads from the seed CSVs, per-pair retrieval CSV, and pooling-probe CSV produced in steps 2 and 4. It will fail with a missing-file error if those raw outputs are not present locally; the shipped pre-rendered PDFs and `.tex` files reflect a prior run.

## Repository structure

```
configs/                  Experiment-level configurations (models, severity, retrieval N, manifest path)
src/
├── corruptions/         Image and text corruption implementations
├── models/              Model loading and pooling-operator wrappers (open_clip + PE-Core backends)
├── spokes/              Five measurement components: image fidelity, text fidelity, match retention, retrieval, layer-wise depth
├── analysis/            Hierarchical bootstrap, significance tests, cross-seed reliability, damage-ruler metrics
├── data/                COCO manifest building and seeded-shuffle pair loader
└── plots/               Per-figure rendering modules (F1-F6) plus shared style/data loaders
scripts/                  Entry-point scripts for inference, post-inference analyses, and paper-artifact rebuilds
results/                  Canonical aggregated CSVs, figure PDFs, table .tex files, metadata-only manifest
tests/                    Unit tests for corruption operators, damage rulers, bootstrap, and config loaders
```

## Methodology notes

The paper's methodological contribution — **within-modality damage percentile binning** for cross-modal robustness evaluation — is implemented across these modules:

- **`src/analysis/quintile.py`** — within-modality quintile assignment (`assign_quintile` wraps `pd.qcut`).
- **`src/analysis/bootstrap.py`** — hierarchical bootstrap with seed-then-pair resampling at $n_{\text{boot}} = 10{,}000$ iterations, preserving paired comparisons (`hier_boot_mean`, `hier_boot_diff`).
- **`src/analysis/damage_metrics.py`** — the four input-side damage rulers (SSIM, PSNR, normalized token edit distance, BLEU).
- **`src/analysis/significance.py`** — paired Wilcoxon signed-rank test, Cohen's $d$ (paired and binary-independent variants).
- **`src/analysis/reliability.py`** — cross-seed Spearman correlation; both per-cell (`per_cell_spearman`) and cell-mean (`per_cell_mean_spearman`) variants for retrieval data with non-overlapping pair_id semantics across seeds.
- **`src/spokes/match_retention.py`** — the match-retention spoke (per-pair retention margin against a $K=64$ disjoint mismatch caption pool).
- **`src/spokes/retrieval.py`** — Recall@$k$ retrieval against a 1,000-candidate pool, with per-pair indicator logging.

Per-spoke pair counts: image fidelity, text fidelity, and match retention each use $n = 300$ pairs per (seed, condition) cell; retrieval uses $n = 1{,}000$. Three seeds (0, 1, 2). Five severity levels (1 through 5). Six corruption types (image: Gaussian noise, Gaussian blur, cutout; text: mask, shuffle, replace).

Headline figures and tables are produced by `src/plots/` (one module per figure) and the `tables` module within. The master rebuild script is `scripts/build_paper_artifacts.py`. Configurations are centralised in `configs/{experiment,models,severity}.yaml`; editing these yaml files rescales runs without touching code.

## Tests

Unit tests cover corruption operators, damage rulers, bootstrap CIs, quintile binning, and config loaders. From the repository root:

```bash
pytest tests/ -v
```

Runs in under five seconds.

## License

Code: MIT License. Documentation, figures, and tables: CC BY 4.0. MS-COCO 2017 captions are used under their original CC BY 4.0 license; the five evaluated model checkpoints (CLIP, OpenCLIP, SigLIP 2, PE-Core) are used per their respective public licenses.
