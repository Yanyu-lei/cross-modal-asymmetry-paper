"""Every module in src/ imports without error."""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "src",
    "src.file_paths",
    "src.load_data",
    "src.load_models",
    "src.clip_and_siglip_models",
    "src.pecore_model",
    "src.corruptions",
    "src.fidelity_retention_retrieval_metrics",
    "src.evaluate_model",
    "src.input_damage_metrics",
    "src.text_pooling_ablation_and_depth_diagnostic",
    "src.settings",
    "src.load_results",
    "src.bootstrap_and_quintiles",
    "src.paper_figures",
    "src.paper_tables",
    "src.paper_numbers",
    "src.paper_appendix_tables",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)
