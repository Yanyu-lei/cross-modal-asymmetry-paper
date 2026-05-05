# Cheap smoke: every src/ module must import without error.
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "src",
    "src.runner",
    "src.schema",
    "src.analysis",
    "src.analysis.bootstrap",
    "src.analysis.baseline",
    "src.analysis.calibration_test",
    "src.analysis.corruption_breakdown",
    "src.analysis.damage_metrics",
    "src.analysis.quintile",
    "src.analysis.reliability",
    "src.analysis.retrieval_eval",
    "src.analysis.significance",
    "src.corruptions",
    "src.corruptions.image",
    "src.corruptions.severity",
    "src.corruptions.text",
    "src.data",
    "src.data.coco",
    "src.models",
    "src.models._hooks",
    "src.models._text_pooling",
    "src.models.base",
    "src.models.openclip_backend",
    "src.models.pecore_backend",
    "src.models.registry",
    "src.plots._data",
    "src.plots._style",
    "src.plots.f1_retrieval_asymmetry",
    "src.plots.f2_calibration_curves",
    "src.plots.f3_joint_heatmaps",
    "src.plots.f4_layer_fidelity",
    "src.plots.f5_two_component_decomposition",
    "src.plots.f6_corruption_breakdown",
    "src.plots.tables",
    "src.spokes",
    "src.spokes.image_fidelity",
    "src.spokes.match_retention",
    "src.spokes.retrieval",
    "src.spokes.text_fidelity",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)
