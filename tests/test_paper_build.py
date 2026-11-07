"""The paper build reports what it left out, and fails on it when asked to."""
from __future__ import annotations

import sys

import pytest

import scripts.make_figures_tables_numbers as build
from src.load_results import SKIPPED, note_skipped


def test_require_all_fails_when_a_step_left_something_out(monkeypatch, capsys):
    monkeypatch.setattr(build, "STEPS", {"incomplete": lambda: note_skipped("a run is missing")})
    try:
        monkeypatch.setattr(sys, "argv", ["build"])
        build.main()
        assert "a run is missing" in capsys.readouterr().out
        monkeypatch.setattr(sys, "argv", ["build", "--require-all"])
        with pytest.raises(SystemExit, match="incomplete"):
            build.main()
    finally:
        SKIPPED.clear()
