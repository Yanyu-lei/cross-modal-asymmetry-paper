"""The runner's GPU choice, its refusal to run model jobs without a GPU, and its joining of
part files."""
from __future__ import annotations

import os

import pytest
import torch

import scripts.run_experiments as runner
from scripts.run_experiments import _device_token, join


def test_each_job_gets_the_gpu_the_scheduler_gave():
    assert _device_token(1, None) == "1"
    assert _device_token(0, "2,3") == "2"
    assert _device_token(1, "2, 3") == "3"
    assert _device_token(1, "GPU-aaaa,GPU-bbbb") == "GPU-bbbb"


def test_join_keeps_one_header_and_refuses_different_ones(tmp_path):
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    a.write_text("model,value\nx,1\n")
    b.write_text("model,value\ny,2\n")
    c.write_text("value,model\n3,z\n")
    out = tmp_path / "out.csv"
    assert join([a, b], out).read_text() == "model,value\nx,1\ny,2\n"
    assert not out.exists()  # the caller puts the joined file in place
    with pytest.raises(ValueError, match="different header"):
        join([a, c], out)


def test_starting_a_job_leaves_the_parent_environment_as_it_was(monkeypatch):
    seen = []

    class FakeProcess:
        def __init__(self, target, args, daemon):
            pass

        def start(self):
            seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr(runner.multiprocessing, "get_context", lambda method: FakeContext)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    runner._start({}, 1)
    assert seen == ["3"] and os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    runner._start({}, 0)
    assert seen == ["3", "0"] and "CUDA_VISIBLE_DEVICES" not in os.environ


def test_model_jobs_do_not_fall_back_to_the_cpu_unless_asked(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(SystemExit, match="no GPU"):
        runner.run_all([{"run": "r", "kind": "retrieval", "model": "m", "seed": 0}])
