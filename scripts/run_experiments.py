"""Run the configured experiments in separate processes with one model job per GPU.

    python -m scripts.run_experiments
    python -m scripts.run_experiments coco_2000_pairs_retrieval
    python -m scripts.run_experiments --list

Damage jobs use the CPU. Model jobs require CUDA unless --allow-cpu is given.
Completed parts and final outputs are reused without checking their provenance.
If saving the final outputs is interrupted, move the partial outputs aside before resuming.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import file_paths, settings  # noqa: E402
from src.evaluate_model import job_parts, run_job  # noqa: E402


def plan(name: str, run: dict) -> list[dict]:
    """List a run's jobs in their output order."""
    if run["kind"] == "text_depth_diagnostic":
        return [{"run": name, "kind": run["kind"], "model": m} for m in settings.MODELS]
    common = {"run": name, "dataset": run["dataset"], "pairs": run["pairs"],
              "image_corruptions": tuple(run.get("image_corruptions", settings.IMAGE_CORRUPTIONS)),
              "text_corruptions": tuple(run.get("text_corruptions", settings.TEXT_CORRUPTIONS))}
    models = [{**common, "kind": run["kind"], "model": m, "seed": s} for s in settings.SEEDS for m in settings.MODELS]
    damage = [{**common, "kind": "damage", "run_kind": run["kind"], "seed": s} for s in settings.SEEDS]
    return models + damage


def outputs(name: str, run: dict) -> dict[Path, list[Path]]:
    """Map each output file to the part files joined into it, in order."""
    jobs = plan(name, run)
    if run["kind"] == "text_depth_diagnostic":
        return {file_paths.TEXT_DEPTH_DIAGNOSTIC: [job_parts(j)[0] for j in jobs]}
    model_jobs = [j for j in jobs if j["kind"] != "damage"]
    joined = {file_paths.damage_output(name): [job_parts(j)[0] for j in jobs if j["kind"] == "damage"]}
    if run["kind"] == "retrieval":
        joined[file_paths.run_output(name)] = [job_parts(j)[0] for j in model_jobs]
        return joined
    for seed in settings.SEEDS:
        joined[file_paths.run_output(name, seed)] = [job_parts(j)[0] for j in model_jobs if j["seed"] == seed]
    # Group the ablation rows by model, then seed.
    by_model = sorted(model_jobs, key=lambda j: settings.MODELS.index(j["model"]))
    joined[file_paths.text_pooling_ablation_output(run["dataset"])] = [job_parts(j)[1] for j in by_model]
    return joined


def join(parts: list[Path], out: Path) -> Path:
    """Join the CSV parts with identical headers into a temporary file and return its path."""
    tmp = out.with_suffix(".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as dst:
        for i, part in enumerate(parts):
            with open(part) as src:
                header = src.readline()
                if i == 0:
                    first = header
                    dst.write(header)
                elif header != first:
                    raise ValueError(f"{part} has a different header from {parts[0]}, so they cannot be joined.")
                for line in src:
                    dst.write(line)
    return tmp


def _device_token(gpu: int, visible: str | None) -> str:
    """Map a visible GPU index to the scheduler's device ordinal or UUID."""
    if not visible:
        return str(gpu)
    return [t.strip() for t in visible.split(",") if t.strip()][gpu]


def _start(job: dict, gpu: int | None) -> multiprocessing.Process:
    """Spawn a job with its assigned GPU visible, then restore the parent's device mask."""
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _device_token(gpu, previous)
        process = multiprocessing.get_context("spawn").Process(target=run_job, args=(job,), daemon=False)
        process.start()
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    return process


def run_all(jobs: list[dict], allow_cpu: bool = False) -> list[dict]:
    """Schedule the model and damage jobs on GPU and CPU slots and return the failed jobs."""
    import torch

    gpu_slots: list[int | None] = list(range(torch.cuda.device_count())) or [None]
    if gpu_slots == [None] and not allow_cpu and any(j["kind"] != "damage" for j in jobs):
        sys.exit("PyTorch finds no GPU, so the model jobs would run on the CPU. Check nvidia-smi and that the "
                 "installed PyTorch build supports this driver, or pass --allow-cpu to run on the CPU anyway.")
    # Respect the scheduler's CPU allocation.
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    cpu_slots = max(1, cores // 4)
    waiting = [j for j in jobs if not all(p.exists() for p in job_parts(j))]
    print(f"{len(jobs) - len(waiting)} of {len(jobs)} jobs already done; {len(waiting)} to run "
          f"on {len([g for g in gpu_slots if g is not None]) or 'no'} GPU(s).", flush=True)
    running: dict[multiprocessing.Process, tuple[dict, int | None]] = {}
    failed, done, t0 = [], 0, time.time()
    while waiting or running:
        for job in list(waiting):
            on_cpu = job["kind"] == "damage"
            busy_cpu = sum(1 for j, _ in running.values() if j["kind"] == "damage")
            free_gpus = [g for g in gpu_slots if g not in {gpu for j, gpu in running.values() if j["kind"] != "damage"}]
            if on_cpu and busy_cpu < cpu_slots:
                running[_start(job, None)] = (job, None)
            elif not on_cpu and free_gpus:
                running[_start(job, free_gpus[0])] = (job, free_gpus[0])
            else:
                continue
            waiting.remove(job)
        for process in [p for p in running if not p.is_alive()]:
            process.join()
            job, _ = running.pop(process)
            done += 1
            label = f"{job['run']} {job.get('model', 'damage')} seed {job.get('seed', '-')}"
            if process.exitcode != 0:
                failed.append(job)
                print(f"[{time.time() - t0:7.0f}s] FAILED {label} (exit {process.exitcode})", flush=True)
            else:
                print(f"[{time.time() - t0:7.0f}s] done {label} ({done} finished)", flush=True)
        time.sleep(2)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experiments behind the paper.")
    parser.add_argument("runs", nargs="*", help="runs to do (default: every run in configs/paper_runs.yaml)")
    parser.add_argument("--list", action="store_true", help="list the runs and their jobs, then stop")
    parser.add_argument("--allow-cpu", action="store_true", help="run the model jobs on the CPU when there is no GPU")
    args = parser.parse_args()

    names = list(dict.fromkeys(args.runs or settings.RUNS))
    unknown = [n for n in names if n not in settings.RUNS]
    if unknown:
        sys.exit(f"unknown run(s): {', '.join(unknown)}. See --list.")
    if args.list:
        for name in names:
            run = settings.RUNS[name]
            state = "done" if all(p.exists() for p in outputs(name, run)) else "to do"
            print(f"{name:42s} {len(plan(name, run)):3d} jobs  {state:5s}  {' '.join(run['what'].split())}")
        return

    todo = []
    for name in names:
        existing = [p for p in outputs(name, settings.RUNS[name]) if p.exists()]
        if len(existing) == len(outputs(name, settings.RUNS[name])):
            print(f"{name}: its output files already exist, so it is skipped.")
        elif existing:
            sys.exit(f"{name} has some of its output files but not all. Move these aside first:\n  "
                     + "\n  ".join(str(p) for p in existing))
        else:
            todo.append(name)

    failed = run_all([job for name in todo for job in plan(name, settings.RUNS[name])], args.allow_cpu)
    failed_runs = {job["run"] for job in failed}
    for name in todo:
        if name in failed_runs:
            continue
        # A failed join must not leave a partial set of final files.
        joined = {out: join(parts, out) for out, parts in outputs(name, settings.RUNS[name]).items()}
        for out, tmp in joined.items():
            os.replace(tmp, out)
        print(f"{name}: joined into its output files.")
    if failed:
        sys.exit(f"{len(failed)} job(s) failed in {', '.join(sorted(failed_runs))}. Rerun the same command to "
                 "retry them; finished jobs are kept.")


if __name__ == "__main__":
    main()
