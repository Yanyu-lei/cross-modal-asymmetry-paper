#!/bin/bash
# Rebuild the outputs in the current Python environment. Run the tests, then build both
# manifests, run every experiment, build the paper and check the built files against each
# other.
# GPU smoke tests run when CUDA is available.
#
#   scripts/reproduce_all.sh          # uses `python`; set PYTHON=... to choose another
#
# The runs need a GPU and take hours. Each step stops the script if it fails. Running it
# again after an interruption skips the jobs and runs that already finished.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" -m pytest -q tests
"$PY" -m scripts.build_manifests --dataset coco
"$PY" -m scripts.build_manifests --dataset flickr30k
"$PY" -m scripts.run_experiments
"$PY" -m scripts.make_figures_tables_numbers --require-all
"$PY" -m pytest -q tests/test_paper_outputs.py -k built
