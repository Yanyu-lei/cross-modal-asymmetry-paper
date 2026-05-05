"""One entrypoint to (re)build every paper figure and table."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    t0 = time.time()
    print(f"\n[{time.time()-t0:6.1f}s] F1 (headline)")
    from src.plots import f1_retrieval_asymmetry; f1_retrieval_asymmetry.build()
    print(f"\n[{time.time()-t0:6.1f}s] T1, T2, T3, T4, T5, T6, T7")
    from src.plots import tables; tables.main()
    print(f"\n[{time.time()-t0:6.1f}s] F2 (calibration curves)")
    from src.plots import f2_calibration_curves; f2_calibration_curves.build()
    print(f"\n[{time.time()-t0:6.1f}s] F3 (joint heatmaps, appendix)")
    from src.plots import f3_joint_heatmaps; f3_joint_heatmaps.build()
    print(f"\n[{time.time()-t0:6.1f}s] F4 (layer-wise fidelity, appendix)")
    from src.plots import f4_layer_fidelity; f4_layer_fidelity.build()
    print(f"\n[{time.time()-t0:6.1f}s] F5 (two-component decomposition, appendix)")
    from src.plots import f5_two_component_decomposition; f5_two_component_decomposition.build()
    print(f"\n[{time.time()-t0:6.1f}s] F6 (per-corruption breakdown, appendix)")
    from src.plots import f6_corruption_breakdown; f6_corruption_breakdown.build()
    print(f"\n[{time.time()-t0:6.1f}s] DONE")


if __name__ == "__main__":
    main()
