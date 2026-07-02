#!/usr/bin/env python3
"""S3: null control for the evaluation-only error-top20 region.

For each scene and each anchor seed A: define the error-top20 region from
baseline(seed A), then measure baseline(seed B != A) MAE inside that region.
The drop (B minus A inside A's worst region) quantifies pure regression-to-
the-mean under retraining noise - the null against which any method's
"improvement in high-error regions" must be compared.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EVAL_ROOT", "/root/autodl-tmp/eogs"))
import weak_proxy_lib as base  # noqa: E402

REPO = base.REPO
OUT = ROOT / "eval_tools/out"
SCENES = ["JAX_260", "JAX_214", "JAX_004", "JAX_068", "IARPA_001", "IARPA_002", "IARPA_003"]
ALL_SEEDS = [1337, 2024, 3407, 5150, 6001, 7777]


def bpath(scene, seed):
    sh = scene.replace("_", "")
    exp = f"phase1_eogsplus_rawrpc_{sh}_pan_3PAN_5000" if seed == 1337 \
        else f"q1_baseline_s{seed}_{sh}_pan_3PAN_5000"
    return REPO / "output" / exp / "test_opNone/ours_5000/rdsm" / f"{scene}_rdsm.tif"


def opath(scene, seed):
    sh = scene.replace("_", "")
    if seed == 1337:
        exp = "phase1b_structure_protected_JAX260_pan_3PAN_5000" if scene == "JAX_260" \
            else f"phase3_phase1b_default_{sh}_pan_3PAN_5000"
    else:
        exp = f"q1_ours_s{seed}_{sh}_pan_3PAN_5000"
    return REPO / "output" / exp / "test_opNone/ours_5000/rdsm" / f"{scene}_rdsm.tif"


def main():
    results = []
    for scene in SCENES:
        truth, _ = base.read_1(REPO / "data/Truth" / f"{scene}_DSM.tif")
        seeds = [s for s in ALL_SEEDS if bpath(scene, s).exists()]
        if len(seeds) < 2:
            print(f"[skip] {scene}: {len(seeds)} baseline seeds")
            continue
        dsms = {s: base.read_1(bpath(scene, s))[0] for s in seeds}
        ours = {s: base.read_1(opath(scene, s))[0] for s in seeds if opath(scene, s).exists()}
        arrays = base.align(truth, *dsms.values(), *ours.values())
        truth_a = arrays[0]
        dsms = dict(zip(dsms.keys(), arrays[1:1 + len(dsms)]))
        ours = dict(zip(ours.keys(), arrays[1 + len(dsms):]))
        valid = np.isfinite(truth_a)
        for a in list(dsms.values()) + list(ours.values()):
            valid &= np.isfinite(a)

        null_drops, method_drops = [], []
        for A in seeds:
            errA = np.abs(dsms[A] - truth_a)
            thr = float(np.nanpercentile(errA[valid], 80))
            region = valid & (errA >= thr)
            maeA = float(np.nanmean(errA[region]))
            for B in seeds:
                if B == A:
                    continue
                errB = np.abs(dsms[B] - truth_a)
                null_drops.append(float(np.nanmean(errB[region])) - maeA)
            if A in ours:
                errO = np.abs(ours[A] - truth_a)
                method_drops.append(float(np.nanmean(errO[region])) - maeA)

        row = {
            "scene": scene,
            "n_seeds": len(seeds),
            "null_mean_drop": round(float(np.mean(null_drops)), 4),
            "null_std": round(float(np.std(null_drops)), 4),
            "null_min_max": [round(float(np.min(null_drops)), 4), round(float(np.max(null_drops)), 4)],
            "method_same_seed_drop_mean": round(float(np.mean(method_drops)), 4) if method_drops else None,
            "method_drops": [round(d, 4) for d in method_drops],
        }
        results.append(row)
        print(scene, "NULL(seed-swap) drop = %.4f±%.4f | METHOD(same-seed) drop = %s" %
              (row["null_mean_drop"], row["null_std"], row["method_same_seed_drop_mean"]))

    (OUT / "s3_null_control.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("WROTE", OUT / "s3_null_control.json")


if __name__ == "__main__":
    main()
