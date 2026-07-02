#!/usr/bin/env python3
"""S6: multi-temporal complementarity statistics across all 7 AOIs.

For each scene, within two weak-region definitions:
  (a) evaluation-only baseline-error top20  (diagnostic, uses truth)
  (b) truth-free weak-proxy top20
report the fraction of pixels with >=1 good view (partial-good), plus
mean error in weak vs strong regions. Consumes weak_proxy_rasters produced
by eval_pairs_generic.py (which anchors on the canonical baseline).
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


def main() -> None:
    rows = []
    for scene in SCENES:
        short = base.scene_short(scene)
        rast = OUT / scene / "weak_proxy_rasters"
        need = {
            "good": rast / f"{scene}_good_view_count.tif",
            "valid": rast / f"{scene}_valid_view_count.tif",
            "score": rast / f"{scene}_weak_proxy_score.tif",
            "top20": rast / f"{scene}_weak_proxy_top20.tif",
        }
        if not all(p.exists() for p in need.values()):
            print(f"[skip] {scene}: proxy rasters missing (run eval_pairs_generic first)")
            continue
        good, _ = base.read_1(need["good"])
        validc, _ = base.read_1(need["valid"])
        score, _ = base.read_1(need["score"])
        top20, _ = base.read_1(need["top20"])
        baseline, _ = base.read_1(
            REPO / "output" / f"phase1_eogsplus_rawrpc_{short}_pan_3PAN_5000"
            / "test_opNone/ours_5000/rdsm" / f"{scene}_rdsm.tif")
        truth, _ = base.read_1(REPO / "data/Truth" / f"{scene}_DSM.tif")

        good, validc, score, top20, baseline, truth = base.align(good, validc, score, top20, baseline, truth)
        valid = np.isfinite(baseline) & np.isfinite(truth) & (validc > 0)
        err = np.abs(baseline - truth)

        thr = float(np.nanpercentile(err[valid], 80))
        err_top20 = valid & (err >= thr)
        weak_top20 = valid & (top20 > 0)
        strong = valid & (score <= 0.30) & (good >= 4) & (err < thr)

        def stats(mask: np.ndarray) -> dict:
            n = int(mask.sum())
            if n == 0:
                return {"pixels": 0}
            return {
                "pixels": n,
                "partial_good_ratio": float((good[mask] >= 1).mean()),
                "all_weak_ratio": float((good[mask] == 0).mean()),
                "mean_good_views": float(good[mask].mean()),
                "mean_error": float(np.nanmean(err[mask])),
            }

        row = {
            "scene": scene,
            "error_top20": stats(err_top20),
            "weak_proxy_top20": stats(weak_top20),
            "strong_reference": stats(strong),
        }
        rows.append(row)
        print(scene,
              "err20 partial-good %.4f" % row["error_top20"].get("partial_good_ratio", float("nan")),
              "weak20 partial-good %.4f" % row["weak_proxy_top20"].get("partial_good_ratio", float("nan")),
              "err(weak20)=%.3f err(strong)=%.3f" % (
                  row["weak_proxy_top20"].get("mean_error", float("nan")),
                  row["strong_reference"].get("mean_error", float("nan"))))

    out_path = OUT / "s6_phase0_stats_7scenes.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
