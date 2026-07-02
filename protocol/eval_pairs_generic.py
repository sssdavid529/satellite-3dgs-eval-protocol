#!/usr/bin/env python3
"""Generic region-metric evaluator (M3/S2/S5/M2a shared protocol).

Same unmasked regional protocol as phase3 eval, but accepts arbitrary
label=expname pairs. Weak-proxy and error-top20 regions are ALWAYS anchored
on the canonical seed-1337 baseline (phase1_eogsplus_rawrpc_*) so that all
seeds/configs are evaluated on identical regions and stay comparable.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EVAL_ROOT", "/root/autodl-tmp/eogs"))
import weak_proxy_lib as base  # noqa: E402

REPO = base.REPO


def rdsm_path(expname: str, scene: str) -> Path:
    if expname.startswith("/") and expname.endswith(".tif"):
        return Path(expname)  # direct absolute path (e.g. M2a/M2b registered DSMs)
    iters = "15000" if expname.endswith("_15000") else "5000"
    return REPO / "output" / expname / f"test_opNone/ours_{iters}/rdsm" / f"{scene}_rdsm.tif"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="label=expname")
    ap.add_argument("--anchor", default=None,
                    help="expname anchoring weak-proxy/error-top20 (default: canonical phase1 baseline)")
    ap.add_argument("--out-dir", default=str(ROOT / "eval_tools/out"))
    args = ap.parse_args()

    scene = args.scene
    short = base.scene_short(scene)
    anchor_exp = args.anchor or f"phase1_eogsplus_rawrpc_{short}_pan_3PAN_5000"
    data_root = REPO / "data"

    truth, truth_profile = base.read_1(data_root / "Truth" / f"{scene}_DSM.tif")
    cls, _ = base.read_1(data_root / "Truth" / f"{scene}_CLS.tif")
    anchor, _ = base.read_1(rdsm_path(anchor_exp, scene))

    out_dir = Path(args.out_dir) / scene
    anchor_a, _truth_a, _cls_a = base.align(anchor, truth, cls)
    proxy, proxy_meta = base.compute_weak_proxy(scene, anchor_a, truth_profile, out_dir)
    weak_top20 = proxy["weak_proxy_top20"].astype("float32")

    labels, exps = [], []
    for spec in args.methods:
        label, exp = spec.split("=", 1)
        labels.append(label)
        exps.append(exp)
    preds = []
    for exp in exps:
        p = rdsm_path(exp, scene)
        if not p.exists():
            raise FileNotFoundError(p)
        arr, _ = base.read_1(p)
        preds.append(arr)

    arrays = base.align(anchor, truth, cls, weak_top20, *preds)
    anchor2, truth2, cls2, weak2 = arrays[0], arrays[1], arrays[2].astype("int32"), arrays[3].astype(bool)
    pred_arrays = arrays[4:]

    common = np.isfinite(anchor2) & np.isfinite(truth2)
    for a in pred_arrays:
        common &= np.isfinite(a)
    anchor_err = np.abs(anchor2 - truth2)
    thr = float(np.nanpercentile(anchor_err[common], 80))
    err_top20 = common & (anchor_err >= thr)

    def region_metrics_ext(pred):
        row = base.region_metrics(pred, truth2, cls2, common, weak2, err_top20)
        diff = np.abs(pred - truth2).astype("float32")
        regions = {
            "overall": np.ones_like(common, dtype=bool),
            "water_cls9": cls2 == 9,
            "building_cls6": cls2 == 6,
            "ground_cls2": cls2 == 2,
            "tree_cls5": cls2 == 5,
            "weak_proxy_top20": weak2,
            "phase0_error_top20_eval_only": err_top20,
        }
        for name, mask0 in regions.items():
            mask = common & mask0
            vals = diff[mask]
            row[f"{name}_rmse"] = float(np.sqrt(np.nanmean(vals ** 2))) if vals.size else None
            row[f"{name}_comp1m"] = float(np.nanmean(vals < 1.0)) if vals.size else None
        return row

    rows = []
    for label, arr in zip(labels, pred_arrays):
        rows.append({
            "scene": scene,
            "method": label,
            **region_metrics_ext(arr),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{scene}_metrics.csv"
    base.write_csv(csv_path, rows)
    (out_dir / f"{scene}_meta.json").write_text(json.dumps({
        "anchor": anchor_exp,
        "error_top20_p80_threshold": thr,
        "common_valid_pixels": int(common.sum()),
        "weak_proxy": {k: v for k, v in proxy_meta.items() if k != "per_image_reports"},
        "methods": dict(zip(labels, exps)),
    }, indent=2), encoding="utf-8")

    brief = {r["method"]: {k.replace("_mae", ""): round(r[k], 4)
                           for k in r if k.endswith("_mae") and r[k] is not None}
             for r in rows}
    print(json.dumps(brief, indent=2))
    print("WROTE", csv_path)


if __name__ == "__main__":
    main()
