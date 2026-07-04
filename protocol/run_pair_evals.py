#!/usr/bin/env python3
"""Pair-grid eval: EOGS++ (6 seeds) vs original EOGS (6 seeds) per scene.

Evaluates the published pair on the intersection grid of the two GS families
only (no checkpoint-limited Sat-NeRF coverage), so the two-sample comparison
in Sec. VI is grid-clean. Writes eval_tools/out_pair/<S>/<S>_metrics.csv.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/eogs")
V1OUT = Path("/root/autodl-tmp/eogs_v1/output")
EVAL = ROOT / "eval_tools/eval_pairs_generic.py"
OUTDIR = ROOT / "eval_tools/out_pair"
SCENES = ["JAX_260", "JAX_214", "JAX_004", "JAX_068", "IARPA_001", "IARPA_002", "IARPA_003"]
SEEDS = [2024, 3407, 5150, 6001, 7777]


def v1path(scene, seed=None):
    d = V1OUT / (f"m2a_eogsv1_{scene}" if seed is None else f"m2a_eogsv1_{scene}_s{seed}")
    p = d / f"{scene}_rdsm.tif"
    if not p.exists():
        cands = list(d.glob("**/*rdsm.tif")) if d.exists() else []
        p = cands[0] if cands else p
    return p


def main():
    for scene in SCENES:
        sh = scene.replace("_", "")
        methods = [f"eogspp_s1337=phase1_eogsplus_rawrpc_{sh}_pan_3PAN_5000"]
        methods += [f"eogspp_s{k}=q1_baseline_s{k}_{sh}_pan_3PAN_5000" for k in SEEDS]
        p0 = v1path(scene)
        if p0.exists():
            methods.append(f"eogsv1_default={p0}")
        for k in SEEDS:
            p = v1path(scene, k)
            if p.exists():
                methods.append(f"eogsv1_s{k}={p}")
        print(f"===== PAIR EVAL {scene}: {len(methods)} methods =====", flush=True)
        cmd = [sys.executable, str(EVAL), "--scene", scene,
               "--out-dir", str(OUTDIR), "--methods"] + methods
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("\n".join(r.stdout.strip().splitlines()[-2:]), flush=True)
        if r.returncode != 0:
            print(f"[ERROR] {scene}:\n{r.stderr[-1500:]}", flush=True)
            sys.exit(1)
    print("===== PAIR EVALS ALL DONE =====", flush=True)


if __name__ == "__main__":
    main()
