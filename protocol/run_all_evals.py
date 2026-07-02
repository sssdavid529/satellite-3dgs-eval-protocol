#!/usr/bin/env python3
"""Master evaluation: discover every relevant run for each of the 7 AOIs and
evaluate them all under the unified protocol (eval_pairs_generic), then run
the S6 diagnostics. Designed to run after the training pipeline finishes.

Discovered method families per scene:
  baseline_s1337   phase1_eogsplus_rawrpc_<S>_pan_3PAN_5000
  ours_s1337       phase3_phase1b_default_<S> | phase1b_structure_protected_JAX260
  baseline_s<K>    q1_baseline_s<K>_<S>_pan_3PAN_5000   (K in 2024,3407,5150,6001,7777)
  ours_s<K>        q1_ours_s<K>_<S>_pan_3PAN_5000
  nostruct_s1337   s2_nostruct_<S>_pan_3PAN_5000
  b15k_s<K>/o15k_s<K>  q1_*_s<K>_<S>_pan_3PAN_15000     (JAX_260 only)
  eogsv1           /root/autodl-tmp/eogs_v1/output/m2a_eogsv1_<S>/... via registered rdsm
  satnerf_ds / satnerf   /root/autodl-tmp/satnerf/m2b_out/registered/<S>_Sat-NeRF*_/<S>_rdsm.tif
"""
import subprocess
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EVAL_ROOT", "/root/autodl-tmp/eogs"))
REPO = ROOT / "EOGS2"
EVAL = ROOT / "eval_tools/eval_pairs_generic.py"
SCENES = ["JAX_260", "JAX_214", "JAX_004", "JAX_068", "IARPA_001", "IARPA_002", "IARPA_003"]
SEEDS = [2024, 3407, 5150, 6001, 7777]


def short(s: str) -> str:
    return s.replace("_", "")


def exp_rdsm(exp: str, scene: str, iters: str = "5000") -> Path:
    return REPO / "output" / exp / f"test_opNone/ours_{iters}/rdsm" / f"{scene}_rdsm.tif"


def main() -> None:
    for scene in SCENES:
        sh = short(scene)
        methods = []

        b0 = f"phase1_eogsplus_rawrpc_{sh}_pan_3PAN_5000"
        if exp_rdsm(b0, scene).exists():
            methods.append(f"baseline_s1337={b0}")
        if scene == "JAX_260":
            o0 = "phase1b_structure_protected_JAX260_pan_3PAN_5000"
        else:
            o0 = f"phase3_phase1b_default_{sh}_pan_3PAN_5000"
        if exp_rdsm(o0, scene).exists():
            methods.append(f"ours_s1337={o0}")

        for k in SEEDS:
            for cfg in ("baseline", "ours"):
                e = f"q1_{cfg}_s{k}_{sh}_pan_3PAN_5000"
                if exp_rdsm(e, scene).exists():
                    methods.append(f"{cfg}_s{k}={e}")

        e = f"s2_nostruct_{sh}_pan_3PAN_5000"
        if exp_rdsm(e, scene).exists():
            methods.append(f"nostruct_s1337={e}")

        if scene == "JAX_260":
            for k in (1337, 2024):
                for cfg, lab in (("baseline", "b15k"), ("ours", "o15k")):
                    e = f"q1_{cfg}_s{k}_{sh}_pan_3PAN_15000"
                    if exp_rdsm(e, scene, "15000").exists():
                        methods.append(f"{lab}_s{k}={e}")

        # cross-method group is evaluated SEPARATELY: their registration edge NaNs
        # (up to 27% for released Sat-NeRF ckpts) must not shrink the EOGS-family
        # common-valid region used for T1/T2/T3/T6.
        cross = []
        if methods and exp_rdsm(b0, scene).exists():
            cross.append(f"baseline_s1337={b0}")
        m2a = Path(f"/root/autodl-tmp/eogs_v1/output/m2a_eogsv1_{scene}/{scene}_rdsm.tif")
        if not m2a.exists():
            cands = list(Path(f"/root/autodl-tmp/eogs_v1/output/m2a_eogsv1_{scene}").glob("**/*rdsm.tif")) \
                if Path(f"/root/autodl-tmp/eogs_v1/output/m2a_eogsv1_{scene}").exists() else []
            m2a = cands[0] if cands else m2a
        if m2a.exists():
            cross.append(f"eogsv1={m2a}")
        for k in (2024, 3407):
            p = Path(f"/root/autodl-tmp/eogs_v1/output/m2a_eogsv1_{scene}_s{k}/{scene}_rdsm.tif")
            if p.exists():
                cross.append(f"eogsv1_s{k}={p}")
        for tag, lab in ((f"{scene}_Sat-NeRF_DS", "satnerf_ds"), (f"{scene}_Sat-NeRF", "satnerf")):
            p = Path(f"/root/autodl-tmp/satnerf/m2b_out/registered/{tag}/{scene}_rdsm.tif")
            if p.exists():
                cross.append(f"{lab}={p}")

        if len(methods) < 2:
            print(f"[skip] {scene}: only {len(methods)} methods found")
            continue
        print(f"===== EVAL {scene}: {len(methods)} eogs-family methods =====", flush=True)
        cmd = [sys.executable, str(EVAL), "--scene", scene, "--methods"] + methods
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = "\n".join(r.stdout.strip().splitlines()[-3:])
        print(tail, flush=True)
        if r.returncode != 0:
            print(f"[ERROR] {scene}:\n{r.stderr[-1500:]}", flush=True)

        if len(cross) >= 2:
            print(f"===== EVAL {scene}: {len(cross)} cross-method =====", flush=True)
            cmd = [sys.executable, str(EVAL), "--scene", scene,
                   "--out-dir", str(ROOT / "eval_tools/out_cross"), "--methods"] + cross
            r = subprocess.run(cmd, capture_output=True, text=True)
            print("\n".join(r.stdout.strip().splitlines()[-2:]), flush=True)
            if r.returncode != 0:
                print(f"[CROSS ERROR] {scene}:\n{r.stderr[-1000:]}", flush=True)

    print("===== S6 DIAGNOSTICS =====", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "eval_tools/s6_phase0_stats.py")],
                       capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print("[S6 ERROR]", r.stderr[-800:], flush=True)
    print("===== ALL EVALS DONE =====", flush=True)


if __name__ == "__main__":
    main()
