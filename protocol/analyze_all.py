#!/usr/bin/env python3
"""Aggregate all per-scene metrics.csv into paper tables T1/T2/T3/T4/T6 (markdown + JSON)."""
import csv
import json
import re
import statistics as st
from pathlib import Path

OUT = Path("/root/autodl-tmp/eogs/eval_tools/out")
SCENES = ["JAX_260", "JAX_214", "JAX_004", "JAX_068", "IARPA_001", "IARPA_002", "IARPA_003"]
METRICS = ["overall_mae", "water_cls9_mae", "building_cls6_mae", "ground_cls2_mae",
           "tree_cls5_mae", "weak_proxy_top20_mae", "phase0_error_top20_eval_only_mae"]
SHORT = {"overall_mae": "overall", "water_cls9_mae": "water", "building_cls6_mae": "building",
         "ground_cls2_mae": "ground", "tree_cls5_mae": "tree",
         "weak_proxy_top20_mae": "weak20", "phase0_error_top20_eval_only_mae": "err20"}


def load(scene):
    p = OUT / scene / f"{scene}_metrics.csv"
    if not p.exists():
        return {}
    rows = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["method"]] = {m: (float(r[m]) if r.get(m) not in (None, "", "None") else None)
                                 for m in METRICS if m in r}
    return rows


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals), min(vals), max(vals)


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    data = {s: load(s) for s in SCENES}
    lines = []
    summary = {}

    # T1: baseline seed variance per scene
    lines.append("## T1 Baseline seed variance (EOGS++, per scene)\n")
    lines.append("| scene | n | " + " | ".join(SHORT[m] + " mean±std (range)" for m in METRICS) + " |")
    lines.append("|" + "---|" * (len(METRICS) + 2))
    t1 = {}
    for s in SCENES:
        fam = {k: v for k, v in data[s].items() if re.fullmatch(r"baseline_s\d+", k)}
        if not fam:
            continue
        row, cells = {}, []
        for m in METRICS:
            r = ms([v[m] for v in fam.values()])
            row[SHORT[m]] = r
            cells.append("—" if r is None else f"{r[0]:.3f}±{r[1]:.3f} ({r[4]-r[3]:.3f})")
        t1[s] = {SHORT[m]: (None if row[SHORT[m]] is None else row[SHORT[m]][:2]) for m in METRICS}
        lines.append(f"| {s} | {len(fam)} | " + " | ".join(cells) + " |")
    summary["T1_baseline_variance"] = t1

    # T2: paired deltas ours - baseline per seed
    lines.append("\n## T2 Paired deltas (ours − baseline, same seed)\n")
    lines.append("| scene | metric | per-seed Δ | mean Δ ± std | improved/total |")
    lines.append("|---|---|---|---|---|")
    t2 = {}
    for s in SCENES:
        seeds = sorted({int(k.split("_s")[1]) for k in data[s] if re.fullmatch(r"baseline_s\d+", k)})
        pairs = [(k, f"ours_s{k}") for k in seeds
                 if f"baseline_s{k}" in data[s] and f"ours_s{k}" in data[s]]
        t2[s] = {}
        for m in METRICS:
            ds = []
            for k, ok in pairs:
                b, o = data[s][f"baseline_s{k}"].get(m), data[s][ok].get(m)
                if b is not None and o is not None:
                    ds.append((k, o - b))
            if not ds:
                continue
            vals = [d for _, d in ds]
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            imp = sum(1 for v in vals if v < 0)
            t2[s][SHORT[m]] = {"per_seed": {str(k): round(v, 4) for k, v in ds},
                               "mean": round(mean, 4), "std": round(sd, 4),
                               "improved": imp, "n": len(vals)}
            per = ", ".join(f"s{k}:{v:+.3f}" for k, v in ds)
            lines.append(f"| {s} | {SHORT[m]} | {per} | {mean:+.4f}±{sd:.4f} | {imp}/{len(vals)} |")
    summary["T2_paired_deltas"] = t2

    # T3: structure ablation (s1337)
    lines.append("\n## T3 Structure-protection ablation (seed 1337)\n")
    lines.append("| scene | config | " + " | ".join(SHORT[m] for m in METRICS) + " |")
    lines.append("|" + "---|" * (len(METRICS) + 2))
    for s in SCENES:
        for cfg in ("baseline_s1337", "nostruct_s1337", "ours_s1337"):
            if cfg in data[s]:
                lines.append(f"| {s} | {cfg.replace('_s1337','')} | " +
                             " | ".join(fmt(data[s][cfg].get(m)) for m in METRICS) + " |")
    # T4: cross-method (separate intersection, own out_cross dir; coverage-sensitive)
    lines.append("\n## T4 Cross-method under unified protocol (intersection of the cross group only)\n")
    lines.append("| scene | method | " + " | ".join(SHORT[m] for m in METRICS) + " |")
    lines.append("|" + "---|" * (len(METRICS) + 2))
    CROSS = Path("/root/autodl-tmp/eogs/eval_tools/out_cross")
    for s in SCENES:
        p = CROSS / s / f"{s}_metrics.csv"
        if not p.exists():
            continue
        eogsv1_fam = {}
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vals = {m: (float(r[m]) if r.get(m) not in (None, "", "None") else None) for m in METRICS if m in r}
                if re.fullmatch(r"eogsv1(_s\d+)?", r["method"]):
                    eogsv1_fam[r["method"]] = vals
                lines.append(f"| {s} | {r['method']} | " +
                             " | ".join(fmt(vals.get(m)) for m in METRICS) + " |")
        if len(eogsv1_fam) > 1:
            cells = []
            for m in METRICS:
                r2 = ms([v[m] for v in eogsv1_fam.values()])
                cells.append("—" if r2 is None else f"{r2[0]:.3f}±{r2[1]:.3f}")
            lines.append(f"| {s} | eogsv1-family({len(eogsv1_fam)}s) | " + " | ".join(cells) + " |")
        fam = {k: v for k, v in data[s].items() if re.fullmatch(r"baseline_s\d+", k)}
        if fam:
            cells = []
            for m in METRICS:
                r2 = ms([v[m] for v in fam.values()])
                cells.append("—" if r2 is None else f"{r2[0]:.3f}±{r2[1]:.3f}")
            lines.append(f"| {s} | eogspp-fullgrid({len(fam)}s) | " + " | ".join(cells) + " |")

    # T6: convergence (JAX_260, 5k vs 15k)
    lines.append("\n## T6 Convergence diagnostic (JAX_260, 5k vs 15k iterations)\n")
    lines.append("| config | seed | iters | " + " | ".join(SHORT[m] for m in METRICS) + " |")
    lines.append("|" + "---|" * (len(METRICS) + 3))
    d260 = data.get("JAX_260", {})
    for k in (1337, 2024):
        for lab5, lab15, cfg in ((f"baseline_s{k}", f"b15k_s{k}", "baseline"),
                                 (f"ours_s{k}", f"o15k_s{k}", "ours")):
            if lab5 in d260:
                lines.append(f"| {cfg} | {k} | 5000 | " +
                             " | ".join(fmt(d260[lab5].get(m)) for m in METRICS) + " |")
            if lab15 in d260:
                lines.append(f"| {cfg} | {k} | 15000 | " +
                             " | ".join(fmt(d260[lab15].get(m)) for m in METRICS) + " |")

    md = "\n".join(lines)
    (OUT / "summary_tables.md").write_text(md, encoding="utf-8")
    (OUT / "summary_tables.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(md)
    print("\nWROTE", OUT / "summary_tables.md")


if __name__ == "__main__":
    main()
