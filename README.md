# Unmasked Multi-Seed Evaluation Protocol for Satellite 3DGS

Companion code and per-run results for the paper
*On the Reliability of DSM Improvements in Satellite 3D Gaussian Splatting: An Unmasked
Multi-Seed Evaluation Protocol for Weak-Evidence Regions* (under review, 2026).

The protocol evaluates registered DSMs from any reconstruction method over **unmasked
semantic regions**, a **truth-free weak-evidence proxy region**, and an evaluation-only
high-error diagnostic guarded by a **seed-swap null control**, with **multi-seed
reporting** (paired per-seed differences, sign consistency, significance tests). Every
number in the paper's tables is generated from the CSVs in this repository — no manual
transcription.

## Repository layout

```
protocol/    reference implementation of the protocol (paper Sec. III)
casestudy/   exact training / extraction scripts behind every run set in the paper
results/     per-run regional metrics for all evaluated DSMs, on all three grids
tables_out/  output of the table generators (created on run)
```

### `protocol/`

| File | Purpose |
|---|---|
| `eval_pairs_generic.py` | Region-stratified evaluation of an arbitrary set of registered DSM GeoTIFFs against one AOI (MAE/RMSE/completeness per region; anchored weak proxy; error-top20 diagnostic) |
| `weak_proxy_lib.py` | Self-contained weak-evidence proxy construction and region metrics |
| `run_all_evals.py` | Batch driver: EOGS-family full-grid group and cross-method intersection group, evaluated separately |
| `run_pair_evals.py` | Batch driver: two-system pair grid (EOGS vs. EOGS++ incl. the TSDF one-factor variant) |
| `analyze_all.py` | Aggregation to per-scene summary tables (markdown/JSON) |
| `s3_null_control.py` | Seed-swap null control for the error-top20 region (paper Eq. (2)) |
| `s6_phase0_stats.py` | Multi-temporal evidence-complementarity diagnostics |
| `gen_latex_tables.py` | `results/` CSVs → the paper's LaTeX tables (variance, paired deltas, ablation, cross-method) |
| `pair_stats.py` | `results/` CSVs → the published-pair table (two-sample Welch, TSDF one-factor contrast, Holm bookkeeping) |

The batch drivers encode the directory layout of our compute node (documented at the top
of each file); `eval_pairs_generic.py`, `weak_proxy_lib.py`, and both table generators
are layout-independent.

### `casestudy/`

Training scripts for every run set referenced in the paper: the six-seed batteries for
baseline and the evidence-weighted case-study method (`run_q1*.sh`, with the full
hyperparameter environment), structure-protection ablation (`run_s2_nostruct.sh`),
original-EOGS multi-seed reruns (`run_m2a_eogsv1.sh`, `run_q1e/q1g_eogsv1_seeds.sh`,
`eogs_seed.patch`), **unmasked re-registration** of original-EOGS DSMs
(`eval_dsm_unmasked.py`, `run_unmask_v1.sh` — the released EOGS evaluation code sets
water pixels to NaN inside DSM registration; see paper Sec. VI), the **TSDF one-factor
extraction** from existing checkpoints (`run_tsdf_extract.sh`), and the Sat-NeRF
checkpoint evaluation (`run_m2b_satnerf_ckpt.sh`).

### `results/`

- `out/<AOI>/<AOI>_metrics.csv` — EOGS-family full grid: baseline / weighted / ablation
  runs × six seeds per scene, all regional metrics.
- `out_cross/<AOI>/…` — cross-method intersection grid (EOGS, EOGS++, released Sat-NeRF
  checkpoints), with per-grid coverage in the `_meta.json`.
- `out_pair/<AOI>/…` — two-system pair grid (EOGS × 6 seeds, EOGS++ × 6 seeds,
  EOGS++ + TSDF × 6 seeds).
- `out/<AOI>/weak_proxy_rasters/*.tif` — the anchored weak-evidence proxy rasters
  (score, top-20% mask, good-view counts), archived for exact reproducibility.
- `out/s3_null_control.json`, `out/s6_phase0_stats_7scenes.json`,
  `out/summary_tables.{md,json}`.

Seeds used throughout: `{1337 (framework default; disclosed tuning seed), 2024, 3407,
5150, 6001, 7777}`.

## Reproduce the paper's tables

```bash
pip install -r requirements.txt
python protocol/gen_latex_tables.py   # variance / paired / ablation / cross-method tables
python protocol/pair_stats.py         # published-pair table + all Sec. VI statistics
```

Both read only `results/` and write LaTeX fragments to `tables_out/`; `pair_stats.py`
additionally prints the Welch/Wilcoxon/Holm statistics quoted in the paper.

## Evaluate your own method

`eval_pairs_generic.py --scene <AOI> --methods name=/path/to/registered_dsm.tif …`
evaluates any registered DSMs on identical regions (the weak-proxy region is anchored
once per scene and is insensitive to the anchor choice; see paper Sec. III-B). Ground
truth and imagery are **not redistributed** here; obtain them from the public sources
below and adjust the data paths at the top of the scripts.

## Data sources (not redistributed)

- DFC2019 / US3D benchmark (JAX AOIs; truth DSMs and semantic classes), IEEE DataPort.
- IARPA MVS3DM benchmark (IARPA AOIs, San Fernando).
- Official EOGS / EOGS++ data package and code releases; official Sat-NeRF checkpoints.

## License

MIT — see `LICENSE`. Citation information: `CITATION.cff`.
