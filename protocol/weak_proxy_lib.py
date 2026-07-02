#!/usr/bin/env python3
"""Evaluate fixed phase1b-default generalization across scenes.

Truth DSM/CLS are used only for evaluation. The weak proxy used for regional
evaluation is derived from projected pan-image evidence and baseline DSM
geometry, not from Truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox

try:
    from scipy.ndimage import gaussian_filter, sobel
except Exception:  # pragma: no cover
    gaussian_filter = None
    sobel = None


ROOT = Path("/root/autodl-tmp/eogs")
REPO = ROOT / "EOGS2"
CLASS_NAMES = {2: "ground", 5: "tree", 6: "building", 9: "water"}
JAX_CRS = "EPSG:32617"


def scene_short(scene: str) -> str:
    return scene.replace("_", "")


def read_1(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    return arr, profile


def write_raster(path: Path, arr: np.ndarray, profile: dict, nodata=np.nan, dtype="float32") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    prof.pop("blockxsize", None)
    prof.pop("blockysize", None)
    prof.update(driver="GTiff", dtype=dtype, count=1, nodata=nodata, tiled=False)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def align(*arrays: np.ndarray) -> list[np.ndarray]:
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    return [a[:h, :w] for a in arrays]


def truth_georef(data_root: Path, scene: str, src) -> tuple[str | None, Affine, BoundingBox, str]:
    has_geo = src.crs is not None and src.transform is not None and not src.transform.is_identity
    if has_geo:
        return src.crs, src.transform, src.bounds, "GeoTIFF metadata"

    txt_path = data_root / "Truth" / f"{scene}_DSM.txt"
    if txt_path.exists() and txt_path.stat().st_size > 0:
        values = [float(x) for x in txt_path.read_text().split()]
        x0, y0, _nominal_size, pixel_size = values[:4]
        transform = Affine(pixel_size, 0.0, x0, 0.0, pixel_size, y0)
        bounds = BoundingBox(
            left=x0,
            bottom=y0,
            right=x0 + src.width * pixel_size,
            top=y0 + src.height * pixel_size,
        )
        return JAX_CRS if scene.startswith("JAX") else None, transform, bounds, f"{txt_path.name} metadata"

    transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    bounds = BoundingBox(left=0, bottom=0, right=src.width, top=src.height)
    return None, transform, bounds, "identity fallback"


def robust_scale(arr: np.ndarray, lo_q: float = 2.0, hi_q: float = 98.0) -> tuple[np.ndarray, dict]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float32"), {"lo": None, "hi": None}
    lo, hi = np.nanpercentile(vals, [lo_q, hi_q])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0, 1).astype("float32")
    return out, {"lo": float(lo), "hi": float(hi)}


def local_texture(gray: np.ndarray) -> np.ndarray:
    if gaussian_filter is not None:
        smooth = gaussian_filter(gray.astype("float32"), sigma=1.0, mode="nearest")
    else:
        smooth = gray.astype("float32")
    if sobel is not None:
        grad = np.hypot(sobel(smooth, axis=1, mode="nearest"), sobel(smooth, axis=0, mode="nearest"))
    else:
        gy, gx = np.gradient(smooth)
        grad = np.hypot(gx, gy)
    tex, _ = robust_scale(grad.astype("float32"), 5, 98)
    return tex


def load_pan_gray_01(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
    if arr.shape[0] >= 3:
        red, green, blue = arr[0], arr[1], arr[2]
        gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    else:
        gray = arr[0]
    gray01, scale = robust_scale(gray, 2, 98)
    return gray01.astype("float32"), scale


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    valid = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    out = np.full(u.shape, np.nan, dtype="float32")
    if not valid.any():
        return out, valid
    uu = np.clip(u[valid], 0, w - 1)
    vv = np.clip(v[valid], 0, h - 1)
    x0 = np.floor(uu).astype(int)
    y0 = np.floor(vv).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = uu - x0
    dy = vv - y0
    vals = (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    )
    out[valid] = vals.astype("float32")
    return out, valid


def find_image(img_dir: Path, img_name: str) -> Path | None:
    candidates = [
        img_name,
        img_name.replace("_PAN", "_RGB"),
        img_name.replace("PAN", "RGB"),
        img_name.replace("_PAN.tif", "_RGB.tif"),
    ]
    digits = "".join(ch for ch in img_name if ch.isdigit())
    if digits:
        candidates += [p.name for p in sorted(img_dir.glob(f"*{digits}*.tif"))]
    for name in dict.fromkeys(candidates):
        path = img_dir / name
        if path.exists():
            return path
    return None


def compute_weak_proxy(
    scene: str,
    baseline: np.ndarray,
    truth_profile: dict,
    out_dir: Path,
    texture_quantile: float = 60.0,
    brightness_quantile: float = 20.0,
) -> tuple[dict[str, np.ndarray], dict]:
    data_root = REPO / "data"
    truth_path = data_root / "Truth" / f"{scene}_DSM.tif"
    with rasterio.open(truth_path) as src:
        crs, transform, _bounds, georef_source = truth_georef(data_root, scene, src)
        profile = src.profile.copy()
        profile.update(crs=crs, transform=transform)
    h, w = baseline.shape
    profile.update(height=h, width=w)

    z = np.nan_to_num(baseline, nan=float(np.nanmedian(baseline))).astype("float32")
    rows, cols = np.indices((h, w), dtype="float32")
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e

    affine_path = REPO / "scripts/dataset_creation/affine_models" / f"{scene}_pan_rpc_raw" / "affine_models.json"
    affine_all = json.loads(affine_path.read_text())
    metadata = [m for m in affine_all["pan"] if not m.get("virtual_camera", False)]
    img_dir = data_root / "images" / "pan" / scene
    if not metadata:
        raise RuntimeError(f"No pan affine metadata for {scene}")

    center = np.array(metadata[0]["model"]["center"], dtype="float32")
    scale = float(metadata[0]["model"]["scale"])
    xyz_norm = (np.stack([xs, ys, z], axis=-1) - center) / scale

    texture_samples = []
    brightness_samples = []
    valid_samples = []
    reports = []
    for meta in metadata:
        img_path = find_image(img_dir, meta["img"])
        if img_path is None:
            reports.append({"img": meta["img"], "status": "missing"})
            continue
        gray, gray_scale = load_pan_gray_01(img_path)
        tex = local_texture(gray)
        coef = np.array(meta["model"]["coef_"], dtype="float32")
        inter = np.array(meta["model"]["intercept_"], dtype="float32")
        view = xyz_norm @ coef + inter
        u = ((view[..., 0] + 1.0) * 0.5) * float(meta["width"]) - 0.5
        v = ((view[..., 1] + 1.0) * 0.5) * float(meta["height"]) - 0.5
        tex_s, valid_t = bilinear_sample(tex, u, v)
        bright_s, valid_b = bilinear_sample(gray, u, v)
        valid = valid_t & valid_b & np.isfinite(tex_s) & np.isfinite(bright_s)
        texture_samples.append(tex_s)
        brightness_samples.append(bright_s)
        valid_samples.append(valid)
        reports.append(
            {
                "img": meta["img"],
                "status": "used",
                "valid_fraction": float(valid.mean()),
                "gray_scale": gray_scale,
            }
        )

    if not texture_samples:
        raise RuntimeError(f"No usable pan images for {scene}")

    texture_stack = np.stack(texture_samples).astype("float32")
    brightness_stack = np.stack(brightness_samples).astype("float32")
    valid_stack = np.stack(valid_samples).astype(bool)
    texture_stack[~valid_stack] = np.nan
    brightness_stack[~valid_stack] = np.nan

    valid_view_count = valid_stack.sum(axis=0).astype("float32")
    valid_texture = texture_stack[np.isfinite(texture_stack)]
    valid_brightness = brightness_stack[np.isfinite(brightness_stack)]
    texture_thr = float(np.nanpercentile(valid_texture, texture_quantile))
    brightness_thr = float(np.nanpercentile(valid_brightness, brightness_quantile))
    good_stack = valid_stack & (texture_stack >= texture_thr) & (brightness_stack >= brightness_thr)
    good_view_count = good_stack.sum(axis=0).astype("float32")

    mean_texture = np.nanmean(texture_stack, axis=0).astype("float32")
    median_brightness = np.nanmedian(brightness_stack, axis=0).astype("float32")
    good_ratio = good_view_count / np.maximum(valid_view_count, 1.0)
    low_texture = 1.0 - np.nan_to_num(mean_texture, nan=0.0)
    dark = 1.0 - np.nan_to_num(median_brightness, nan=0.0)
    uncovered = (valid_view_count <= 0).astype("float32")
    weak_score = 0.60 * (1.0 - good_ratio) + 0.25 * low_texture + 0.15 * dark
    weak_score = np.clip(weak_score + 0.50 * uncovered, 0, 1).astype("float32")

    valid_proxy = valid_view_count > 0
    thr = float(np.nanpercentile(weak_score[valid_proxy], 80)) if valid_proxy.any() else 1.0
    weak_top20 = valid_proxy & (weak_score >= thr)

    rasters = {
        "valid_view_count": valid_view_count,
        "good_view_count": good_view_count,
        "mean_projected_texture": mean_texture,
        "median_projected_brightness": median_brightness,
        "weak_proxy_score": weak_score,
        "weak_proxy_top20": weak_top20.astype("uint8"),
    }
    raster_dir = out_dir / "weak_proxy_rasters"
    for name, arr in rasters.items():
        dtype = "uint8" if name.endswith("top20") else "float32"
        nodata = 0 if dtype == "uint8" else np.nan
        write_raster(raster_dir / f"{scene}_{name}.tif", arr, profile, nodata=nodata, dtype=dtype)

    meta = {
        "scene": scene,
        "georef_source": georef_source,
        "affine_models": str(affine_path),
        "pan_images": str(img_dir),
        "num_pan_images_used": len(texture_samples),
        "texture_quantile": texture_quantile,
        "brightness_quantile": brightness_quantile,
        "texture_threshold": texture_thr,
        "brightness_threshold": brightness_thr,
        "weak_proxy_rule": "top 20% of 0.60*(1-good_view_ratio)+0.25*(1-mean_texture)+0.15*(1-median_brightness), image-derived only",
        "weak_proxy_threshold_p80": thr,
        "weak_proxy_area_ratio_valid_proxy": float(weak_top20.sum() / max(1, valid_proxy.sum())),
        "per_image_reports": reports,
    }
    return rasters, meta


def region_metrics(pred: np.ndarray, truth: np.ndarray, cls: np.ndarray, valid: np.ndarray, weak_top20: np.ndarray, baseline_error_top20: np.ndarray) -> dict:
    diff = np.abs(pred - truth).astype("float32")
    total = float(np.nansum(diff[valid]))
    regions = {
        "overall": np.ones_like(valid, dtype=bool),
        "water_cls9": cls == 9,
        "building_cls6": cls == 6,
        "ground_cls2": cls == 2,
        "tree_cls5": cls == 5,
        "weak_proxy_top20": weak_top20 > 0,
        "phase0_error_top20_eval_only": baseline_error_top20,
    }
    row = {}
    for name, mask0 in regions.items():
        mask = valid & mask0
        vals = diff[mask]
        row[f"{name}_mae"] = float(np.nanmean(vals)) if vals.size else None
        row[f"{name}_median"] = float(np.nanmedian(vals)) if vals.size else None
        row[f"{name}_pixels"] = int(mask.sum())
        row[f"{name}_area_ratio"] = float(mask.sum() / max(1, valid.sum()))
        row[f"{name}_error_contribution"] = float(np.nansum(vals) / max(total, 1e-6)) if vals.size else 0.0
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_scene(out_path: Path, scene: str, baseline: np.ndarray, phase3: np.ndarray, truth: np.ndarray, cls: np.ndarray, weak_score: np.ndarray, weak_top20: np.ndarray, valid: np.ndarray) -> None:
    base_err = np.abs(baseline - truth)
    p3_err = np.abs(phase3 - truth)
    change = p3_err - base_err
    vmin = float(np.nanpercentile(np.concatenate([baseline[valid], phase3[valid]]), 2))
    vmax = float(np.nanpercentile(np.concatenate([baseline[valid], phase3[valid]]), 98))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [
        (baseline, "baseline DSM", "terrain", (vmin, vmax)),
        (phase3, "phase3 DSM", "terrain", (vmin, vmax)),
        (np.clip(base_err, 0, 6), "baseline abs error", "magma", (0, 6)),
        (np.clip(p3_err, 0, 6), "phase3 abs error", "magma", (0, 6)),
        (np.clip(change, -2, 2), "error change p3-baseline", "coolwarm", (-2, 2)),
        (weak_score, "weak proxy score", "inferno", (0, 1)),
        (weak_top20, "weak proxy top20", "gray", (0, 1)),
        (cls, "Truth CLS eval only", "tab20", None),
    ]
    for ax, (arr, title, cmap, lim) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Phase3 fixed phase1b-default: {scene}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def eval_scene(scene: str, out_dir: Path) -> tuple[list[dict], dict]:
    data_root = REPO / "data"
    short = scene_short(scene)
    baseline_path = REPO / "output" / f"phase1_eogsplus_rawrpc_{short}_pan_3PAN_5000" / "test_opNone/ours_5000/rdsm" / f"{scene}_rdsm.tif"
    phase3_path = REPO / "output" / f"phase3_phase1b_default_{short}_pan_3PAN_5000" / "test_opNone/ours_5000/rdsm" / f"{scene}_rdsm.tif"
    truth_path = data_root / "Truth" / f"{scene}_DSM.tif"
    cls_path = data_root / "Truth" / f"{scene}_CLS.tif"
    for p in [baseline_path, phase3_path, truth_path, cls_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    baseline, base_profile = read_1(baseline_path)
    phase3, _ = read_1(phase3_path)
    truth, truth_profile = read_1(truth_path)
    cls, _ = read_1(cls_path)
    baseline, phase3, truth, cls = align(baseline, phase3, truth, cls)
    cls = cls.astype("int32")

    scene_out = out_dir / scene
    proxy, proxy_meta = compute_weak_proxy(scene, baseline, truth_profile, scene_out)
    weak_top20 = proxy["weak_proxy_top20"].astype(bool)
    weak_score = proxy["weak_proxy_score"]
    baseline, phase3, truth, cls, weak_top20, weak_score = align(baseline, phase3, truth, cls, weak_top20, weak_score)
    common_valid = np.isfinite(baseline) & np.isfinite(phase3) & np.isfinite(truth)
    baseline_error = np.abs(baseline - truth)
    error_thr = float(np.nanpercentile(baseline_error[common_valid], 80))
    baseline_error_top20 = common_valid & (baseline_error >= error_thr)

    rows = []
    base_row = {"scene": scene, "method": "baseline", **region_metrics(baseline, truth, cls, common_valid, weak_top20, baseline_error_top20)}
    p3_row = {"scene": scene, "method": "phase3_phase1b_default", **region_metrics(phase3, truth, cls, common_valid, weak_top20, baseline_error_top20)}
    rows.extend([base_row, p3_row])
    delta = {"scene": scene, "method": "delta_phase3_minus_baseline"}
    for k, v in p3_row.items():
        if k in {"scene", "method"}:
            continue
        bv = base_row.get(k)
        delta[k] = (v - bv) if isinstance(v, (int, float)) and isinstance(bv, (int, float)) and "pixels" not in k and "area_ratio" not in k else None
    rows.append(delta)

    write_raster(scene_out / f"{scene}_phase3_error_change.tif", np.abs(phase3 - truth) - baseline_error, base_profile, nodata=np.nan)
    plot_scene(scene_out / f"{scene}_phase3_maps.png", scene, baseline, phase3, truth, cls, weak_score, weak_top20.astype("float32"), common_valid)

    summary = {
        "scene": scene,
        "inputs": {
            "baseline": str(baseline_path),
            "phase3": str(phase3_path),
            "truth_dsm": str(truth_path),
            "truth_cls": str(cls_path),
        },
        "truth_usage": "Truth DSM/CLS used only for evaluation and phase0_error_top20_eval_only diagnostic.",
        "weak_proxy": proxy_meta,
        "common_valid_pixels": int(common_valid.sum()),
        "baseline_error_p80_for_eval_diagnostic": error_thr,
        "rows": rows,
        "figures": {
            "maps": str(scene_out / f"{scene}_phase3_maps.png"),
            "error_change": str(scene_out / f"{scene}_phase3_error_change.tif"),
        },
    }
    (scene_out / f"{scene}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/root/autodl-tmp/eogs/phase3/eval")
    parser.add_argument("--scenes", nargs="+", default=["JAX_214", "JAX_004", "JAX_068"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summaries = []
    failures = []
    for scene in args.scenes:
        try:
            rows, summary = eval_scene(scene, out_dir)
            all_rows.extend(rows)
            summaries.append(summary)
            print(f"[OK] {scene}")
        except Exception as exc:
            failures.append({"scene": scene, "error": repr(exc)})
            print(f"[FAIL] {scene}: {exc}")

    write_csv(out_dir / "phase3_metrics.csv", all_rows)
    summary = {
        "task": "Phase3 fixed phase1b-default cross-scene generalization",
        "fixed_parameters": {
            "EOGS_VIEW_EVIDENCE_FLOOR": 0.60,
            "EOGS_VIEW_EVIDENCE_TEXTURE_Q": 0.70,
            "EOGS_VIEW_EVIDENCE_BRIGHT_Q": 0.15,
            "EOGS_VIEW_EVIDENCE_TEXTURE_WEIGHT": 0.45,
            "EOGS_VIEW_EVIDENCE_BRIGHT_WEIGHT": 0.25,
            "EOGS_VIEW_EVIDENCE_RESIDUAL_WEIGHT": 0.30,
            "EOGS_VIEW_EVIDENCE_RESIDUAL_START": 1500,
            "EOGS_VIEW_EVIDENCE_NORMALIZE": 1,
            "EOGS_VIEW_EVIDENCE_BLEND": 1.0,
            "EOGS_VIEW_STRUCTURE_EDGE_Q": 0.78,
            "EOGS_VIEW_STRUCTURE_LOW": 0.20,
            "EOGS_VIEW_STRUCTURE_HIGH": 0.58,
            "EOGS_VIEW_STRUCTURE_POWER": 1.20,
            "EOGS_VIEW_STRUCTURE_KERNEL": 5,
        },
        "truth_usage": "Training/refinement uses no truth. Truth is used only in this evaluation script.",
        "scenes": args.scenes,
        "summaries": summaries,
        "failures": failures,
        "metrics_csv": str(out_dir / "phase3_metrics.csv"),
    }
    (out_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
