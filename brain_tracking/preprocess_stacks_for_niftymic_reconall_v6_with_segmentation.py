#!/usr/bin/env python3
"""
Recon-all-clinical-inspired preprocessing with edge-line repair (no brain segmentation).

This script is intentionally separate from prior scripts and writes to new output
folders so previous outputs remain untouched.

Pipeline per stack (axial/coronal/sagittal):
1) Raw baseline
2) Robust clip only (no denoise)
3) Edge-line repair (boundary stripe/dropout fix)
4) N4 bias field correction
5) Slice intensity harmonization (reduce motion-like slice jumps)
6) Outside-noise attenuation using robust foreground mask (light)
7) Background ghost suppression (slice-wise outside-mask cleanup)
8) Brain-focused robust normalization
9) Final outside cleanup

Cross-stack clinical-style step:
9) Histogram harmonization of stacks to a common reference (axial)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation, label
except Exception:
    gaussian_filter = None
    binary_fill_holes = None
    binary_dilation = None
    label = None

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


@dataclass
class StepMetrics:
    snr95: float
    nonzero_fraction: float
    outside_energy_fraction: float
    slice_mean_cv: float


@dataclass
class StackReport:
    name: str
    input_path: str
    shape: Tuple[int, int, int]
    zooms_mm: Tuple[float, float, float]
    orientation: Tuple[str, str, str]
    metrics_by_step: Dict[str, StepMetrics]
    outputs: Dict[str, str]


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _clear_pngs(path: str) -> None:
    if not os.path.isdir(path):
        return
    for fn in os.listdir(path):
        if fn.lower().endswith(".png"):
            try:
                os.remove(os.path.join(path, fn))
            except OSError:
                pass


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img


def save_nifti_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: str, dtype=np.float32) -> None:
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    out = nib.Nifti1Image(data.astype(dtype), affine=ref_img.affine, header=hdr)
    nib.save(out, out_path)


def robust_foreground_mask(data: np.ndarray, low_q: float = 20.0) -> np.ndarray:
    pos = data[data > 0]
    if pos.size == 0:
        return np.zeros_like(data, dtype=bool)

    thr = float(np.percentile(pos, low_q))
    m = data > thr

    if label is not None:
        cc, n_cc = label(m)
        if n_cc > 1:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            m = cc == int(sizes.argmax())

    if binary_fill_holes is not None:
        m = binary_fill_holes(m)

    return m.astype(bool)


def estimate_snr95(data: np.ndarray) -> float:
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return 0.0

    p95 = float(np.percentile(vals, 95))
    nx, ny, _nz = data.shape
    cx = max(4, nx // 16)
    cy = max(4, ny // 16)

    corners = np.concatenate(
        [
            data[:cx, :cy, :].ravel(),
            data[nx - cx :, :cy, :].ravel(),
            data[:cx, ny - cy :, :].ravel(),
            data[nx - cx :, ny - cy :, :].ravel(),
        ]
    )
    mu = float(np.median(corners))
    sigma = float(np.std(corners))
    return float((p95 - mu) / (sigma + 1e-6))


def compute_metrics(data: np.ndarray, fg_mask: np.ndarray | None = None) -> StepMetrics:
    if fg_mask is None:
        fg_mask = robust_foreground_mask(data, low_q=20.0)

    fg = fg_mask > 0
    bg = ~fg

    fg_energy = float(np.sum(np.square(data[fg]))) if np.any(fg) else 0.0
    bg_energy = float(np.sum(np.square(data[bg]))) if np.any(bg) else 0.0
    outside_energy_fraction = bg_energy / (fg_energy + bg_energy + 1e-6)

    z_profile = data.mean(axis=(0, 1))
    z_mean = float(np.mean(z_profile))
    z_std = float(np.std(z_profile))
    slice_mean_cv = z_std / (abs(z_mean) + 1e-6)

    return StepMetrics(
        snr95=float(estimate_snr95(data)),
        nonzero_fraction=float(np.count_nonzero(data > 0) / data.size),
        outside_energy_fraction=float(outside_energy_fraction),
        slice_mean_cv=float(slice_mean_cv),
    )


def clip_outliers(data: np.ndarray) -> np.ndarray:
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(vals, [0.5, 99.8])
    return np.clip(data, lo, hi).astype(np.float32)


def _edge_dropout_width(row_or_col_means: np.ndarray, max_width: int = 2, ratio: float = 0.45) -> Tuple[int, int]:
    """Estimate dropout width at both ends of a 1D profile (top/bottom or left/right)."""
    n = row_or_col_means.size
    if n < 8:
        return 0, 0

    inner = row_or_col_means[max_width : n - max_width]
    inner_ref = float(np.median(inner)) if inner.size > 0 else float(np.median(row_or_col_means))
    if inner_ref <= 1e-6:
        return 0, 0

    top_w = 0
    for i in range(max_width):
        if float(row_or_col_means[i]) < ratio * inner_ref:
            top_w += 1
        else:
            break

    bottom_w = 0
    for i in range(max_width):
        if float(row_or_col_means[n - 1 - i]) < ratio * inner_ref:
            bottom_w += 1
        else:
            break

    return top_w, bottom_w


def _repair_slice_edges(sl: np.ndarray, max_width: int = 2) -> np.ndarray:
    """Repair missing/weak border lines by interior extrapolation (no blur)."""
    out = sl.copy().astype(np.float32)
    h, w = out.shape
    if h < 8 or w < 8:
        return out

    row_means = out.mean(axis=1)
    col_means = out.mean(axis=0)
    top_w, bottom_w = _edge_dropout_width(row_means, max_width=max_width)
    left_w, right_w = _edge_dropout_width(col_means, max_width=max_width)

    # Top repair
    if top_w > 0 and top_w + 2 < h:
        src = out[top_w : top_w + 2, :]
        for i in range(top_w):
            out[top_w - 1 - i, :] = src[0, :] + (src[0, :] - src[1, :]) * (i + 1)

    # Bottom repair
    if bottom_w > 0 and h - bottom_w - 3 >= 0:
        src = out[h - bottom_w - 2 : h - bottom_w, :]
        for i in range(bottom_w):
            out[h - bottom_w + i, :] = src[1, :] + (src[1, :] - src[0, :]) * (i + 1)

    # Left repair
    if left_w > 0 and left_w + 2 < w:
        src = out[:, left_w : left_w + 2]
        for i in range(left_w):
            out[:, left_w - 1 - i] = src[:, 0] + (src[:, 0] - src[:, 1]) * (i + 1)

    # Right repair
    if right_w > 0 and w - right_w - 3 >= 0:
        src = out[:, w - right_w - 2 : w - right_w]
        for i in range(right_w):
            out[:, w - right_w + i] = src[:, 1] + (src[:, 1] - src[:, 0]) * (i + 1)

    return np.maximum(out, 0.0).astype(np.float32)


def repair_edge_lines(data: np.ndarray, max_width: int = 2) -> np.ndarray:
    """Apply edge-line repair to each thick-slice plane (axis=2)."""
    out = np.zeros_like(data, dtype=np.float32)
    for z in range(data.shape[2]):
        out[:, :, z] = _repair_slice_edges(data[:, :, z], max_width=max_width)
    return out


def mri_edge_preserving_denoise(data: np.ndarray) -> np.ndarray:
    x = clip_outliers(data)
    if sitk is None:
        return x

    itk = sitk.GetImageFromArray(np.transpose(x, (2, 1, 0)).astype(np.float32))
    den = sitk.CurvatureAnisotropicDiffusion(
        itk,
        timeStep=0.05,
        conductanceParameter=2.0,
        numberOfIterations=6,
    )
    out = sitk.GetArrayFromImage(den)
    return np.transpose(out, (2, 1, 0)).astype(np.float32)


def estimate_background_sigma(data: np.ndarray, fg_mask: np.ndarray | None = None) -> float:
    """Estimate background noise sigma from corners and/or outside foreground."""
    nx, ny, _nz = data.shape
    cx = max(4, nx // 16)
    cy = max(4, ny // 16)

    corners = np.concatenate(
        [
            data[:cx, :cy, :].ravel(),
            data[nx - cx :, :cy, :].ravel(),
            data[:cx, ny - cy :, :].ravel(),
            data[nx - cx :, ny - cy :, :].ravel(),
        ]
    )

    samples = corners
    if fg_mask is not None:
        bg = data[~(fg_mask > 0)]
        if bg.size > 100:
            samples = np.concatenate([corners, bg.ravel()])

    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return 0.0
    return float(np.std(samples))


def rician_noise_floor_correction(data: np.ndarray, fg_mask: np.ndarray | None = None) -> np.ndarray:
    """Approximate Rician bias correction: S_corr = sqrt(max(S^2 - 2*sigma^2, 0))."""
    sigma = estimate_background_sigma(data, fg_mask)
    if sigma <= 1e-6:
        return data.astype(np.float32)

    corrected = np.sqrt(np.maximum(np.square(data) - 2.0 * sigma * sigma, 0.0))
    return corrected.astype(np.float32)


def n4_bias_correct(data: np.ndarray) -> np.ndarray:
    if sitk is None:
        return data.astype(np.float32)

    arr = np.transpose(data, (2, 1, 0)).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected = n4.Execute(img, mask)

    out = sitk.GetArrayFromImage(corrected)
    return np.transpose(out, (2, 1, 0)).astype(np.float32)


def suppress_gibbs_ringing_inplane(data: np.ndarray, strength: float = 0.06) -> np.ndarray:
    """Conservative in-plane Gibbs suppression via mild k-space tapering.

    Note: this is intentionally weak to avoid noticeable blurring.
    """
    strength = float(np.clip(strength, 0.0, 0.2))
    if strength <= 0.0:
        return data.astype(np.float32)

    nx, ny, nz = data.shape
    wx = np.hanning(nx)
    wy = np.hanning(ny)
    window = np.outer(wx, wy).astype(np.float32)
    window = (1.0 - strength) + strength * window

    out = np.zeros_like(data, dtype=np.float32)
    for z in range(nz):
        sl = data[:, :, z]
        k = np.fft.fft2(sl)
        sl_f = np.real(np.fft.ifft2(k * window))
        out[:, :, z] = sl_f.astype(np.float32)

    return out


def slice_intensity_harmonize(data: np.ndarray, target_percentile: float = 90.0) -> np.ndarray:
    out = np.zeros_like(data, dtype=np.float32)
    slice_scale: List[float] = []

    for z in range(data.shape[2]):
        sl = data[:, :, z]
        pos = sl[sl > 0]
        if pos.size < 20:
            slice_scale.append(1.0)
            continue
        slice_scale.append(float(np.percentile(pos, target_percentile)))

    ref = max(float(np.median(slice_scale)) if len(slice_scale) > 0 else 1.0, 1e-6)

    for z in range(data.shape[2]):
        s = max(float(slice_scale[z]), 1e-6)
        gain = np.clip(ref / s, 0.7, 1.4)
        out[:, :, z] = (data[:, :, z] * gain).astype(np.float32)

    return out


def attenuate_outside_noise(data: np.ndarray, fg_mask: np.ndarray, edge_soft_sigma: float = 1.2) -> np.ndarray:
    m = fg_mask.astype(np.float32)
    if gaussian_filter is not None and edge_soft_sigma > 0:
        m = gaussian_filter(m, sigma=edge_soft_sigma)
        m = np.clip(m, 0.0, 1.0)

    soft = 0.15 + 0.85 * m
    return (data * soft).astype(np.float32)


def suppress_background_ghosts(data: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Suppress slice-wise bright background ghosts while preserving brain region."""
    out = data.copy().astype(np.float32)
    fg = fg_mask > 0
    bg = ~fg

    if not np.any(bg):
        return out

    global_bg_median = float(np.median(out[bg]))

    for z in range(out.shape[2]):
        sl = out[:, :, z]
        sl_bg = ~fg[:, :, z]
        if np.count_nonzero(sl_bg) < 64:
            continue

        bvals = sl[sl_bg]
        bmed = float(np.median(bvals))
        b95 = float(np.percentile(bvals, 95.0))

        # Cap high-intensity background ghosts, then align background offset.
        sl[sl_bg] = np.minimum(sl[sl_bg], b95)
        sl = sl - bmed + global_bg_median
        sl = np.maximum(sl, 0.0)
        out[:, :, z] = sl

    return out.astype(np.float32)


def final_outside_cleanup(data: np.ndarray, fg_mask: np.ndarray, outside_scale: float = 0.05) -> np.ndarray:
    """After normalization, clamp residual outside noise to a low foreground-derived floor."""
    out = data.copy().astype(np.float32)
    fg = fg_mask > 0
    if not np.any(fg):
        return out

    fg_vals = out[fg]
    fg_p2 = float(np.percentile(fg_vals, 2.0))
    floor = max(0.0, outside_scale * fg_p2)
    out[~fg] = np.minimum(out[~fg], floor)
    return out.astype(np.float32)


def mri_robust_normalize(data: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    fg = fg_mask > 0
    vals = data[fg] if np.any(fg) else data[np.isfinite(data)]
    if vals.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    p2, p98 = np.percentile(vals, [2.0, 98.0])
    med = float(np.median(vals))
    scale = max(p98 - p2, 1e-6)

    x = (data - med) / scale
    x = np.clip(x, -0.7, 1.3)
    x = (x + 0.7) / 2.0
    x = np.clip(x, 0.0, 1.0)
    return (x * 2047.0).astype(np.float32)


def _slice(data: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 2:
        sl = data[:, :, idx]
    elif axis == 1:
        sl = data[:, idx, :]
    else:
        sl = data[idx, :, :]
    return np.rot90(sl, k=1)


def save_4x4_montage(data: np.ndarray, axis: int, title: str, out_path: str) -> None:
    n = data.shape[axis]
    if n <= 0:
        return

    ids = np.linspace(0, n - 1, min(16, n)).astype(int)

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    rows, cols = 4, 4

    vmin = float(np.percentile(data, 1))
    vmax = float(np.percentile(data, 99))
    vmax = max(vmax, vmin + 1e-6)

    for i, sid in enumerate(ids[:16]):
        r = i // cols
        c = i % cols
        ax = fig.add_axes([c / cols, 1.0 - (r + 1) / rows, 1.0 / cols, 1.0 / rows])
        ax.imshow(_slice(data, axis=axis, idx=int(sid)), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, color="white", fontsize=10, y=0.995)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def save_overlay_montage(data: np.ndarray, mask: np.ndarray, axis: int, title: str, out_path: str) -> None:
    n = data.shape[axis]
    if n <= 0:
        return

    ids = np.linspace(0, n - 1, min(16, n)).astype(int)

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    rows, cols = 4, 4

    vmin = float(np.percentile(data, 1))
    vmax = float(np.percentile(data, 99))
    vmax = max(vmax, vmin + 1e-6)

    for i, sid in enumerate(ids[:16]):
        r = i // cols
        c = i % cols
        ax = fig.add_axes([c / cols, 1.0 - (r + 1) / rows, 1.0 / cols, 1.0 / rows])
        sl = _slice(data, axis=axis, idx=int(sid))
        mk = _slice(mask, axis=axis, idx=int(sid))
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
        ax.contour(mk > 0.5, levels=[0.5], colors="lime", linewidths=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, color="white", fontsize=10, y=0.995)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def save_step_triplet(data: np.ndarray, step_tag: str, qc_dir: str, name: str) -> None:
    save_4x4_montage(data, 2, f"{name} | {step_tag} | axial", os.path.join(qc_dir, f"{step_tag}_axial_4x4.png"))
    save_4x4_montage(data, 1, f"{name} | {step_tag} | coronal", os.path.join(qc_dir, f"{step_tag}_coronal_4x4.png"))
    save_4x4_montage(data, 0, f"{name} | {step_tag} | sagittal", os.path.join(qc_dir, f"{step_tag}_sagittal_4x4.png"))


def histogram_match_to_reference(
    src_data: np.ndarray,
    ref_data: np.ndarray,
    src_mask: np.ndarray,
    ref_mask: np.ndarray,
    n_quantiles: int = 256,
) -> np.ndarray:
    """Match source stack intensity distribution to reference within foreground masks."""
    out = src_data.copy().astype(np.float32)

    svals = src_data[src_mask > 0]
    rvals = ref_data[ref_mask > 0]
    if svals.size < 200 or rvals.size < 200:
        return out

    qs = np.linspace(0.0, 100.0, n_quantiles)
    sq = np.percentile(svals, qs)
    rq = np.percentile(rvals, qs)

    sq = np.maximum.accumulate(sq)
    rq = np.maximum.accumulate(rq)

    out = np.interp(src_data.ravel(), sq, rq, left=rq[0], right=rq[-1]).reshape(src_data.shape)
    return out.astype(np.float32)


def process_stack(name: str, input_path: str, out_dir: str) -> StackReport:
    _mkdir(out_dir)
    qc_dir = os.path.join(out_dir, "qc")
    _mkdir(qc_dir)
    _clear_pngs(qc_dir)

    data0, img = load_nifti(input_path)
    orient = tuple(str(c) for c in nib.aff2axcodes(img.affine))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

    m0 = robust_foreground_mask(data0, low_q=20.0)
    metrics: Dict[str, StepMetrics] = {"step0_raw": compute_metrics(data0, m0)}
    save_step_triplet(data0, "step0_raw", qc_dir, name)

    data1 = clip_outliers(data0)
    metrics["step1_clip_only"] = compute_metrics(data1)
    save_step_triplet(data1, "step1_clip_only", qc_dir, name)

    data2 = repair_edge_lines(data1, max_width=2)
    metrics["step2_edge_line_repair"] = compute_metrics(data2)
    save_step_triplet(data2, "step2_edge_line_repair", qc_dir, name)

    data3 = n4_bias_correct(data2)
    metrics["step3_n4_bias_correct"] = compute_metrics(data3)
    save_step_triplet(data3, "step3_n4_bias_correct", qc_dir, name)

    data4 = slice_intensity_harmonize(data3, target_percentile=90.0)
    metrics["step4_slice_harmonize"] = compute_metrics(data4)
    save_step_triplet(data4, "step4_slice_harmonize", qc_dir, name)

    fg = robust_foreground_mask(data4, low_q=20.0)
    data5 = attenuate_outside_noise(data4, fg, edge_soft_sigma=0.6)
    metrics["step5_outside_noise_atten_light"] = compute_metrics(data5, fg)
    save_step_triplet(data5, "step5_outside_noise_atten_light", qc_dir, name)

    data6 = suppress_background_ghosts(data5, fg)
    metrics["step6_bg_ghost_suppress"] = compute_metrics(data6, fg)
    save_step_triplet(data6, "step6_bg_ghost_suppress", qc_dir, name)

    data7 = mri_robust_normalize(data6, fg)
    metrics["step7_norm"] = compute_metrics(data7, fg)
    save_step_triplet(data7, "step7_norm", qc_dir, name)

    data8 = final_outside_cleanup(data7, fg, outside_scale=0.12)
    metrics["step8_final_outside_cleanup"] = compute_metrics(data8, fg)
    save_step_triplet(data8, "step8_final_outside_cleanup", qc_dir, name)

    for axis, axis_name in [(2, "axial"), (1, "coronal"), (0, "sagittal")]:
        save_overlay_montage(
            data8,
            fg.astype(np.uint8),
            axis,
            f"{name} | step8_final + fg mask | {axis_name}",
            os.path.join(qc_dir, f"step8_final_fg_overlay_{axis_name}.png"),
        )

    preproc_img = os.path.join(out_dir, f"{name}_preproc_reconall_v6_base.nii.gz")
    fg_mask = os.path.join(out_dir, f"{name}_fgmask_reconall_v6.nii.gz")
    save_nifti_like(img, data8, preproc_img, dtype=np.float32)
    save_nifti_like(img, fg.astype(np.uint8), fg_mask, dtype=np.uint8)

    outputs = {
        "preprocessed_image_base": preproc_img,
        "preprocessed_image": preproc_img,
        "foreground_mask": fg_mask,
        "qc_dir": qc_dir,
    }

    return StackReport(
        name=name,
        input_path=input_path,
        shape=tuple(int(x) for x in data0.shape),
        zooms_mm=zooms,
        orientation=orient,
        metrics_by_step=metrics,
        outputs=outputs,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clinical-style stack preprocessing with edge repair (no segmentation)")
    p.add_argument("--input-dir", type=str, default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40")
    p.add_argument("--output-dir", type=str, default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/reconall_v6_with_seg")
    p.add_argument("--fixed-fig-dir", type=str, default="DataSRR/volunteer_xxx/figures/reconall_v6_with_seg")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _mkdir(args.output_dir)
    _mkdir(args.fixed_fig_dir)

    stacks = {
        "axial": os.path.join(args.input_dir, "axial_native.nii.gz"),
        "coronal": os.path.join(args.input_dir, "coronal_native.nii.gz"),
        "sagittal": os.path.join(args.input_dir, "sagittal_native.nii.gz"),
    }

    missing = [k for k, v in stacks.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"Missing stack files for: {missing} in {args.input_dir}")

    reports: List[StackReport] = []
    for name, path in stacks.items():
        stack_out = os.path.join(args.output_dir, name)
        rep = process_stack(name, path, stack_out)
        reports.append(rep)

    # Cross-stack harmonization inspired by recon-all-clinical intensity harmonization.
    ref_name = "axial"
    rep_map = {r.name: r for r in reports}
    ref_rep = rep_map[ref_name]
    ref_data, ref_img = load_nifti(ref_rep.outputs["preprocessed_image_base"])
    ref_mask, _ = load_nifti(ref_rep.outputs["foreground_mask"])

    for name, rep in rep_map.items():
        src_data, src_img = load_nifti(rep.outputs["preprocessed_image_base"])
        src_mask, _ = load_nifti(rep.outputs["foreground_mask"])

        if name == ref_name:
            harm = src_data
        else:
            harm = histogram_match_to_reference(src_data, ref_data, src_mask, ref_mask, n_quantiles=256)

        rep.metrics_by_step["step9_cross_stack_harmonize"] = compute_metrics(harm, src_mask > 0)
        save_step_triplet(harm, "step9_cross_stack_harmonize", rep.outputs["qc_dir"], name)

        final_path = os.path.join(args.output_dir, name, f"{name}_preproc_reconall_v6_no_seg.nii.gz")
        save_nifti_like(src_img, harm, final_path, dtype=np.float32)
        rep.outputs["preprocessed_image"] = final_path

    for rep in reports:
        src_qc = rep.outputs["qc_dir"]
        dst_qc = os.path.join(args.fixed_fig_dir, rep.name)
        _mkdir(dst_qc)
        _clear_pngs(dst_qc)
        for fn in os.listdir(src_qc):
            src = os.path.join(src_qc, fn)
            dst = os.path.join(dst_qc, fn)
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "fixed_figure_dir": args.fixed_fig_dir,
        "segmentation_used": False,
        "pipeline": [
            "step0_raw",
            "step1_clip_only",
            "step2_edge_line_repair",
            "step3_n4_bias_correct",
            "step4_slice_harmonize",
            "step5_outside_noise_atten_light",
            "step6_bg_ghost_suppress",
            "step7_norm",
            "step8_final_outside_cleanup",
            "step9_cross_stack_harmonize",
        ],
        "stacks": [asdict(r) for r in reports],
    }

    summary_path = os.path.join(args.output_dir, "preprocess_reconall_v6_with_segmentation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Finished clinical-style preprocessing with edge-line repair (no segmentation, v6)")
    print("=" * 80)
    print("Input dir:", args.input_dir)
    print("Output dir:", args.output_dir)
    print("Figure dir:", args.fixed_fig_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
