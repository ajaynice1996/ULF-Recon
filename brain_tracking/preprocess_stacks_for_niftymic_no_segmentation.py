#!/usr/bin/env python3
"""
Preprocess low-field orthogonal stacks for NiftyMIC without segmentation.

This pipeline avoids SynthStrip/skull stripping and focuses on stepwise quality
improvements for axial/coronal/sagittal stacks with QC after every step.

MRI-specific steps per stack (inspired by recon-all preprocessing logic):
1) Raw baseline
2) Edge-preserving MRI denoise (anisotropic diffusion / ITK)
3) N4 bias-field correction (recon-all-style intensity inhomogeneity fix)
4) Slice-wise intensity harmonization (motion-related inter-slice inconsistency)
5) Outside-noise attenuation using robust foreground mask (no segmentation model)
6) MRI robust intensity normalization (brain-focused scaling)

Default input folder:
DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40
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
    from scipy.ndimage import gaussian_filter, binary_fill_holes, label
except Exception:
    gaussian_filter = None
    binary_fill_holes = None
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


def denoise_and_clip(data: np.ndarray) -> np.ndarray:
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    lo, hi = np.percentile(vals, [0.5, 99.8])
    x = np.clip(data, lo, hi)

    return x.astype(np.float32)


def mri_edge_preserving_denoise(data: np.ndarray) -> np.ndarray:
    """MRI-oriented denoising that avoids Gaussian blurring of anatomy edges."""
    x = denoise_and_clip(data)
    if sitk is None:
        return x

    itk = sitk.GetImageFromArray(np.transpose(x, (2, 1, 0)).astype(np.float32))
    # Curvature anisotropic diffusion is a classic MRI denoise option used to
    # reduce noise while preserving strong tissue boundaries.
    den = sitk.CurvatureAnisotropicDiffusion(
        itk,
        timeStep=0.05,
        conductanceParameter=2.0,
        numberOfIterations=6,
    )
    out = sitk.GetArrayFromImage(den)
    out = np.transpose(out, (2, 1, 0))
    return out.astype(np.float32)

    return x.astype(np.float32)


def n4_bias_correct(data: np.ndarray) -> np.ndarray:
    """N4 bias correction (recon-all style NU intensity correction concept)."""
    if sitk is None:
        return data.astype(np.float32)

    arr = np.transpose(data, (2, 1, 0)).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected = corrector.Execute(img, mask)

    out = sitk.GetArrayFromImage(corrected)
    out = np.transpose(out, (2, 1, 0))
    return out.astype(np.float32)


def slice_intensity_harmonize(data: np.ndarray, target_percentile: float = 90.0) -> np.ndarray:
    """Reduce inter-slice intensity jumps often caused by motion/inconsistency."""
    out = np.zeros_like(data, dtype=np.float32)
    slice_scale = []

    for z in range(data.shape[2]):
        sl = data[:, :, z]
        pos = sl[sl > 0]
        if pos.size < 20:
            slice_scale.append(1.0)
            continue
        p = float(np.percentile(pos, target_percentile))
        slice_scale.append(p)

    ref = float(np.median(slice_scale)) if len(slice_scale) > 0 else 1.0
    ref = max(ref, 1e-6)

    for z in range(data.shape[2]):
        sl = data[:, :, z]
        s = float(slice_scale[z]) if z < len(slice_scale) else ref
        s = max(s, 1e-6)
        gain = np.clip(ref / s, 0.7, 1.4)
        out[:, :, z] = (sl * gain).astype(np.float32)

    return out


def attenuate_outside_noise(data: np.ndarray, fg_mask: np.ndarray, edge_soft_sigma: float = 1.2) -> np.ndarray:
    m = fg_mask.astype(np.float32)
    if gaussian_filter is not None and edge_soft_sigma > 0:
        m = gaussian_filter(m, sigma=edge_soft_sigma)
        m = np.clip(m, 0.0, 1.0)

    # Keep some residual context (0.15) while strongly suppressing outside noise.
    soft = 0.15 + 0.85 * m
    return (data * soft).astype(np.float32)


def robust_normalize(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    lo, hi = np.percentile(vals, [p_low, p_high])
    hi = max(hi, lo + 1e-6)

    x = np.clip(data, lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 2047.0).astype(np.float32)


def mri_robust_normalize(data: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Brain-focused normalization similar in spirit to mri_normalize."""
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


def save_step_triplet(data: np.ndarray, step_tag: str, qc_dir: str, title_prefix: str) -> None:
    save_4x4_montage(data, 2, f"{title_prefix} | {step_tag} | axial", os.path.join(qc_dir, f"{step_tag}_axial_4x4.png"))
    save_4x4_montage(data, 1, f"{title_prefix} | {step_tag} | coronal", os.path.join(qc_dir, f"{step_tag}_coronal_4x4.png"))
    save_4x4_montage(data, 0, f"{title_prefix} | {step_tag} | sagittal", os.path.join(qc_dir, f"{step_tag}_sagittal_4x4.png"))


def process_stack(name: str, input_path: str, stack_out_dir: str) -> StackReport:
    _mkdir(stack_out_dir)
    qc_dir = os.path.join(stack_out_dir, "qc")
    _mkdir(qc_dir)
    _clear_pngs(qc_dir)

    data0, img = load_nifti(input_path)
    orient = tuple(str(c) for c in nib.aff2axcodes(img.affine))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

    m0 = robust_foreground_mask(data0, low_q=20.0)
    metrics = {"step0_raw": compute_metrics(data0, m0)}
    save_step_triplet(data0, "step0_raw", qc_dir, name)

    data1 = mri_edge_preserving_denoise(data0)
    metrics["step1_mri_denoise"] = compute_metrics(data1)
    save_step_triplet(data1, "step1_mri_denoise", qc_dir, name)

    data2 = n4_bias_correct(data1)
    metrics["step2_n4_bias_correct"] = compute_metrics(data2)
    save_step_triplet(data2, "step2_n4_bias_correct", qc_dir, name)

    data3 = slice_intensity_harmonize(data2, target_percentile=90.0)
    metrics["step3_slice_harmonize"] = compute_metrics(data3)
    save_step_triplet(data3, "step3_slice_harmonize", qc_dir, name)

    fg = robust_foreground_mask(data3, low_q=20.0)
    data4 = attenuate_outside_noise(data3, fg, edge_soft_sigma=1.2)
    metrics["step4_outside_noise_atten"] = compute_metrics(data4, fg)
    save_step_triplet(data4, "step4_outside_noise_atten", qc_dir, name)

    data5 = mri_robust_normalize(data4, fg)
    metrics["step5_norm"] = compute_metrics(data5, fg)
    save_step_triplet(data5, "step5_norm", qc_dir, name)

    # Also show foreground mask overlay for inspection.
    for axis, axis_name in [(2, "axial"), (1, "coronal"), (0, "sagittal")]:
        save_overlay_montage(
            data5,
            fg.astype(np.uint8),
            axis,
            f"{name} | step5_norm + fg mask | {axis_name}",
            os.path.join(qc_dir, f"step5_norm_fg_overlay_{axis_name}.png"),
        )

    preproc_img = os.path.join(stack_out_dir, f"{name}_preproc_no_seg.nii.gz")
    fg_mask_path = os.path.join(stack_out_dir, f"{name}_fgmask_no_seg.nii.gz")
    save_nifti_like(img, data5, preproc_img, dtype=np.float32)
    save_nifti_like(img, fg.astype(np.uint8), fg_mask_path, dtype=np.uint8)

    outputs = {
        "preprocessed_image": preproc_img,
        "foreground_mask": fg_mask_path,
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
    p = argparse.ArgumentParser(description="No-segmentation stepwise preprocessing for NiftyMIC stacks")
    p.add_argument("--input-dir", type=str, default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40")
    p.add_argument("--output-dir", type=str, default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/no_seg_preproc")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _mkdir(args.output_dir)
    fixed_fig_root = "DataSRR/volunteer_xxx/figures/no_seg_preproc"
    _mkdir(fixed_fig_root)

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
        report = process_stack(name=name, input_path=path, stack_out_dir=stack_out)
        reports.append(report)

        # Mirror QC into fixed volunteer figure folder.
        src_qc = report.outputs["qc_dir"]
        dst_qc = os.path.join(fixed_fig_root, name)
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
        "fixed_figure_dir": fixed_fig_root,
        "segmentation_used": False,
        "stacks": [asdict(r) for r in reports],
        "recommended_next_steps": [
            "Use *_preproc_no_seg.nii.gz as NiftyMIC inputs and *_fgmask_no_seg.nii.gz as stack masks.",
            "Start with --intensity-correction 0 in NiftyMIC, then test 1 only if registration is stable.",
            "Use outlier rejection to reduce motion-corrupted slices in final SRR.",
            "Inspect per-step QC and retune threshold percentile in robust_foreground_mask if anatomy is clipped.",
        ],
    }

    summary_path = os.path.join(args.output_dir, "preprocess_no_seg_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Finished no-segmentation preprocessing workflow")
    print("=" * 80)
    print("Input dir:", args.input_dir)
    print("Output dir:", args.output_dir)
    print("Figure dir:", fixed_fig_root)
    print("Summary:", summary_path)
    for r in reports:
        print("-" * 80)
        print(f"{r.name}: shape={r.shape}, zooms={r.zooms_mm}, orient={r.orientation}")
        m0 = r.metrics_by_step["step0_raw"]
        m5 = r.metrics_by_step["step5_norm"]
        print(
            "  SNR95 raw->final:",
            f"{m0.snr95:.2f} -> {m5.snr95:.2f}",
            "| outside_energy raw->final:",
            f"{m0.outside_energy_fraction:.3f} -> {m5.outside_energy_fraction:.3f}",
        )
        print("  final:", r.outputs["preprocessed_image"])


if __name__ == "__main__":
    main()
