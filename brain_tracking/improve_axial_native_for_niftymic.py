#!/usr/bin/env python3
"""
Improve low-field axial native MRI for better multi-stack registration in NiftyMIC.

What this script does
---------------------
1) Inspects scan metadata and basic quality metrics.
2) Identifies likely problems for NiftyMIC registration.
3) Applies stepwise preprocessing improvements.
4) Saves QC visualizations after each step in 4x4 zero-gap montages for
   axial/coronal/sagittal views.
5) Saves improved NIfTI and a JSON report.

Default input:
DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/axial_native.nii.gz
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


@dataclass
class ScanInfo:
    shape: Tuple[int, int, int]
    zooms_mm: Tuple[float, float, float]
    orient_codes: Tuple[str, str, str]
    dtype: str
    intensity_percentiles: Dict[str, float]
    nonzero_fraction: float
    snr95: float
    slice_mean_std: float


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img


def inspect_scan(data: np.ndarray, img: nib.Nifti1Image) -> ScanInfo:
    vals = data[np.isfinite(data)]
    q_values = np.percentile(vals, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    q_names = ["p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p100"]
    q_dict = {k: float(v) for k, v in zip(q_names, q_values)}

    shape = tuple(int(x) for x in data.shape)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    orient = tuple(str(c) for c in nib.aff2axcodes(img.affine))

    nz_frac = float(np.count_nonzero(data > 0) / data.size)

    nx, ny, _nz = shape
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
    noise_mu = float(np.median(corners))
    noise_sigma = float(np.std(corners))
    snr95 = float((q_dict["p95"] - noise_mu) / (noise_sigma + 1e-6))

    z_profile = data.mean(axis=(0, 1))
    z_profile_std = float(np.std(z_profile))

    return ScanInfo(
        shape=shape,
        zooms_mm=zooms,
        orient_codes=orient,
        dtype=str(data.dtype),
        intensity_percentiles=q_dict,
        nonzero_fraction=nz_frac,
        snr95=snr95,
        slice_mean_std=z_profile_std,
    )


def identify_problems(info: ScanInfo) -> List[str]:
    issues: List[str] = []
    x, y, z = info.zooms_mm
    anisotropy = max(info.zooms_mm) / max(min(info.zooms_mm), 1e-6)

    if anisotropy >= 3.0:
        issues.append(
            f"Strong anisotropy detected ({x:.1f}x{y:.1f}x{z:.1f} mm). Thick-slice axis can reduce through-plane detail and registration stability."
        )

    if info.nonzero_fraction < 0.7:
        issues.append(
            "Large background/zero regions present; this can bias similarity metrics during stack-to-stack registration."
        )

    if info.snr95 < 50:
        issues.append(
            f"Moderate low-field noise profile (SNR95 ~ {info.snr95:.1f}); denoising and robust scaling are recommended."
        )

    if info.intensity_percentiles["p1"] == 0.0 and info.intensity_percentiles["p5"] == 0.0:
        issues.append(
            "Intensity distribution is strongly zero-inflated at low percentiles; background masking/cropping will help."
        )

    if not issues:
        issues.append("No major issues detected by simple heuristics; proceed with light normalization only.")

    return issues


def robust_foreground_mask(data: np.ndarray, low_q: float = 20.0) -> np.ndarray:
    positive = data[data > 0]
    if positive.size == 0:
        return np.zeros_like(data, dtype=bool)
    thr = float(np.percentile(positive, low_q))
    return data > thr


def crop_to_mask(data: np.ndarray, mask: np.ndarray, margin: int = 3) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    idx = np.where(mask)
    if idx[0].size == 0:
        full = (slice(0, data.shape[0]), slice(0, data.shape[1]), slice(0, data.shape[2]))
        return data.copy(), full

    x0, x1 = int(idx[0].min()), int(idx[0].max())
    y0, y1 = int(idx[1].min()), int(idx[1].max())
    z0, z1 = int(idx[2].min()), int(idx[2].max())

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    z0 = max(0, z0 - margin)
    x1 = min(data.shape[0] - 1, x1 + margin)
    y1 = min(data.shape[1] - 1, y1 + margin)
    z1 = min(data.shape[2] - 1, z1 + margin)

    slc = (slice(x0, x1 + 1), slice(y0, y1 + 1), slice(z0, z1 + 1))
    return data[slc].copy(), slc


def inplane_denoise(data: np.ndarray, sigma_xy: float = 0.8) -> np.ndarray:
    if gaussian_filter is None:
        return data
    return gaussian_filter(data, sigma=(sigma_xy, sigma_xy, 0.0))


def per_slice_bias_flatten(data: np.ndarray, sigma_xy: float = 10.0) -> np.ndarray:
    if gaussian_filter is None:
        return data
    out = np.zeros_like(data, dtype=np.float32)
    for z in range(data.shape[2]):
        sl = data[:, :, z]
        blur = gaussian_filter(sl, sigma=sigma_xy)
        blur = np.maximum(blur, 1e-6)
        corrected = sl / blur
        out[:, :, z] = corrected.astype(np.float32)
    return out


def robust_intensity_normalize(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    vals = data[np.isfinite(data)]
    lo, hi = np.percentile(vals, [p_low, p_high])
    hi = max(hi, lo + 1e-6)
    x = np.clip(data, lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 2047.0).astype(np.float32)


def _plane_slice(data: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 2:  # axial
        sl = data[:, :, idx]
    elif axis == 1:  # coronal
        sl = data[:, idx, :]
    else:  # sagittal axis=0
        sl = data[idx, :, :]
    return np.rot90(np.asarray(sl), k=1)


def save_4x4_plane_montage(
    data: np.ndarray,
    axis: int,
    title: str,
    out_path: str,
    n_show: int = 16,
) -> None:
    n_slices = data.shape[axis]
    if n_slices <= 0:
        return

    if n_slices >= n_show:
        ids = np.linspace(0, n_slices - 1, n_show).astype(int)
    else:
        ids = np.arange(n_slices, dtype=int)

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    cols = 4
    rows = 4
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows

    vmin = float(np.percentile(data, 1))
    vmax = float(np.percentile(data, 99))
    vmax = max(vmax, vmin + 1e-6)

    for i in range(min(len(ids), 16)):
        r = i // cols
        c = i % cols
        left = c * cell_w
        bottom = 1.0 - (r + 1) * cell_h
        ax = fig.add_axes([left, bottom, cell_w, cell_h])
        ax.imshow(_plane_slice(data, axis=axis, idx=int(ids[i])), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, color="white", fontsize=10, y=0.995)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def save_step_visuals(data: np.ndarray, step_name: str, out_dir: str) -> None:
    _safe_makedirs(out_dir)
    save_4x4_plane_montage(
        data=data,
        axis=2,
        title=f"{step_name} | axial (axis=2)",
        out_path=os.path.join(out_dir, f"{step_name}_axial_4x4.png"),
    )
    save_4x4_plane_montage(
        data=data,
        axis=1,
        title=f"{step_name} | coronal (axis=1)",
        out_path=os.path.join(out_dir, f"{step_name}_coronal_4x4.png"),
    )
    save_4x4_plane_montage(
        data=data,
        axis=0,
        title=f"{step_name} | sagittal (axis=0)",
        out_path=os.path.join(out_dir, f"{step_name}_sagittal_4x4.png"),
    )


def save_nifti_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: str) -> None:
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    out = nib.Nifti1Image(data.astype(np.float32), affine=ref_img.affine, header=hdr)
    nib.save(out, out_path)


def run_pipeline(input_path: str, output_dir: str) -> None:
    _safe_makedirs(output_dir)
    qc_dir = os.path.join(output_dir, "qc_steps")
    _safe_makedirs(qc_dir)
    volunteer_root = os.path.dirname(os.path.abspath(__file__))
    volunteer_fig_dir = os.path.join(volunteer_root, "figures", "axial_native_improve_qc")
    _safe_makedirs(volunteer_fig_dir)

    data0, img = load_nifti(input_path)
    info = inspect_scan(data0, img)
    issues = identify_problems(info)

    # Step 0: raw
    save_step_visuals(data0, "step0_raw", qc_dir)
    save_step_visuals(data0, "step0_raw", volunteer_fig_dir)

    # Step 1: foreground mask + crop
    mask = robust_foreground_mask(data0, low_q=20.0)
    data1, slc = crop_to_mask(data0, mask, margin=3)
    save_step_visuals(data1, "step1_crop", qc_dir)
    save_step_visuals(data1, "step1_crop", volunteer_fig_dir)

    # Step 2: in-plane denoise
    data2 = inplane_denoise(data1, sigma_xy=0.8)
    save_step_visuals(data2, "step2_denoise", qc_dir)
    save_step_visuals(data2, "step2_denoise", volunteer_fig_dir)

    # Step 3: per-slice bias flatten
    data3 = per_slice_bias_flatten(data2, sigma_xy=10.0)
    save_step_visuals(data3, "step3_bias_flatten", qc_dir)
    save_step_visuals(data3, "step3_bias_flatten", volunteer_fig_dir)

    # Step 4: robust normalization to NIfTI-friendly range
    data4 = robust_intensity_normalize(data3, p_low=1.0, p_high=99.5)
    save_step_visuals(data4, "step4_norm", qc_dir)
    save_step_visuals(data4, "step4_norm", volunteer_fig_dir)

    improved_path = os.path.join(output_dir, "axial_native_improved_for_niftymic.nii.gz")
    save_nifti_like(img.slicer[slc], data4, improved_path)

    report = {
        "input_path": input_path,
        "output_nifti": improved_path,
        "qc_dir": qc_dir,
        "volunteer_fig_dir": volunteer_fig_dir,
        "scan_info": {
            "shape": info.shape,
            "zooms_mm": info.zooms_mm,
            "orient_codes": info.orient_codes,
            "dtype": info.dtype,
            "nonzero_fraction": info.nonzero_fraction,
            "snr95": info.snr95,
            "slice_mean_std": info.slice_mean_std,
            "intensity_percentiles": info.intensity_percentiles,
        },
        "identified_problems": issues,
        "steps": [
            "step0_raw: baseline visual QC",
            "step1_crop: remove excess background using robust foreground mask",
            "step2_denoise: mild in-plane Gaussian denoising",
            "step3_bias_flatten: per-slice low-frequency intensity flattening",
            "step4_norm: robust clipping and normalization to [0, 2047]",
        ],
        "recommendations_for_niftymic": [
            "Apply the same preprocessing logic to coronal and sagittal stacks for intensity consistency.",
            "Keep correct slice spacing metadata (e.g., 2x2x10 mm) before reconstruction.",
            "Use stack masks to reduce background-driven registration errors.",
            "Start with rigid stack-to-stack registration and inspect outlier slices before SRR.",
            "If motion remains high, reject severe outlier slices before final NiftyMIC reconstruction.",
        ],
    }

    report_path = os.path.join(output_dir, "axial_native_improvement_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("Axial scan inspection")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Shape: {info.shape}")
    print(f"Voxel size (mm): {info.zooms_mm}")
    print(f"Orientation: {info.orient_codes}")
    print(f"Nonzero fraction: {info.nonzero_fraction:.3f}")
    print(f"SNR95: {info.snr95:.2f}")
    print("Identified problems:")
    for i, issue in enumerate(issues, start=1):
        print(f"  {i}. {issue}")
    print("Saved improved NIfTI:", improved_path)
    print("Saved QC folder:", qc_dir)
    print("Saved volunteer figure folder:", volunteer_fig_dir)
    print("Saved report:", report_path)


def parse_args() -> argparse.Namespace:
    default_input = "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/axial_native.nii.gz"
    default_output = "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/axial_improve_qc"

    p = argparse.ArgumentParser(description="Improve low-field axial native NIfTI for NiftyMIC registration")
    p.add_argument("--input", type=str, default=default_input, help="Path to input axial_native NIfTI")
    p.add_argument("--output-dir", type=str, default=default_output, help="Output directory for improved scan and QC")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.input, args.output_dir)


if __name__ == "__main__":
    main()
