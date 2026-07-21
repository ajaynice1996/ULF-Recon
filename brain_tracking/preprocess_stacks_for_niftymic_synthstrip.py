#!/usr/bin/env python3
"""
Preprocess low-field orthogonal stacks for robust NiftyMIC registration.

Pipeline per stack (axial/coronal/sagittal):
1) In-plane denoise (mild Gaussian)
2) Per-slice low-frequency bias flattening
3) Robust intensity normalization to [0, 2047]
4) SynthStrip skull stripping via local synthstrip-docker wrapper
5) Mask cleanup (largest connected component, hole fill, optional dilation)
6) Save QC montages and masked final image

Default input folder:
DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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


@dataclass
class StackReport:
    name: str
    input_path: str
    shape: Tuple[int, int, int]
    zooms_mm: Tuple[float, float, float]
    orientation: Tuple[str, str, str]
    snr95_before: float
    snr95_after: float
    mask_fraction: float
    outputs: Dict[str, str]


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img


def save_nifti_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: str, dtype=np.float32) -> None:
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    out = nib.Nifti1Image(data.astype(dtype), affine=ref_img.affine, header=hdr)
    nib.save(out, out_path)


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
        low = gaussian_filter(sl, sigma=sigma_xy)
        low = np.maximum(low, 1e-6)
        out[:, :, z] = (sl / low).astype(np.float32)
    return out


def robust_normalize(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(vals, [p_low, p_high])
    hi = max(hi, lo + 1e-6)
    x = np.clip(data, lo, hi)
    x = (x - lo) / (hi - lo)
    return (x * 2047.0).astype(np.float32)


def cleanup_mask(mask: np.ndarray, dilate_iter: int = 0) -> np.ndarray:
    m = mask > 0
    if label is not None:
        cc, n_cc = label(m)
        if n_cc > 1:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            keep = sizes.argmax()
            m = cc == keep
    if binary_fill_holes is not None:
        m = binary_fill_holes(m)
    if dilate_iter > 0 and binary_dilation is not None:
        for _ in range(dilate_iter):
            m = binary_dilation(m)
    return m.astype(np.uint8)


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


def run_synthstrip(wrapper_path: str, in_nii: str, out_brain_nii: str, out_mask_nii: str, gpu: bool = False) -> None:
    cmd = [wrapper_path, "-i", in_nii, "-o", out_brain_nii, "-m", out_mask_nii]
    if gpu:
        cmd.extend(["--gpu"])
    subprocess.run(cmd, check=True)


def process_stack(
    name: str,
    input_path: str,
    stack_out_dir: str,
    wrapper_path: str,
    use_synthstrip: bool,
    mask_dilate_iter: int,
) -> StackReport:
    _mkdir(stack_out_dir)
    qc_dir = os.path.join(stack_out_dir, "qc")
    _mkdir(qc_dir)

    data0, img = load_nifti(input_path)
    orient = tuple(str(c) for c in nib.aff2axcodes(img.affine))
    zooms = tuple(float(x) for x in img.header.get_zooms()[:3])

    snr0 = estimate_snr95(data0)

    data1 = inplane_denoise(data0, sigma_xy=0.8)
    data2 = per_slice_bias_flatten(data1, sigma_xy=10.0)
    data3 = robust_normalize(data2, p_low=1.0, p_high=99.5)

    preproc_path = os.path.join(stack_out_dir, f"{name}_preproc_for_synthstrip.nii.gz")
    save_nifti_like(img, data3, preproc_path, dtype=np.float32)

    synth_brain = os.path.join(stack_out_dir, f"{name}_synthstrip_brain.nii.gz")
    synth_mask = os.path.join(stack_out_dir, f"{name}_synthstrip_mask.nii.gz")

    if use_synthstrip:
        run_synthstrip(wrapper_path, preproc_path, synth_brain, synth_mask)
        brain = nib.load(synth_brain).get_fdata(dtype=np.float32)
        mask_raw = nib.load(synth_mask).get_fdata(dtype=np.float32)
        mask_clean = cleanup_mask(mask_raw, dilate_iter=mask_dilate_iter)
    else:
        # Fallback simple mask if docker is not available.
        thr = float(np.percentile(data3[data3 > 0], 20)) if np.any(data3 > 0) else 0.0
        mask_clean = cleanup_mask((data3 > thr).astype(np.uint8), dilate_iter=mask_dilate_iter)
        brain = data3 * mask_clean

    brain_final = (data3 * mask_clean).astype(np.float32)

    mask_clean_path = os.path.join(stack_out_dir, f"{name}_brain_mask_clean.nii.gz")
    brain_final_path = os.path.join(stack_out_dir, f"{name}_brain_final.nii.gz")
    save_nifti_like(img, mask_clean.astype(np.uint8), mask_clean_path, dtype=np.uint8)
    save_nifti_like(img, brain_final, brain_final_path, dtype=np.float32)

    # QC output
    for axis, label_name in [(2, "axial"), (1, "coronal"), (0, "sagittal")]:
        save_4x4_montage(data0, axis, f"{name}: raw | {label_name}", os.path.join(qc_dir, f"step0_raw_{label_name}.png"))
        save_4x4_montage(data3, axis, f"{name}: preproc | {label_name}", os.path.join(qc_dir, f"step1_preproc_{label_name}.png"))
        save_overlay_montage(data3, mask_clean, axis, f"{name}: mask overlay | {label_name}", os.path.join(qc_dir, f"step2_mask_overlay_{label_name}.png"))
        save_4x4_montage(brain_final, axis, f"{name}: brain final | {label_name}", os.path.join(qc_dir, f"step3_brain_final_{label_name}.png"))

    snr_final = estimate_snr95(brain_final)
    outputs = {
        "preproc_for_synthstrip": preproc_path,
        "synthstrip_brain": synth_brain if use_synthstrip else "not_run",
        "synthstrip_mask": synth_mask if use_synthstrip else "not_run",
        "brain_mask_clean": mask_clean_path,
        "brain_final": brain_final_path,
        "qc_dir": qc_dir,
    }

    return StackReport(
        name=name,
        input_path=input_path,
        shape=tuple(int(x) for x in data0.shape),
        zooms_mm=zooms,
        orientation=orient,
        snr95_before=float(snr0),
        snr95_after=float(snr_final),
        mask_fraction=float(np.count_nonzero(mask_clean) / mask_clean.size),
        outputs=outputs,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess and skull-strip orthogonal stacks for NiftyMIC")
    p.add_argument("--input-dir", type=str, default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40")
    p.add_argument("--output-dir", type=str, default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/synthstrip_preproc")
    p.add_argument("--synthstrip-wrapper", type=str, default="./synthstrip-docker")
    p.add_argument("--no-synthstrip", action="store_true", help="Skip docker SynthStrip and use threshold fallback mask")
    p.add_argument("--mask-dilate-iter", type=int, default=1, help="Optional brain mask dilation iterations")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _mkdir(args.output_dir)
    fixed_fig_root = "DataSRR/volunteer_xxx/figures/synthstrip_preproc"
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
        out_dir = os.path.join(args.output_dir, name)
        r = process_stack(
            name=name,
            input_path=path,
            stack_out_dir=out_dir,
            wrapper_path=args.synthstrip_wrapper,
            use_synthstrip=not args.no_synthstrip,
            mask_dilate_iter=args.mask_dilate_iter,
        )
        reports.append(r)

        # Mirror QC into fixed volunteer_xxx figure folder.
        src_qc = r.outputs["qc_dir"]
        dst_qc = os.path.join(fixed_fig_root, name)
        _mkdir(dst_qc)
        for fn in os.listdir(src_qc):
            src = os.path.join(src_qc, fn)
            dst = os.path.join(dst_qc, fn)
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "fixed_figure_dir": fixed_fig_root,
        "synthstrip_used": not args.no_synthstrip,
        "stacks": [asdict(r) for r in reports],
        "recommended_next_steps": [
            "Use *_brain_final.nii.gz and *_brain_mask_clean.nii.gz as NiftyMIC inputs.",
            "Enable NiftyMIC stack masks to suppress phase/noise background influence.",
            "If registration drifts, reduce mask dilation or disable intensity correction in first reconstruction cycle.",
            "Reject obvious corrupted slices before final SRR when phase artifacts are severe.",
        ],
    }

    summary_path = os.path.join(args.output_dir, "preprocess_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Finished preprocess + skull-strip workflow")
    print("=" * 80)
    print("Input dir:", args.input_dir)
    print("Output dir:", args.output_dir)
    print("Figure dir:", fixed_fig_root)
    print("Summary:", summary_path)
    for r in reports:
        print("-" * 80)
        print(f"{r.name}: shape={r.shape}, zooms={r.zooms_mm}, orient={r.orientation}")
        print(f"  SNR95 before={r.snr95_before:.2f}, after={r.snr95_after:.2f}, mask_fraction={r.mask_fraction:.3f}")
        print(f"  final={r.outputs['brain_final']}")


if __name__ == "__main__":
    main()
