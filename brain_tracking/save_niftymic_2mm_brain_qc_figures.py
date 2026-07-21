#!/usr/bin/env python3
"""Generate brain-extracted 2mm NiftyMIC QC figures in axial/coronal/sagittal planes.

Creates 4x4 slice montages and mask overlays for:
- srr_final_2x2x2mm_3scan_brain
- srr_final_2x2x2mm_3scan_brain_corrected

Outputs are saved in:
- <output-dir>/niftymic_qc_2mm_brain
- <fixed-fig-dir>/niftymic_qc_2mm_brain
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clear_pngs(path: Path) -> None:
    if not path.exists():
        return
    for fn in path.iterdir():
        if fn.suffix.lower() == ".png":
            fn.unlink(missing_ok=True)


def _slice(data: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 2:
        sl = data[:, :, idx]
    elif axis == 1:
        sl = data[:, idx, :]
    else:
        sl = data[idx, :, :]
    return np.rot90(sl, k=1)


def save_4x4_montage(data: np.ndarray, axis: int, title: str, out_path: Path) -> None:
    n = data.shape[axis]
    ids = np.linspace(0, n - 1, min(16, n)).astype(int)

    vmin = float(np.percentile(data, 1))
    vmax = float(np.percentile(data, 99))
    vmax = max(vmax, vmin + 1e-6)

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    rows, cols = 4, 4

    for i, sid in enumerate(ids):
        r = i // cols
        c = i % cols
        ax = fig.add_axes([c / cols, 1.0 - (r + 1) / rows, 1.0 / cols, 1.0 / rows])
        ax.imshow(_slice(data, axis=axis, idx=int(sid)), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, color="white", fontsize=10, y=0.995)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def save_overlay_mid(mask: np.ndarray, data: np.ndarray, axis: int, title: str, out_path: Path) -> None:
    idx = data.shape[axis] // 2

    sl = _slice(data, axis=axis, idx=idx)
    mk = _slice(mask, axis=axis, idx=idx)

    vmin = float(np.percentile(data, 1))
    vmax = float(np.percentile(data, 99))
    vmax = max(vmax, vmin + 1e-6)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="black")
    ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
    ax.contour(mk > 0.5, levels=[0.5], colors="lime", linewidths=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color="white")

    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def make_fig_set(tag: str, image_path: Path, mask_path: Path, out_dir: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    img = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    msk = nib.load(str(mask_path)).get_fdata(dtype=np.float32)

    axes = [(2, "axial"), (1, "coronal"), (0, "sagittal")]

    for axis, axis_name in axes:
        save_4x4_montage(
            img,
            axis=axis,
            title=f"{tag} | {axis_name} | 2mm",
            out_path=out_dir / f"{tag}_{axis_name}_4x4.png",
        )
        save_overlay_mid(
            msk,
            img,
            axis=axis,
            title=f"{tag} | {axis_name} mid overlay",
            out_path=out_dir / f"{tag}_{axis_name}_mid_overlay.png",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create brain-extracted 2mm NiftyMIC QC figures")
    p.add_argument(
        "--output-dir",
        default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/reconall_v7_no_seg_outside_cleanup_exp",
    )
    p.add_argument(
        "--fixed-fig-dir",
        default="DataSRR/volunteer_xxx/figures/reconall_v7_no_seg_outside_cleanup_exp",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fixed_fig_dir = Path(args.fixed_fig_dir)

    qc_dir = output_dir / "niftymic_qc_2mm_brain"
    fixed_qc_dir = fixed_fig_dir / "niftymic_qc_2mm_brain"

    _mkdir(qc_dir)
    _mkdir(fixed_qc_dir)
    _clear_pngs(qc_dir)
    _clear_pngs(fixed_qc_dir)

    brain_img = output_dir / "srr_final_2x2x2mm_3scan_brain.nii.gz"
    brain_mask = output_dir / "srr_final_2x2x2mm_3scan_brain_mask.nii.gz"
    make_fig_set("final_2mm_3scan_brain", brain_img, brain_mask, qc_dir)

    corrected_img = output_dir / "srr_final_2x2x2mm_3scan_brain_corrected.nii.gz"
    corrected_mask = output_dir / "srr_final_2x2x2mm_3scan_brain_mask_corrected.nii.gz"
    if corrected_img.exists() and corrected_mask.exists():
        make_fig_set("final_2mm_3scan_brain_corrected", corrected_img, corrected_mask, qc_dir)

    for src in qc_dir.iterdir():
        if src.suffix.lower() == ".png":
            shutil.copy2(src, fixed_qc_dir / src.name)

    print("=" * 80)
    print("Saved brain-extracted 2mm NiftyMIC QC figures")
    print("=" * 80)
    print("QC dir:", qc_dir)
    print("Mirrored QC dir:", fixed_qc_dir)


if __name__ == "__main__":
    main()
