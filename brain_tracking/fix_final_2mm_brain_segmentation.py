#!/usr/bin/env python3
"""Fix brain segmentation for final 2mm SRR when SynthStrip under-segments.

Strategy:
- Compare SynthStrip mask and NiftyMIC reconstruction mask.
- If SynthStrip is much smaller, fall back to cleaned NiftyMIC mask.
- Save corrected mask + brain image and a small report JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from scipy.ndimage import binary_fill_holes, binary_dilation, label
except Exception:
    binary_fill_holes = None
    binary_dilation = None
    label = None


def _load_bool(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata() > 0


def _cleanup_mask(mask: np.ndarray, dilate_iter: int = 0) -> np.ndarray:
    m = mask.astype(bool)

    if label is not None:
        cc, n_cc = label(m)
        if n_cc > 1:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            m = cc == int(np.argmax(sizes))

    if binary_fill_holes is not None:
        m = binary_fill_holes(m)

    if dilate_iter > 0 and binary_dilation is not None:
        for _ in range(dilate_iter):
            m = binary_dilation(m)

    return m.astype(bool)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix final 2mm brain segmentation")
    p.add_argument(
        "--output-dir",
        default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/reconall_v7_no_seg_outside_cleanup_exp",
    )
    p.add_argument("--synthstrip-min-relative", type=float, default=0.6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    img_path = out_dir / "srr_final_2x2x2mm_3scan.nii.gz"
    synth_mask_path = out_dir / "srr_final_2x2x2mm_3scan_brain_mask.nii.gz"
    nifty_mask_path = out_dir / "srr_final_2x2x2mm_3scan_mask.nii.gz"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not synth_mask_path.exists():
        raise FileNotFoundError(f"Missing SynthStrip mask: {synth_mask_path}")
    if not nifty_mask_path.exists():
        raise FileNotFoundError(f"Missing NiftyMIC mask: {nifty_mask_path}")

    img_nii = nib.load(str(img_path))
    img = img_nii.get_fdata(dtype=np.float32)

    synth_mask = _cleanup_mask(_load_bool(synth_mask_path), dilate_iter=0)
    nifty_mask = _cleanup_mask(_load_bool(nifty_mask_path), dilate_iter=1)

    synth_vox = int(np.count_nonzero(synth_mask))
    nifty_vox = int(np.count_nonzero(nifty_mask))
    rel = float(synth_vox / (nifty_vox + 1e-8))

    inter = np.count_nonzero(synth_mask & nifty_mask)
    dice = float((2.0 * inter) / (synth_vox + nifty_vox + 1e-8))

    use_nifty = rel < float(args.synthstrip_min_relative)
    chosen_name = "niftymic_mask_fallback" if use_nifty else "synthstrip"
    chosen_mask = nifty_mask if use_nifty else synth_mask

    brain = (img * chosen_mask.astype(np.float32)).astype(np.float32)

    corrected_mask_path = out_dir / "srr_final_2x2x2mm_3scan_brain_mask_corrected.nii.gz"
    corrected_brain_path = out_dir / "srr_final_2x2x2mm_3scan_brain_corrected.nii.gz"
    report_path = out_dir / "srr_final_2x2x2mm_3scan_brain_segmentation_fix_report.json"

    nib.save(nib.Nifti1Image(chosen_mask.astype(np.uint8), img_nii.affine, img_nii.header), str(corrected_mask_path))
    nib.save(nib.Nifti1Image(brain, img_nii.affine, img_nii.header), str(corrected_brain_path))

    report = {
        "image": str(img_path),
        "synthstrip_mask": str(synth_mask_path),
        "niftymic_mask": str(nifty_mask_path),
        "chosen_method": chosen_name,
        "synth_voxels": synth_vox,
        "nifty_voxels": nifty_vox,
        "synth_over_nifty_ratio": rel,
        "dice_synth_vs_nifty": dice,
        "corrected_mask": str(corrected_mask_path),
        "corrected_brain": str(corrected_brain_path),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("Fixed final 2mm brain segmentation")
    print("=" * 80)
    print("Chosen method:", chosen_name)
    print("Synth voxels:", synth_vox)
    print("Nifty voxels:", nifty_vox)
    print("Synth/Nifty ratio:", rel)
    print("Dice:", dice)
    print("Corrected mask:", corrected_mask_path)
    print("Corrected brain:", corrected_brain_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
