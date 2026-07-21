#!/usr/bin/env python3
"""
Run full v7 no-seg pipeline and NiftyMIC reconstruction with 3 scans + 3 masks.

Steps:
1) Run preprocessing script (v7 outside-cleanup experiment).
2) Read generated summary and collect final image+mask for axial/coronal/sagittal.
3) Save/standardize three final NIfTI inputs and three final masks for NiftyMIC.
4) Run geometry-check NiftyMIC reconstruction.
5) Run final NiftyMIC reconstruction at 2x2x2 mm.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], log_path: Path | None = None) -> None:
    print("\n$ " + " ".join(cmd))
    if log_path is None:
        subprocess.run(cmd, check=True)
        return

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def _load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_inputs_from_summary(summary: Dict, output_root: Path) -> Dict[str, Path]:
    stacks = {entry["name"]: entry for entry in summary["stacks"]}
    required = ["axial", "coronal", "sagittal"]
    missing = [k for k in required if k not in stacks]
    if missing:
        raise ValueError(f"Missing stacks in summary: {missing}")

    niftymic_input_dir = output_root / "niftymic_3scan_inputs"
    _mkdir(niftymic_input_dir)

    paths: Dict[str, Path] = {}
    for name in required:
        out = stacks[name]["outputs"]
        src_img = Path(out["preprocessed_image"])
        src_mask = Path(out["foreground_mask"])
        if not src_img.exists() or not src_mask.exists():
            raise FileNotFoundError(f"Missing final image/mask for {name}: {src_img} | {src_mask}")

        dst_img = niftymic_input_dir / f"{name}_final_v7_no_seg.nii.gz"
        dst_mask = niftymic_input_dir / f"{name}_final_v7_no_seg_mask.nii.gz"
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)

        paths[f"{name}_img"] = dst_img
        paths[f"{name}_mask"] = dst_mask

    return paths


def _build_niftymic_cmd(
    docker_image: str,
    paths: Dict[str, Path],
    output_nii: Path,
    alpha: float,
    outlier_rejection: int,
    intensity_correction: int,
    two_step_cycles: int,
    isotropic_resolution: int,
    reconstruction_type: str,
    verbose: int,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        "$PWD:$PWD",
        "-w",
        "$PWD",
        docker_image,
        "niftymic_reconstruct_volume",
        "--filenames",
        str(paths["axial_img"]),
        str(paths["coronal_img"]),
        str(paths["sagittal_img"]),
        "--filenames-masks",
        str(paths["axial_mask"]),
        str(paths["coronal_mask"]),
        str(paths["sagittal_mask"]),
        "--alpha",
        str(alpha),
        "--outlier-rejection",
        str(outlier_rejection),
        "--intensity-correction",
        str(intensity_correction),
        "--two-step-cycles",
        str(two_step_cycles),
        "--isotropic-resolution",
        str(isotropic_resolution),
        "--output",
        str(output_nii),
        "--verbose",
        str(verbose),
        "--reconstruction-type",
        reconstruction_type,
    ]


def _run_in_shell(cmd_list: list[str], log_path: Path) -> None:
    cmd = " ".join(cmd_list)
    _run(["zsh", "-lc", cmd], log_path=log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full v7 no-seg + NiftyMIC pipeline")
    p.add_argument("--python-bin", default="/Users/ashar126/Documents/Development/Work/ULF-Recon/.venv-1/bin/python")
    p.add_argument("--preprocess-script", default="DataSRR/volunteer_xxx/preprocess_stacks_for_niftymic_no_segmentation_reconall_v7_outside_cleanup_exp.py")
    p.add_argument("--input-dir", default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40")
    p.add_argument("--output-dir", default="DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/reconall_v7_no_seg_outside_cleanup_exp")
    p.add_argument("--fixed-fig-dir", default="DataSRR/volunteer_xxx/figures/reconall_v7_no_seg_outside_cleanup_exp")
    p.add_argument("--docker-image", default="renbem/niftymic")
    p.add_argument("--skip-preprocess", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    _mkdir(output_dir)
    logs_dir = output_dir / "niftymic_logs"
    _mkdir(logs_dir)

    if not args.skip_preprocess:
        preproc_log = logs_dir / "preprocess_v7.log"
        _run(
            [
                args.python_bin,
                args.preprocess_script,
                "--input-dir",
                args.input_dir,
                "--output-dir",
                args.output_dir,
                "--fixed-fig-dir",
                args.fixed_fig_dir,
            ],
            log_path=preproc_log,
        )

    summary_path = output_dir / "preprocess_reconall_v7_no_seg_outside_cleanup_exp_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    summary = _load_summary(summary_path)
    paths = _copy_inputs_from_summary(summary, output_dir)

    geometry_output = output_dir / "srr_geometry_check_2x2x2mm_3scan.nii.gz"
    final_output = output_dir / "srr_final_2x2x2mm_3scan.nii.gz"

    geometry_cmd = _build_niftymic_cmd(
        docker_image=args.docker_image,
        paths=paths,
        output_nii=geometry_output,
        alpha=0.02,
        outlier_rejection=0,
        intensity_correction=0,
        two_step_cycles=0,
        isotropic_resolution=2,
        reconstruction_type="HuberL2",
        verbose=1,
    )

    final_cmd = _build_niftymic_cmd(
        docker_image=args.docker_image,
        paths=paths,
        output_nii=final_output,
        alpha=0.03,
        outlier_rejection=1,
        intensity_correction=0,
        two_step_cycles=2,
        isotropic_resolution=2,
        reconstruction_type="HuberL2",
        verbose=1,
    )

    _run_in_shell(geometry_cmd, logs_dir / "niftymic_geometry_check.log")
    _run_in_shell(final_cmd, logs_dir / "niftymic_final_2mm.log")

    print("\n" + "=" * 80)
    print("Full v7 no-seg + NiftyMIC pipeline complete")
    print("=" * 80)
    print("Summary:", summary_path)
    print("Geometry check output:", geometry_output)
    print("Final 2x2x2 output:", final_output)
    print("Input images/masks dir:", output_dir / "niftymic_3scan_inputs")
    print("Logs dir:", logs_dir)


if __name__ == "__main__":
    main()
