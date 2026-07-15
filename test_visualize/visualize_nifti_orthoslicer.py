#!/usr/bin/env python3

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib

from nibabel.viewers import OrthoSlicer3D


# ---------------------------------------------------------
# Choose display intensity window
# ---------------------------------------------------------

def choose_clim(data, mode="percentile"):

    finite = data[np.isfinite(data)]

    if mode == "percentile":

        p1 = np.percentile(finite, 1)
        p99 = np.percentile(finite, 99)

        return p1, p99

    elif mode == "minmax":

        return finite.min(), finite.max()

    else:
        raise ValueError(
            "Unknown clim mode"
        )


# ---------------------------------------------------------
# OrthoSlicer viewer
# ---------------------------------------------------------

def browse_with_orthoslicer(
        img,
        title="NIfTI",
        clim_mode="percentile"):

    data = img.get_fdata().astype(np.float32)

    clim = choose_clim(
        data,
        mode=clim_mode
    )

    print("")
    print("="*80)
    print(f"Opening OrthoSlicer3D: {title}")
    print("="*80)

    print("backend:",
          matplotlib.get_backend())

    print("shape:",
          data.shape)

    print("dtype:",
          data.dtype)

    print("orientation:",
          nib.aff2axcodes(img.affine))

    print("zooms:",
          img.header.get_zooms()[:3])

    print("clim:",
          clim)


    try:

        slicer = OrthoSlicer3D(data)

        slicer.clim = clim

        slicer.show()


    except Exception as e:

        print("")
        print("WARNING: OrthoSlicer3D failed.")
        print("Reason:")
        print(e)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    nii_file = "DataSRR/volunteer_xxx/10mm/npy/synthstrip_output/srr_alpha001_out1_ic0_cycle2_iso2_HuberL2_skullstrip.nii.gz"
    if not os.path.exists(nii_file):

        raise FileNotFoundError(
            nii_file
        )

    print("")
    print("Loading:")
    print(nii_file)

    img = nib.load(nii_file)

    title = os.path.basename(
        nii_file
    )

    browse_with_orthoslicer(
        img,
        title=title,
        clim_mode="percentile"
    )