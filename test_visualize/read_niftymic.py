import os
import numpy as np
import nibabel as nib
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('./data_read_code')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import os
import glob
import pydicom
from demo_read_data import read_lf_data
import json

import numpy as np
import nibabel as nib
from read_kea3d import kea3d
from kea2nifti import make_nifti
from nibabel.viewers import OrthoSlicer3D
import os
import math
import matplotlib.pyplot as plt

# ----------------------------------
# SELECT FOLDER
# ----------------------------------

def display_all_slices(
    volume,
    file_name,
    cols=4,
    overlap=0.001,
    start_slice=50,
    end_slice=53
):
    """
    Display selected axial (Z) slices with ZERO spacing and slight overlap.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (H, W, Z)
    cols : int
        Number of columns
    overlap : float
        Fractional overlap between images (0.0–0.1 recommended)
    start_slice : int
        First slice index to display
    end_slice : int or None
        Last slice index to display (inclusive). If None → till last slice.
    """

    if volume is None or volume.ndim != 3:
        print("Invalid volume. Expected (H, W, Z).")
        return

    H, W, Z = volume.shape

    if end_slice is None or end_slice >= Z:
        end_slice = Z - 1

    start_slice = max(0, start_slice)
    end_slice = min(Z - 1, end_slice)

    slice_indices = list(range(start_slice, end_slice + 1))
    num_slices = len(slice_indices)

    print(f"Volume shape: H={H}, W={W}, Z={Z}")
    print(f"Displaying slices: {start_slice} → {end_slice}")

    rows = math.ceil(num_slices / cols)

    fig = plt.figure(figsize=(cols, rows))
    fig.patch.set_facecolor('black')

    cell_w = 1.0 / cols
    cell_h = 1.0 / rows

    for i, z in enumerate(slice_indices):
        r, c = divmod(i, cols)

        left = c * cell_w - overlap * c
        bottom = 1 - (r + 1) * cell_h + overlap * r

        ax = fig.add_axes([
            left,
            bottom,
            cell_w + overlap,
            cell_h + overlap
        ])

        ax.imshow(np.rot90(np.abs(volume[:, :, z]), k=1), cmap='gray')
        ax.axis('off')

    print(f"Saving figure as: {file_name}")
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.show()

# ----------------------------------
# MAIN INTERACTIVE TOOL

# make main and call the above function
if __name__ == "__main__":

    # ---- Input NIfTI file ----
    nii_path = "srr_1cycle_2mm_Huber_test2.nii.gz"

    # ---- Load NIfTI ----
    img = nib.load(nii_path)
    volume = np.abs(img.get_fdata())

    print("Loaded volume shape:", volume.shape)

    # ---- Output PNG ----
    output_png = "srr_1cycle_2mm_Huber_test2_mask.png"

    # ---- Display & Save ----
    display_all_slices(
        volume,
        file_name=output_png,
        cols=4,
        overlap=0.001
    )


    




