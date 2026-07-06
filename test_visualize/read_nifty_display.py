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
def select_folder(title="Select folder"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


# ----------------------------------
# LOAD VOLUME (nii / npy)
# ----------------------------------
def load_volume(fpath):
    if fpath.endswith((".nii", ".nii.gz")):
        img = nib.load(fpath)
        return np.abs(img.get_fdata())

    elif fpath.endswith(".npy"):
        return np.abs(np.load(fpath))

    else:
        raise ValueError("Unsupported file format")

def display_all_slices(volume, file_name, cols=4, overlap=0.001):

    """
    Display all axial (Z) slices with ZERO spacing and slight overlap.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (H, W, Z)
    cols : int
        Number of columns
    overlap : float
        Fractional overlap between images (0.0–0.1 recommended)
    """

    if volume is None or volume.ndim != 3:
        print("Invalid volume. Expected (H, W, Z).")
        return

    H, W, Z = volume.shape
    print(f"Volume shape: H={H}, W={W}, Z={Z}")
    rows = math.ceil(Z / cols)

    fig = plt.figure(figsize=(cols, rows))
    fig.patch.set_facecolor('black')

    # Effective cell size with overlap
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows

    for z in range(Z):
        r, c = divmod(z, cols)

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
# ----------------------------------
def interactive_display_and_save(
    cols=4,
    overlap=0.001
):
    source_dir = select_folder("Select INPUT folder (nii / npy)")
    if not source_dir:
        print("❌ No input folder selected")
        return

    target_dir = select_folder("Select OUTPUT folder (PNG)")
    if not target_dir:
        print("❌ No output folder selected")
        return

    os.makedirs(target_dir, exist_ok=True)

    files = sorted([
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.lower().endswith((".nii", ".nii.gz", ".npy"))
    ])

    if not files:
        print("❌ No valid files found")
        return

    print(f"🔍 Found {len(files)} files")

    for idx, fpath in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] Processing:")
        print(f"📄 {fpath}")

        try:
            volume = load_volume(fpath)
        except Exception as e:
            print(f"⚠️ Failed to load: {e}")
            continue

        basename = os.path.basename(fpath)
        name_no_ext = basename.replace(".nii.gz", "").replace(".nii", "").replace(".npy", "")
        save_png = os.path.join(target_dir, f"{name_no_ext}.png")

        # ---- DISPLAY & SAVE ----
        display_all_slices(
            volume,
            file_name=save_png,
            cols=cols,
            overlap=overlap
        )

        print(f"💾 Saved PNG → {save_png}")
        input("➡️ Press ENTER for next file...")

    print("🎉 All files processed")


# ----------------------------------
# RUN
# ----------------------------------
interactive_display_and_save(
    cols=4,
    overlap=0.001
)