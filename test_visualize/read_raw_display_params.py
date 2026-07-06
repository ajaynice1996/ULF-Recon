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

def read_lf_data(
    data_folder='Ajay_training/Ajay_training_01',
    output_folder='Ajay_training/Output_nifti',
    subject="",
    sub_folder='3DTSE/3',
    file_name='lf_mri.nii.gz'
):
    try:
        # print(f"Data folder: {data_folder}")
        # print(f"Output folder: {output_folder}")
        # print(f"Subject: {subject}")
        # print(f"Sub folder: {sub_folder}")
        # print(f"File name: {file_name}")

        subject_folder = os.path.join(output_folder, subject)
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)

        # Include subfolder name in the output filename for differentiation
        filename = file_name
        fname_nii = os.path.join(subject_folder, filename)
        print(f"Output NIfTI file will be saved as: {fname_nii}")

        sample_data = kea3d(data_folder=data_folder, sub_folder=sub_folder)
        kspace = sample_data.kspace_gauss_filter
        im = np.abs(np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kspace)))))

        if im is None:
            print("No data found in the specified folder.")
            return None
        
        # s = OrthoSlicer3D(np.abs(im))
        # s.clim = [0, np.abs(1.5 * np.max(np.abs(im)))]
        # s.cmap = 'gray'
        # s.show()

        # print(np.max(np.abs(im)))
        # print("Min value:", np.min(np.abs(im)))
        # print("Data type of np.abs(im):", np.abs(im).dtype)
        # print("Shape of im:", im.shape)
        
        # Make nifti in case of need for further inputs to other software 
        make_nifti(
            im,
            fname=fname_nii,
            mask=False,
            res=[sample_data.res_dim1, sample_data.res_dim2, sample_data.res_dim3],
            dim_info=[0, 1, 2]
        )

        if im is None:
            sample_data = []

        return im, sample_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def display_all_views(volume, cols=10):
    """
    Display ALL slices for axial, sagittal, and coronal views
    in a single figure with zero spacing.

    Layout:
    - Top    : All axial slices
    - Middle : All sagittal slices
    - Bottom : All coronal slices

    Parameters
    ----------
    volume : np.ndarray
        3D volume (H, W, Z)
    cols : int
        Number of columns per block
    """

    if volume is None or volume.ndim != 3:
        print("Invalid volume. Expected (H, W, Z).")
        return

    H, W, Z = volume.shape

    # Number of slices per view
    n_axial = Z
    n_sagittal = H
    n_coronal = W

    # Rows needed per block
    r_axial = math.ceil(n_axial / cols)
    r_sagittal = math.ceil(n_sagittal / cols)
    r_coronal = math.ceil(n_coronal / cols)

    total_rows = r_axial + r_sagittal + r_coronal

    fig = plt.figure(figsize=(cols, total_rows))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    slice_id = 0

    # -------- AXIAL --------
    for i in range(n_axial):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(total_rows, cols, slice_id + c + 1 + r * cols)
        ax.imshow(np.abs(volume[:, :, i]), cmap='gray')
        ax.axis('off')
    slice_id += r_axial * cols

    # -------- SAGITTAL --------
    for i in range(n_sagittal):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(total_rows, cols, slice_id + c + 1 + r * cols)
        ax.imshow(np.abs(volume[i, :, :]), cmap='gray')
        ax.axis('off')
    slice_id += r_sagittal * cols

    # -------- CORONAL --------
    for i in range(n_coronal):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(total_rows, cols, slice_id + c + 1 + r * cols)
        ax.imshow(np.abs(volume[:, i, :]), cmap='gray')
        ax.axis('off')

    plt.show()

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

# function to display using ortho slicer
def display_volume_ortho(volume):
    """
    Display a 3D volume using the OrthoSlicer3D from nibabel.
    """
    if volume is None:
        print("No volume data to display.")
        return

    s = OrthoSlicer3D(volume)
    s.clim = [0, np.abs(1.5 * np.max(np.abs(volume)))]
    s.cmap = 'gray'
    s.show()


def read_par_file(file_path):
    """Parses PAR file and computes Matrix Size and Voxel Resolution."""
    params = {}
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("."):
                continue
            
            separator = ":" if ":" in line else "=" if "=" in line else None
            if separator:
                key, value = line.split(separator, 1)
                # Clean value of quotes and extra spaces
                params[key.strip()] = value.strip().replace('"', '')

    # --- NEW: Compute Derived Parameters ---
    try:
        # 1. Extract raw values for computation
        nr_pnts = int(params.get('nrPnts', 0))
        n_p1 = int(params.get('nPhase1', 0))
        n_p2 = int(params.get('nPhase2', 0))
        
        fov_r = float(params.get('FOVread', 0))
        fov_p1 = float(params.get('FOVphase1', 0))
        fov_p2 = float(params.get('FOVphase2', 0))

        # 2. Compute Matrix Size (Read x Phase1 x Phase2)
        params["Matrix Size"] = f"{nr_pnts} x {n_p1} x {n_p2}"

        # 3. Compute Voxel Resolution (FOV / Matrix)
        # We use a check to avoid division by zero
        res_r = fov_r / nr_pnts if nr_pnts > 0 else 0
        res_p1 = fov_p1 / n_p1 if n_p1 > 0 else 0
        res_p2 = fov_p2 / n_p2 if n_p2 > 0 else 0
        
        params["Voxel Resolution"] = f"{res_r:.2f} x {res_p1:.2f} x {res_p2:.2f} mm³"
        
    except (ValueError, ZeroDivisionError):
        params["Matrix Size"] = "Error computing"
        params["Voxel Resolution"] = "Error computing"

    return params

def save_params_to_png(params, output_filename="mri_panel.png"):
    """Displays selected keys in a clean PNG panel with a separated title."""
    keys_of_interest = [
        "experiment", "plane", "FOVphase1", "FOVphase2", "FOVread",
        "nPhase1", "nPhase2", "nrPnts", "repTime", "echoTime","rxGain",
        "etLength", "dwellTime", "nrScans", "bandwidth", "kTraject", "acqTime",
        "pulseLength", "Matrix Size", "Voxel Resolution"
    ]

    table_data = []
    for k in keys_of_interest:
        val = params.get(k, "N/A")
        table_data.append([k, val])

    # 1. Create figure
    fig, ax = plt.subplots(figsize=(8, 7)) # Increased height to fit more rows comfortably
    ax.axis('off')

    # 2. Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        loc='upper center', # Anchor the table to the top of its allowed area
        cellLoc='left',
        colWidths=[0.4, 0.5]
    )

    # 3. Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8) 

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f9f9f9')
            
            # Optional: Color highlight for computed params (the last two)
            if row in [len(keys_of_interest)-1, len(keys_of_interest)]:
                cell.set_facecolor('#e8f4f8') # Light highlight for Matrix/Voxel

    # 4. Separate Title logic
    # Use y=0.98 to push title to the very top edge
    plt.suptitle("ACQUISITION SUMMARY", fontsize=14, weight='bold', color='Blue', y=0.98)

    # Use tight_layout rect to prevent the table from entering the top 10% of the figure
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 5. Save
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Panel saved successfully as: {output_filename}")

# --- Execution ---
# --- Example of how to call this after your parsing logic ---

if __name__ == "__main__":

    for i in ['1', '2', '3','4','5', '6','7','8', '9', '10','11', '12', '13', '14', '15']:
        data_folder = f'Training_DATA/Ajay_training/Ajay_training_06'
        # filename as folder name Ajay_training_06

        try:
            # Automatically generate filename based on folder and index
            filename = f"{os.path.basename(data_folder)}_{i}.nii.gz"
            print(f"Generated filename: {filename}")

            result = read_lf_data(
            data_folder=data_folder,
            output_folder='Scanning_mw/Outputs/nii',
            subject="aj_training",
            sub_folder=f'3DTSE/{i}',
            file_name=filename
            )

            # read_lf_data returns (im, sample_data) or (None, None) on failure
            if not result or result[0] is None:
                print(f"No image returned for {filename}, skipping.")
                continue

            im = np.abs(np.array(result[0]))

        except Exception as e:
            print(f"Exception while processing {filename}: {e}")
            continue
        filename_no_ext = filename.replace('.nii.gz', '')
        # centralized save root so it can be updated in one place
        SAVE_ROOT = 'Scanning_mw/Outputs'

        # input folder containing acqu.par
        sub_folder = f'3DTSE/{i}'
        input_path = os.path.join(data_folder, sub_folder)
        print(f"Reading data from: {input_path}")

        par_path = os.path.join(input_path, "acqu.par")
        mri_data = read_par_file(par_path)

        # Extract relevant parameters

        images_dir = os.path.join(SAVE_ROOT, 'pngs')

        os.makedirs(images_dir, exist_ok=True)

        save_image = os.path.join(images_dir, f'{filename_no_ext}.png')
        display_all_slices(im, file_name=save_image, cols=4, overlap=0.001)

        print(mri_data)
        if mri_data:
            save_params_as_png = os.path.join(images_dir, f'{filename_no_ext}_params.png')
            save_params_to_png(mri_data, save_params_as_png)