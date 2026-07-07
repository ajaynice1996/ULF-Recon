# Read raw data from specified folder and display/save as NIfTI
# Roll and circshift save
# save to NIfTI format
# Then do NiftyMIC and ZSSR reconstructions
# Freesurfer processing

import sys
sys.path.append('./data_read_code')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import os
import glob
import pydicom
from kea2nifti import make_nifti
# import re
from tensorflow.keras.models import load_model
from nibabel.viewers import OrthoSlicer3D
import cv2

from demo_read_data import read_lf_data
import json
from scipy import ndimage
visible = True  # Set to True to visualize images

def resample_image(im, current_spacing, target_spacing):
    # Calculate the zoom factors for each dimension
    zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    # Resample the image using zoom
    resampled_im = ndimage.zoom(im, zoom_factors, order=1)
    return resampled_im

# ============================================================
# 4. Visualization Utility
# ============================================================

def visualize_volume(volume, title="Volume", rows=3, cmap='gray'):
    """
    Visualize all slices of a 3D volume in multiple rows.
    """
    num_slices = volume.shape[2]
    cols = int(np.ceil(num_slices / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = axes.flatten()

    for i in range(num_slices):
        axes[i].imshow(volume[:, :, i], cmap=cmap)
        axes[i].set_title(f'{title} - Slice {i+1}', fontsize=8)
        axes[i].axis('off')

    for j in range(num_slices, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def visualize_volume_slices(volume, title="Volume", n_cols=4, cmap="gray",
                     vmin=None, vmax=None):
    """
    Visualize all slices of a 3D volume.

    Parameters
    ----------
    volume : ndarray
        3D volume of shape (H, W, Z)
    title : str
        Figure title
    n_cols : int
        Number of columns (default = 4)
    cmap : str
        Colormap
    vmin, vmax : float, optional
        Display intensity limits. If None, matplotlib auto-scales.
    """
    n_slices = volume.shape[2]
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2 * n_rows),
        squeeze=False
    )

    axes = axes.ravel()

    for i in range(n_slices):
        axes[i].imshow(volume[:, :, i], cmap=cmap,
                       vmin=vmin, vmax=vmax)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_slices, len(axes)):
        axes[i].axis("off")

    # Remove ALL spacing
    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0,
        hspace=0
    )

    # Optional figure title
    # fig.suptitle(title, fontsize=14)

    plt.show()

def do_resize(im_data: np.ndarray = 0, dim:int =1):
    im_data_new = np.zeros([dim[0], dim[1], dim[2]], dtype=float)
    nx, ny, nz = im_data.shape
    n_idx = np.argmin([nx, ny, nz])

    if n_idx == 0:    # axial
      for z in range(nz):
        im_data_new[:, :, z] = cv2.resize(np.squeeze(im_data[:, :, z]), [dim[1], dim[0]])   #  width, height

    if n_idx == 1:    # cor
      for x in range(nx):
        im_data_new[x, :, :] = cv2.resize(np.squeeze(im_data[x, :, :]), [dim[0], dim[1]])

    if n_idx == 2:    # sag
      for z in range(nx):
        im_data_new[z, :, :] = cv2.resize(np.squeeze(im_data[z, :, :]), [dim[2], dim[1]])

    return im_data_new

def pad_zeros(im_data):
  nx, ny, nz = im_data.shape
  k = np.argmax([nx, ny, nz])
  if k == 0:
    nx_diff = 0
    ny_diff = int(0.5*(nx - ny))
    nz_diff = int(0.5*(nx - nz))
  elif k == 1:
    nx_diff = int(0.5*(ny - nx))
    ny_diff = 0
    nz_diff = int(0.5*(ny - nz))
  elif k == 2:
    nx_diff = int(0.5*(nz - nx))
    ny_diff = int(0.5*(nz - ny))
    nz_diff = 0

  im_data_new = np.pad(im_data, ((nx_diff, nx_diff), (ny_diff, ny_diff), (nz_diff, nz_diff)),
                       constant_values=(0, 0))
  return im_data_new

if __name__ == "__main__":
    
    #-------------------------------
    # 1️⃣ Read raw data and display/ 3DTSE/1
    # Save as NIfTI 
    #-------------------------------  
    #-------------------------------
    # 2️⃣ Read raw nii and circshift/roll and save .npy
    #-------------------------------

    data_folder = 'DataSRR/volunteer_xxx/raw'
    subject="10mm/nii"
    output_folder = 'DataSRR/volunteer_xxx'

    base_path = 'DataSRR/volunteer_xxx/10mm'
    output_folder_path = 'DataSRR/volunteer_xxx/10mm/nii'
    nii_file_path = os.path.join(output_folder_path, 'axial_2.nii.gz')

    im_circshifted_nii_path = os.path.join(base_path, 'circ_shifted')
    if not os.path.exists(im_circshifted_nii_path):
        os.makedirs(im_circshifted_nii_path)

    im_circshifted_nii_path = os.path.join(im_circshifted_nii_path, 'axial_2_circshift_1zy.nii.gz')

    save_path_npy = os.path.join(base_path, 'npy')
    # create the output folder if it doesn't exist
    if not os.path.exists(save_path_npy):
        os.makedirs(save_path_npy)

    save_path_npy = os.path.join(save_path_npy, 'axial_circshift_1zy.npy')

    print(f"Reading raw data from {data_folder} for subject {subject}...")

    im = read_lf_data(
    data_folder=data_folder,
    output_folder=output_folder,
    subject=subject,
    sub_folder='3DTSE/4',
    file_name='axial_4',
    save_kspace_npy=True,
    homodyne_correction=True,
    )

    # call resample_image function with current spacing and target spacing
    current_spacing = [2.0, 2.0, 10.0]  # Example current voxel size
    target_spacing = [2.0, 2.0, 10.0]  # Example current voxel size

    # read saved nifti file to get im variable
    nifti_img = nib.load(nii_file_path)
    im = nifti_img.get_fdata()
    print("Shape of original im:", im.shape)
    print("Completed reading and displaying raw data.")
    # visualize_volume(im, title="Original")

    # --------------------------------
    # 2️⃣ Circular shifts using np.roll
    # --------------------------------
    vol_roll_z = np.roll(im, 2, axis=2)  # shift slices (Z-axis)
    # visualize_volume_slices(vol_roll_z, title="Roll Z (slices)")
    vol_roll_x = np.roll(vol_roll_z, 13, axis=1)   # shift horizontally (X-axis)
    # visualize_volume(vol_roll_x, title="Roll X (horizontal)")
    vol_roll_y = np.roll(vol_roll_x, 1, axis=0)  # shift vertically (Y-axis)
    # visualize_volume(vol_roll_y, title="Roll Y (vertical)")

    im_resampled = resample_image(vol_roll_y, current_spacing, target_spacing)
    print("Shape of resampled im:", im_resampled.shape)

    # rotate by 90 degrees in the opposite direction
    # im_resampled = np.rot90(im_resampled, k=-1, axes=(0, 1))  # Rotate in the plane of first two axes

    # if visible:
    #     s = OrthoSlicer3D(np.abs(im_resampled))
    #     s.clim = [0, np.abs(1.5 * np.max(np.abs(im_resampled)))]
    #     s.cmap = 'gray'
    #     s.show()

    im_axial = np.moveaxis(im_resampled, [0, 1, 2], [0, 1, 2])  # yzx --> xyz
    im_axial = do_resize(im_data=im_axial, dim=[110, 110, 80])
    im_axial = pad_zeros(im_axial)

    # if visible:
    #     s = OrthoSlicer3D(np.abs(im_axial))
    #     s.clim = [0, np.abs(1.5 * np.max(np.abs(im_axial)))]
    #     s.cmap = 'gray'
    #     s.show()
    
    # save resampled image to NIfTI
    make_nifti(
        im_resampled,
        fname=im_circshifted_nii_path,
        mask=False,
        res=target_spacing,
        dim_info=[0, 1, 2]
    )

    np.save(save_path_npy, im_resampled)
    print(f"Resampled image saved to {save_path_npy}")

    #read .npy file and print shape
    npy_data = np.load(save_path_npy)
    print("Shape of .npy file data:", npy_data.shape)