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

    im = read_lf_data(
    data_folder='DataSRR_AJ/SRR/1_ajayscan_leo/raw',
    output_folder='./DataSRR_AJ/SRR/1_ajayscan_leo',
    subject="10mm",
    sub_folder='3DTSE/2',
    file_name='lf_mri.nii.gz'
    )

    # call resample_image function with current spacing and target spacing
    current_spacing = [2.0, 2.0, 10.0]  # Example current voxel size
    target_spacing = [2.0, 2.0, 10.0]

    # read saved nifti file to get im variable
    nifti_img = nib.load('./DataSRR_AJ/SRR/1_ajayscan_leo/10mm/lf_mri_3DTSE_2.nii.gz')
    im = nifti_img.get_fdata()
    print("Shape of original im:", im.shape)
    print("Completed reading and displaying raw data.")
    visualize_volume(im, title="Original")

    # --------------------------------
    # 2️⃣ Circular shifts using np.roll
    # --------------------------------
    vol_roll_z = np.roll(im, 2, axis=2)  # shift slices (Z-axis)
    visualize_volume(vol_roll_z, title="Roll Z (slices)")
    vol_roll_x = np.roll(vol_roll_z, 37, axis=1)   # shift horizontally (X-axis)
    visualize_volume(vol_roll_x, title="Roll X (horizontal)")
    vol_roll_y = np.roll(vol_roll_x, 0, axis=0)  # shift vertically (Y-axis)
    visualize_volume(vol_roll_y, title="Roll Y (vertical)")

    im_resampled = resample_image(vol_roll_y, current_spacing, target_spacing)
    print("Shape of resampled im:", im_resampled.shape)
    # im_resampled = np.rot90(im_resampled, k=1, axes=(1, 2))

    if visible:
        s = OrthoSlicer3D(np.abs(im_resampled))
        s.clim = [0, np.abs(1.5 * np.max(np.abs(im_resampled)))]
        s.cmap = 'gray'
        s.show()

    im_cor = np.moveaxis(im_resampled, [0, 1, 2], [0, 2, 1])  # zxy --> xyz
    im_cor = do_resize(im_data=im_cor, dim=[110, 80, 110])
    im_cor = pad_zeros(im_cor)

    if visible:
        s = OrthoSlicer3D(np.abs(im_cor))
        s.clim = [0, np.abs(1.5 * np.max(np.abs(im_cor)))]
        s.cmap = 'gray'
        s.show()

    # save resampled image to NIfTI
    make_nifti(
        im_resampled,
        fname='./DataSRR_AJ/SRR/1_ajayscan_leo/10mm/lf_mri_3DTSE_2_resampled.nii.gz',
        mask=False,
        res=target_spacing,
        dim_info=[0, 1, 2]
    )

    make_nifti(
        im_cor,
        fname='./DataSRR_AJ/SRR/1_ajayscan_leo/10mm/lf_mri_3DTSE_2_resampled_cor.nii.gz',
        mask=False,
        res=[2,2,2],
        dim_info=[0, 1, 2]
    )

    save_path_npy = './DataSRR_AJ/SRR/1_ajayscan_leo/npy1/coronal_circshift_1zx.npy'

    np.save(save_path_npy, im_resampled)
    print(f"Resampled image saved to {save_path_npy}")

    #read .npy file and print shape
    npy_data = np.load(save_path_npy)
    print("Shape of .npy file data:", npy_data.shape)