import os
import nibabel as nib
from nilearn.plotting import plot_anat
import numpy as np

# read nii.gz files from the base folder and display using Orthoslicer

import matplotlib.pyplot as plt

from nibabel.viewers import OrthoSlicer3D

file_path = 'DataSRR_AJ/SRR/ajay_training_06/10mm/lf_mri_3DTSE_2_resampled.nii.gz'  # Replace with your filename
img = nib.load(file_path)
print(f"Displaying: {os.path.basename(file_path)}")
plot_anat(img, title=os.path.basename(file_path), display_mode='ortho')
plt.show()
header = img.header
print("Header Information of Transformed Image:")
#print vixel size
voxel_sizes = header.get_zooms()
print("Voxel Sizes (mm):", voxel_sizes)

data = img.get_fdata()

s = OrthoSlicer3D(np.abs(img.get_fdata()))
s.clim = [0, np.abs(1.5 * np.max(np.abs(img.get_fdata())))]
s.cmap = 'gray'
s.show()

# move axis 0 to axis 2
data = np.moveaxis(data, 0, 2)

# Rotate all axes by 90 degrees
data_rotated = np.rot90(data, k=1, axes=(0, 1))
data_rotated = np.rot90(data_rotated, k=1, axes=(0, 2))
data_rotated = np.rot90(data_rotated, k=1, axes=(1, 2))
# Create a new NIfTI image with the transformed data
img_transformed = nib.Nifti1Image(data_rotated, img.affine, img.header)
# Display the transformed image
print(f"Displaying Transformed: {os.path.basename(file_path)}")
plot_anat(img_transformed, title=f"Transformed {os.path.basename(file_path)}", display_mode='ortho')
plt.show()

#save the transformed image
output_folder = 'DataSRR_AJ/SRR/ajay_training_06'
output_path = os.path.join(output_folder, 'transformed_' + os.path.basename(file_path))


nib.save(img_transformed, output_path)

# read header information of the transformed image
transformed_img = nib.load(output_path)
header = transformed_img.header
print("Header Information of Transformed Image:")
#print vixel size
voxel_sizes = header.get_zooms()
print("Voxel Sizes (mm):", voxel_sizes)

#view using orthoslicer

s = OrthoSlicer3D(np.abs(transformed_img.get_fdata()))
s.clim = [0, np.abs(1.5 * np.max(np.abs(transformed_img.get_fdata())))]
s.cmap = 'gray'
s.show()