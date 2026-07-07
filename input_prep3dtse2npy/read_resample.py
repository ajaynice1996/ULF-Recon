import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Input and output file paths
input_path = 'DataSRR_AJ/SRR/1_ajayscan_leo/nii/3_sagital.nii.gz'
output_path = 'DataSRR_AJ/SRR/1_ajayscan_leo/resample/3_sagital_resampled.nii.gz'

# Desired voxel size (mm)
# current_voxel_size = np.array([2.0, 2.037037037037037, 10.0])  # axial Example current voxel size
# current_voxel_size = np.array([2.0, 2.037037037037037, 10.0]) #coronal Example current voxel size
current_voxel_size = np.array([2.0, 2.0, 10.0]) #sagittal Example current voxel size
target_voxel_size = np.array([2, 2, 5])

# Load the NIfTI image
img = nib.load(input_path)
data = img.get_fdata()
affine = img.affine

# Get current voxel size from affine
# current_voxel_size = np.abs(affine[:3, :3].diagonal())
print("Current voxel size (mm):", current_voxel_size)
# Compute zoom factors
zoom_factors = current_voxel_size / target_voxel_size

# Resample image data
resampled_data = zoom(data, zoom_factors, order=3)

# Update affine for new voxel size
new_affine = affine.copy()
new_affine[:3, :3] = np.diag(target_voxel_size) * np.sign(affine[:3, :3].diagonal())

# Save resampled image
resampled_img = nib.Nifti1Image(resampled_data, new_affine)
nib.save(resampled_img, output_path)

# print shape of original and resampled data
print("Original shape:", data.shape)
print("Resampled shape:", resampled_data.shape)

# # save resampled data to .npy file
# np.save(output_path.replace('.nii.gz', '.npy'), resampled_data)

# # read .npy file and print shape Data/VLF_invivo/raw/axial_circshift_1yz.npy
# npy_data = np.load('34507/lf_mri_3DTSE_9_resampled.npy')
# print("Shape of .npy file data:", npy_data.shape)