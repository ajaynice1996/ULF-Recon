import nibabel as nib
from nilearn.plotting import plot_img
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

import matplotlib.pyplot as plt

# Path to your NIfTI file
# nii_path = "srr_zssr_3D_full.nii.gz"
# Load the NIfTI image
# img = nib.load(nii_path)

# Load NIfTI
img = nib.load("DataSRR/ajay_scan_2/10mm/npy/native_niftymic_inputs_thr40/srr_alpha001_out0_ic0_cycle1_iso2_HuberL2.nii.gz")

# Extract image data (this is IMPORTANT)
data = img.get_fdata()
# Print shape of the data
print("Data shape:", data.shape)
# min and max values
print("Data min:", data.min())
print("Data max:", data.max())

# Display with OrthoSlicer3D
OrthoSlicer3D(data).show()