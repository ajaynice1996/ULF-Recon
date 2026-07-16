import os
import nibabel as nib
from nibabel.processing import resample_to_output

# =====================================================
# Input NIfTI file
# =====================================================
INPUT_FILE = "atlas_pipeline/input/subject_T2.nii.gz"

# =====================================================
# Load image
# =====================================================
img = nib.load(INPUT_FILE)

print("\nOriginal Image")
print("----------------------------")
print("Shape       :", img.shape)
print("Voxel Size  :", img.header.get_zooms()[:3])
print("Orientation :", nib.aff2axcodes(img.affine))

# =====================================================
# Resample to 1 mm isotropic
# =====================================================
resampled = resample_to_output(
    img,
    voxel_sizes=(1.0, 1.0, 1.0),
    order=1      # Linear interpolation
)

print("\nResampled Image")
print("----------------------------")
print("Shape       :", resampled.shape)
print("Voxel Size  :", resampled.header.get_zooms()[:3])
print("Orientation :", nib.aff2axcodes(resampled.affine))

# =====================================================
# Save in the same folder
# =====================================================
folder = os.path.dirname(INPUT_FILE)

filename = os.path.basename(INPUT_FILE)

if filename.endswith(".nii.gz"):
    out_name = filename.replace(".nii.gz", "_1mm.nii.gz")
elif filename.endswith(".nii"):
    out_name = filename.replace(".nii", "_1mm.nii")
else:
    raise ValueError("Input must be a .nii or .nii.gz file")

output_file = os.path.join(folder, out_name)

nib.save(resampled, output_file)

print("\nSaved to:")
print(output_file)