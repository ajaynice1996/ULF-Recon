import os
import numpy as np
import nibabel as nib
import subprocess
import matplotlib.pyplot as plt

# =========================================================
# INPUT
# =========================================================

input_file = (
    "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/"
    "srr_alpha001_out1_ic0_cycle2_iso2_HuberL2.nii.gz"
)

out_dir = "DataSRR/volunteer_xxx/10mm/npy/synthstrip_output"

os.makedirs(out_dir, exist_ok=True)

# =========================================================
# Load NPY or NIFTI
# =========================================================

def load_image_as_nifti(path):

    ext = path.lower()

    # -------------------------
    # NIfTI
    # -------------------------

    if ext.endswith(".nii") or ext.endswith(".nii.gz"):

        print("Loading NIfTI")

        img = nib.load(path)   # <-- ORIGINAL PATH

        print("Shape:", img.shape)
        print("Zooms:", img.header.get_zooms()[:3])
        print("Orientation:",
              nib.aff2axcodes(img.affine))

        return img

    # -------------------------
    # NPY
    # -------------------------

    elif ext.endswith(".npy"):

        print("Loading NPY")

        data = np.load(path)

        print("Shape:", data.shape)
        print("dtype:", data.dtype)

        affine = np.eye(4)

        img = nib.Nifti1Image(
            data.astype(np.float32),
            affine
        )

        return img

    else:
        raise ValueError(
            "Only .npy, .nii, .nii.gz supported"
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

# =========================================================
# Load image
# =========================================================

img = load_image_as_nifti(input_file)

# =========================================================
# Save SynthStrip input
# =========================================================

base_name = os.path.basename(input_file)

base_name = (
    base_name
    .replace(".nii.gz","")
    .replace(".nii","")
    .replace(".npy","")
)

input_nii = os.path.join(
    out_dir,
    base_name + "_input.nii.gz"
)

# Save WITHOUT changing affine

nib.save(
    img,
    input_nii
)

print("\nSynthStrip input:")
print(input_nii)

# =========================================================
# SynthStrip output names
# =========================================================

skullstrip_img = os.path.join(
    out_dir,
    base_name + "_skullstrip.nii.gz"
)

brain_mask = os.path.join(
    out_dir,
    base_name + "_mask.nii.gz"
)

# =========================================================
# Run SynthStrip Docker
# =========================================================

cmd = [
    "./synthstrip-docker",
    "-i",
    input_nii,
    "-o",
    skullstrip_img,
    "-m",
    brain_mask
]

print("\nRunning SynthStrip:")
print(" ".join(cmd))

subprocess.run(
    cmd,
    check=True
)

print("\nSynthStrip finished")

# =========================================================
# Save NPY outputs
# =========================================================

brain_data = nib.load(
    skullstrip_img
).get_fdata()

mask_data = nib.load(
    brain_mask
).get_fdata()

np.save(
    os.path.join(
        out_dir,
        base_name+"_skullstrip.npy"
    ),
    brain_data
)

np.save(
    os.path.join(
        out_dir,
        base_name+"_mask.npy"
    ),
    mask_data
)

# =========================================================
# QC visualization (Axial / Coronal / Sagittal)
# =========================================================

import matplotlib.pyplot as plt
import numpy as np

# Load arrays
original = img.get_fdata().astype(np.float32)
brain = brain_data.astype(np.float32)
mask = mask_data.astype(np.float32)


print("\nVisualization volumes:")
print("Original:", original.shape)
print("Brain:", brain.shape)
print("Mask:", mask.shape)


# Check same dimensions
assert original.shape == brain.shape == mask.shape, \
    "Original, brain, and mask shapes do not match"


nx, ny, nz = original.shape


# Middle slices
mid_x = nx // 2
mid_y = ny // 2
mid_z = nz // 2


print("\nMiddle slices:")
print("Sagittal X:", mid_x)
print("Coronal Y:", mid_y)
print("Axial Z:", mid_z)



# ---------------------------------------------------------
# Extract 2D slices
# ---------------------------------------------------------

views = [

    (
        "Axial",
        original[:, :, mid_z],
        brain[:, :, mid_z],
        mask[:, :, mid_z]
    ),

    (
        "Coronal",
        original[:, mid_y, :],
        brain[:, mid_y, :],
        mask[:, mid_y, :]
    ),

    (
        "Sagittal",
        original[mid_x, :, :],
        brain[mid_x, :, :],
        mask[mid_x, :, :]
    )

]



# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, axes = plt.subplots(
    3,
    4,
    figsize=(16, 12)
)


for row, (name, orig, brain_slice, mask_slice) in enumerate(views):


    # Original

    axes[row,0].imshow(
        np.rot90(orig),
        cmap="gray"
    )

    axes[row,0].set_title(
        f"{name} - Original"
    )


    # SynthStrip brain

    axes[row,1].imshow(
        np.rot90(brain_slice),
        cmap="gray"
    )

    axes[row,1].set_title(
        f"{name} - Brain"
    )


    # Mask

    axes[row,2].imshow(
        np.rot90(mask_slice),
        cmap="gray"
    )

    axes[row,2].set_title(
        f"{name} - Mask"
    )


    # Overlay

    axes[row,3].imshow(
        np.rot90(orig),
        cmap="gray"
    )

    axes[row,3].imshow(
        np.rot90(mask_slice),
        cmap="jet",
        alpha=0.35
    )

    axes[row,3].set_title(
        f"{name} - Overlay"
    )


    for col in range(4):
        axes[row,col].axis("off")



plt.tight_layout()



qc_file = os.path.join(
    out_dir,
    base_name + "_orthogonal_QC.png"
)


plt.savefig(
    qc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nQC saved:")
print(qc_file)

# View using OrthoSlicer3D if available
