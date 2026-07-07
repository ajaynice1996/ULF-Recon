import os
import sys
import subprocess
import numpy as np
import nibabel as nib
import ants

sys.path.insert(0, './')
sys.path.append('./src')
sys.path.append('./Niftymic_related_r21')

from display_vlf_ni_data import plot_anatomy_raw 
from preprocess4srr import non_local_means_denoising
from prep4srr_2step_v2 import make_nifti, pad_zeros, do_resize

# ========
# SETTINGS
# ========
niftymic = False
visualize = True

dataFolder = r'./DataSRR_AJ/SRR/ajay_training_06/npy'
subjectID = 'sub_007'

outputfolder = os.path.join(
    r'./DataSRR_AJ/SRR/ajay_training_06/Outputs',
    subjectID
)

os.makedirs(outputfolder, exist_ok=True)

# =========================
# LOAD DATA
# =========================
im_axial = np.abs(np.load(os.path.join(dataFolder, 'axial_circshift_1zy.npy')))
im_sag   = np.abs(np.load(os.path.join(dataFolder, 'sagittal_circshift_1yx.npy')))
im_cor   = np.abs(np.load(os.path.join(dataFolder, 'coronal_circshift_1zx.npy')))

# =========================
# REORIENT TO XYZ
# =========================
# =========================
# Convert all stacks to XYZ
# =========================

# axial: yzx -> xyz
im_axial = np.transpose(im_axial, (2, 1, 0))

# sagittal: yxz -> xyz
im_sag = np.transpose(im_sag, (1, 0, 2))

# coronal: zxy -> xyz
im_cor = np.transpose(im_cor, (1, 2, 0))

# =========================
# Visualize
# =========================

# plot_anatomy_raw(im_axial_xyz, clim=[0, 2048])
# plot_anatomy_raw(im_sag_xyz,   clim=[0, 2048])
# plot_anatomy_raw(im_cor_xyz,   clim=[0, 2048])
# =========================
# DENOISING
# =========================
im_axial = non_local_means_denoising(im_axial)
im_sag   = non_local_means_denoising(im_sag)
im_cor   = non_local_means_denoising(im_cor)

# =========================
# PAD ONLY (NO RESIZE)
# =========================
im_axial = do_resize(im_data=im_axial, dim=[80, 110, 110])
im_axial = pad_zeros(im_axial)

im_sag = do_resize(im_data=im_sag, dim=[110, 110, 80])
im_sag = pad_zeros(im_sag)

im_cor = do_resize(im_data=im_cor, dim=[110, 80, 110])
im_cor = pad_zeros(im_cor)


# plot_anatomy_raw(im_axial_xyz, clim=[0, 2048])
# plot_anatomy_raw(im_sag_xyz,   clim=[0, 2048])
# plot_anatomy_raw(im_cor_xyz,   clim=[0, 2048])
# =========================

# =========================
# SAVE NIFTI (CRITICAL FIX)
# =========================

axial_path = os.path.join(outputfolder, 'axial_redo.nii.gz')
sag_path   = os.path.join(outputfolder, 'sag_redo.nii.gz')
cor_path   = os.path.join(outputfolder, 'cor_redo.nii.gz')

# TRUE ACQUISITION SPACING (VERY IMPORTANT)
make_nifti(im_axial, axial_path, mask=False, res=[2,2,2], dim_info=[2,1,0])
make_nifti(im_sag,   sag_path,   mask=False, res=[2,2,2], dim_info=[0,1,2])
make_nifti(im_cor,   cor_path,   mask=False, res=[2,2,2], dim_info=[0,2,1])

# =========================
# MASKS (ROBUST THRESHOLD)
# =========================


th_ax  = np.percentile(im_axial, 0)
th_sag = np.percentile(im_sag, 0)
th_cor = np.percentile(im_cor, 0)

ax_mask = (im_axial > th_ax).astype(np.uint8)
sg_mask = (im_sag > th_sag).astype(np.uint8)
cr_mask = (im_cor > th_cor).astype(np.uint8)

ax_mask_path = os.path.join(outputfolder, 'axial_mask_redo.nii.gz')
sg_mask_path = os.path.join(outputfolder, 'sag_mask_redo.nii.gz')
cr_mask_path = os.path.join(outputfolder, 'cor_mask_redo.nii.gz')

make_nifti(ax_mask, ax_mask_path, mask=True, res=[2,2,2], dim_info=[2,1,0])
make_nifti(sg_mask, sg_mask_path, mask=True, res=[2,2,2], dim_info=[0,1,2])
make_nifti(cr_mask, cr_mask_path, mask=True, res=[2,2,2], dim_info=[0,2,1])

print("Shapes after registration:")
print("Axial:", im_axial.shape)
print("Sag :", im_sag.shape)
print("Cor :", im_cor.shape)


# =========================
# VISUALIZATION
# =========================
if visualize:
    plot_anatomy_raw(im_axial, clim=[0,2048])
    # plot_anatomy_raw(ax_mask, clim=[0,2048])
    plot_anatomy_raw(im_sag,   clim=[0,2048])
    # plot_anatomy_raw(sg_mask,   clim=[0,2048])
    plot_anatomy_raw(im_cor,   clim=[0,2048])
    # plot_anatomy_raw(cr_mask,   clim=[0,2048])

ax_path = r'./Data/Data/In_vivo/2_avg/axial_circshift_1yz.npy'


# load the numpy file, convert complex data to real (magnitude) and cast to float
ax_np = np.load(ax_path)
ax_np = np.abs(ax_np).astype(np.float32)
fixed = np.moveaxis(ax_np, [0, 1, 2], [1, 2, 0])  # yzx --> xyz
fixed = do_resize(im_data=fixed, dim=[80, 110, 110])
fixed = pad_zeros(fixed)
plot_anatomy_raw(fixed, clim=[0,2048])

# create an ANTs image from the numpy array instead of using ants.image_read on a .npy with complex dtype
fixed = ants.from_numpy(fixed)
print("Original shape:", fixed.shape)
plot_anatomy_raw(fixed.numpy(), clim=[0,2048])


axial_img = ants.image_read(axial_path)
sag_img   = ants.image_read(sag_path)
cor_img   = ants.image_read(cor_path)

axial_msk = ants.image_read(ax_mask_path)
sag_msk   = ants.image_read(sg_mask_path)
cor_msk   = ants.image_read(cr_mask_path)

# =========================
# REFERENCE = AXIAL
# =========================

# fixed = axial_img

# =========================
# FUNCTION: RIGID REGISTRATION
# =========================

def rigid_register(fixed, moving):

    # Rigid transform (good for orthogonal stacks)
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='Rigid',   # IMPORTANT
        verbose=True
    )

    warped = reg['warpedmovout']
    transform = reg['fwdtransforms']

    return warped, transform

# =========================
# REGISTER SAGITTAL → AXIAL
# =========================

axial_reg, axial_tf = rigid_register(fixed, axial_img)
sag_reg, sag_tf = rigid_register(fixed, sag_img)
cor_reg, cor_tf = rigid_register(fixed, cor_img)

# =========================
# APPLY SAME TRANSFORMS TO MASKS
# =========================

axial_mask_reg = ants.apply_transforms(
    fixed=fixed,
    moving=axial_msk,
    transformlist=axial_tf,
    interpolator='nearestNeighbor'
)

sag_mask_reg = ants.apply_transforms(
    fixed=fixed,
    moving=sag_msk,
    transformlist=sag_tf,
    interpolator='nearestNeighbor'
)

cor_mask_reg = ants.apply_transforms(
    fixed=fixed,
    moving=cor_msk,
    transformlist=cor_tf,
    interpolator='nearestNeighbor'
)

# =========================
# SAVE REGISTERED VOLUMES
# =========================

axial_reg_path = os.path.join(outputfolder, 'axial_reg.nii.gz')
sag_reg_path   = os.path.join(outputfolder, 'sag_reg.nii.gz')
cor_reg_path   = os.path.join(outputfolder, 'cor_reg.nii.gz')

axial_mask_reg_path = os.path.join(outputfolder, 'axial_mask_reg.nii.gz')
sag_mask_reg_path   = os.path.join(outputfolder, 'sag_mask_reg.nii.gz')
cor_mask_reg_path   = os.path.join(outputfolder, 'cor_mask_reg.nii.gz')

ants.image_write(axial_reg, axial_reg_path)
ants.image_write(sag_reg,   sag_reg_path)
ants.image_write(cor_reg,   cor_reg_path)

ants.image_write(axial_mask_reg, axial_mask_reg_path)
ants.image_write(sag_mask_reg, sag_mask_reg_path)
ants.image_write(cor_mask_reg, cor_mask_reg_path)

print("Registration completed and saved.")

print("Shapes after registration:")
print("Axial:", axial_reg.shape)
print("Sag :", sag_reg.shape)
print("Cor :", cor_reg.shape)

# # =========================
# # NIFTYMIC OUTPUT
# # =========================

niftymic_output = os.path.join(
    outputfolder,
    'srr_output_2mm.nii.gz'
)

# =========================
if visualize:
    plot_anatomy_raw(axial_reg.numpy(), clim=[0,2048])
    plot_anatomy_raw(sag_reg.numpy(), clim=[0,2048])
    plot_anatomy_raw(cor_reg.numpy(),   clim=[0,2048])

img = nib.load(sag_reg_path)
print(img.shape)

# =========================
# RUN NIFTYMIC
# =========================
if niftymic:

    print("Running NiftyMIC...")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.getcwd()}:{os.getcwd()}",
        "-w", os.getcwd(),
        "renbem/niftymic",
        "niftymic_reconstruct_volume",

        "--filenames",
        axial_path, cor_path, sag_path,

        "--filenames-masks",
        ax_mask_path, cr_mask_path, sg_mask_path,

        # "--filenames", 
        # axial_reg_path, cor_reg_path, sag_reg_path,
        # "--filenames-masks", 
        # axial_mask_reg_path, cor_mask_reg_path, sag_mask_reg_path,

        "--alpha", "0.02",
        "--outlier-rejection", "0",
        "--threshold-first", "0.5",
        "--threshold", "0.7",
        "--intensity-correction", "1",
        "--two-step-cycles", "1",
        "--isotropic-resolution", "2",
        "--reconstruction-type", "HuberL2",
        "--output", niftymic_output,
        "--verbose", "1"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)

# ===================
# LOAD RESULT
# ===================
if visualize:

    print("Loading SRR output...")

    im_srr = nib.load(niftymic_output).get_fdata()

    print("SRR shape:", im_srr.shape)

    plot_anatomy_raw(im_srr, clim=[0,2048])