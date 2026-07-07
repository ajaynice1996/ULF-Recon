# This file reproduces the figure in the R21 application

# Import necessary modules
import os.path
import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./src')
sys.path.append('./Niftymic_related_r21')
# sys.path.append('/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/Propsa/Recon/Code2PP/')
sys.path.insert(0, '/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/Propsa/Recon/Code2PP/')
# import cProcessPipeline as cPP
# from sim_input_SR import create_object, down_res
from local.display_vlf_ni_data import plot_anatomy_raw, plot_anatomy_nifti
# import niftyreg as nreg
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from local.preprocess4srr import non_local_means_denoising
from local.prep4srr_2step_v2 import do_resize, make_nifti, create_nifti_header, norm_data, pad_zeros
from src.ZSSR_master import configs, configs_2, ZSSR
from src.ZSSR_master.ZSSR import *
import subprocess

niftymic = False
zssr = True
visualize = True

# Read the data from the scanner and convert to .npy files
dataFolder = r'./Training_subject10mm/npy'
subjectID = 'sub_10mm'
outputfolder = os.path.join(r'./Training_subject10mm/Outputs', subjectID)
os.makedirs(outputfolder, exist_ok=True)

im_yz_folder = 'lf_mri_3DTSE_9_resampled.npy'  # axial
im_yx_folder = 'lf_mri_3DTSE_10_resampled.npy'  # sagittal
im_zx_folder = 'lf_mri_3DTSE_11_resampled.npy'  # coronal

im_axial = np.abs(np.load(os.path.join(dataFolder, im_yz_folder)))
im_sag = np.abs(np.load(os.path.join(dataFolder, im_yx_folder)))
im_cor = np.abs(np.load(os.path.join(dataFolder, im_zx_folder)))

im_axial = np.moveaxis(im_axial, [0, 1, 2], [1, 2, 0])  # yzx --> xyz
im_sag = np.moveaxis(im_sag, [0, 1, 2], [1, 0, 2])  # yxz --> xyz
im_cor = np.moveaxis(im_cor, [0, 1, 2], [2, 0, 1])  # zxy --> xyz

# Preprocess the data (resizing and zero-padding)
# denoise these images first
im_axial = non_local_means_denoising(im_axial)
im_sag = non_local_means_denoising(im_sag)
im_cor = non_local_means_denoising(im_cor)

im_axial = do_resize(im_data=im_axial, dim=[80, 110, 110])
im_axial = pad_zeros(im_axial)
# join name with outputfolder
fname='axial_redo.nii.gz'
fname = os.path.join(outputfolder, fname)
make_nifti(data=im_axial, fname=fname, mask=False,
               res=[2, 2, 2], dim_info=[2, 1, 0]) # phase, freq, slice

im_sag = do_resize(im_data=im_sag, dim=[110, 110, 80])
im_sag = pad_zeros(im_sag)
fname='sag_redo.nii.gz'
fname = os.path.join(outputfolder, fname)
make_nifti(data=im_sag, fname=fname, mask=False,
               res=[2, 2, 2], dim_info=[0, 1, 2])  # phase, freq, slice

im_cor = do_resize(im_data=im_cor, dim=[110, 80, 110])
im_cor = pad_zeros(im_cor)
fname='cor_redo.nii.gz'
fname = os.path.join(outputfolder, fname)
make_nifti(data=im_cor, fname=fname, mask=False,
               res=[2, 2, 2], dim_info=[0, 2, 1])  # phase, freq, slice

# Save the nifti files for NiftyMIC processing via docker
thresh = 0
# ----- Axial -----
im_axial_mask = im_axial > thresh
fname='axial_mask_redo.nii.gz'
fname = os.path.join(outputfolder, fname)
make_nifti(data=im_axial_mask, fname=fname, mask=True,
               res=[2, 2, 2], dim_info=[2, 1, 0])  # phase, freq, slice

# Visualize the image and mask to confirm correctness
plot_anatomy_raw(im_axial, clim=[0, 2048])

# ----- Sagittal -----
# im_sag = nib.load('/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/OSI/Superresolution/super_resolution/Data/In_vivo/2_avg/circ-shifted/sag.nii.gz').get_fdata()
im_sag_mask = im_sag > thresh
fname='sag_mask_redo.nii.gz'
fname = os.path.join(outputfolder, fname)
make_nifti(data=im_sag_mask, fname=fname, mask=True,
               res=[2, 2, 2], dim_info=[2, 1, 0])  # phase, freq, slice

# Visualize the image and mask to confirm correctness
plot_anatomy_raw(im_sag, clim=[0, 2048])

# ----- Coronal -----
# im_cor = nib.load('/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/OSI/Superresolution/super_resolution/Data/In_vivo/2_avg/circ-shifted/cor.nii.gz').get_fdata()
im_cor_mask = im_cor > thresh
fname='cor_mask_redo.nii.gz'
fname = os.path.join(outputfolder, fname)
make_nifti(data=im_cor_mask, fname=fname, mask=True,
               res=[2, 2, 2], dim_info=[2, 1, 0])  # phase, freq, slice

# Visualize the image and mask to confirm correctness
plot_anatomy_raw(im_cor, clim=[0, 2048])