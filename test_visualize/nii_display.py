# make a code to read the nii.gz file and display the images in 3D using OrthoSlicer3D from kea3d, and also print the max, min, data type, shape of the image, and resolution if available. Also save the image as a nifti file for further use.
import os
import glob
import nibabel as nib
import numpy as np

from nibabel.viewers import OrthoSlicer3D
# from kea2nifti import make_nifti

data_dir = "DataSRR/volunteer_x/10mm/circ_shifted"

# Directory containing .nii.gz files
# data_dir = os.path.dirname(os.path.abspath(__file__))

# Find all .nii.gz files in the directory
nii_files = glob.glob(os.path.join(data_dir, "*.nii.gz"))   

if not nii_files:
    print("No .nii.gz files found in the directory.")
    exit(1)

for nii_path in nii_files:  

    print(f"Displaying: {os.path.basename(nii_path)}")
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    # Display the image using OrthoSlicer3D
    s = OrthoSlicer3D(np.abs(data))
    s.clim = [0, np.abs(1.5 * np.max(np.abs(data)))]
    s.cmap = 'gray'
    s.show()
    
    # Print max, min, data type, shape, and resolution if available
    print("Max value:", np.max(np.abs(data)))
    print("Min value:", np.min(np.abs(data)))
    print("Data type of np.abs(data):", np.abs(data).dtype)
    print("Shape of data:", data.shape)


