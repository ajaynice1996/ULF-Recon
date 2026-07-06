import os
import glob
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

data_dir = "DataSRR_AJ/SRR/ajay_training_06/npy1"  # change to the folder with your .npy files

npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
if not npy_files:
    print("No .npy files found in the directory.")
    raise SystemExit(1)

for npy_path in npy_files:
    basename = os.path.splitext(os.path.basename(npy_path))[0]
    print(f"Processing: {basename}.npy")

    data = np.load(npy_path, allow_pickle=False)
    # If 4D (e.g. X,Y,Z,T) take the first volume
    if data.ndim == 4:
        print("4D data detected - using first volume (index 0).")
        data = data[..., 0]

    data_abs = np.abs(data)

    # Display
    s = OrthoSlicer3D(data_abs)
    s.clim = [0, 1.5 * np.max(data_abs)]
    s.cmap = "gray"
    s.show()

    # Stats
    print("Max value:", np.max(data_abs))
    print("Min value:", np.min(data_abs))
    print("Data type of np.abs(data):", data_abs.dtype)
    print("Shape of data:", data.shape)

    # # Try to discover resolution:
    # resolution = None
    # npz_path = os.path.join(data_dir, basename + ".npz")
    # if os.path.exists(npz_path):
    #     try:
    #         meta = np.load(npz_path, allow_pickle=True)
    #         if "zooms" in meta:
    #             resolution = tuple(meta["zooms"])
    #         elif "affine" in meta:
    #             affine = meta["affine"]
    #             # voxel sizes approximated as norms of affine's first 3 columns
    #             resolution = tuple(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))
    #         elif "spacing" in meta:
    #             resolution = tuple(meta["spacing"])
    #     except Exception:
    #         pass

    # if resolution is not None:
    #     print("Resolution (voxel sizes):", resolution)
    # else:
    #     print("Resolution not available in .npy. Provide an affine or .npz with 'affine'/'zooms' to set it.")

    # # Save as NIfTI for further use
    # affine_to_use = np.eye(4)
    # if resolution is not None:
    #     # build a diagonal affine preserving voxel sizes (no rotation)
    #     affine_to_use = np.diag(list(resolution) + [1.0])
    # elif os.path.exists(npz_path):
    #     try:
    #         meta = np.load(npz_path, allow_pickle=True)
    #         if "affine" in meta:
    #             affine_to_use = meta["affine"]
    #     except Exception:
    #         pass

    # nifti_img = nib.Nifti1Image(data, affine_to_use)
    # out_path = os.path.join(data_dir, basename + ".nii.gz")
    # nib.save(nifti_img, out_path)
    # print("Saved NIfTI to:", out_path)
    # print("-" * 40)
