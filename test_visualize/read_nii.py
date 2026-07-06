import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

def robust_clim(data, low=1, high=99.5, nonzero_only=True):
    vals = np.asarray(data)
    vals = vals[np.isfinite(vals)]

    if nonzero_only:
        vals = vals[vals != 0]

    if vals.size == 0:
        return 0, 1

    return tuple(np.percentile(vals, [low, high]).astype(float))

def choose_clim(data, mode="percentile"):
    data = np.asarray(data)

    if mode == "max_1p25":
        vmax = 1.25 * float(np.nanmax(data))
        return 0, vmax

    return robust_clim(data, low=1, high=99.5, nonzero_only=True)

def central_slices(data):
    cx, cy, cz = [s // 2 for s in data.shape[:3]]

    return {
        "axis0_mid": data[cx, :, :],
        "axis1_mid": data[:, cy, :],
        "axis2_mid": data[:, :, cz],
    }

def save_recon_qc_png(data, save_path, title="Refined SRR", clim_mode="percentile"):
    # data = img.get_fdata().astype(np.float32)
    clim = choose_clim(data, mode=clim_mode)

    slices = central_slices(data)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"{title}\n"
        f"shape={img.shape[:3]}, zooms={img.header.get_zooms()[:3]}, "
        f"orientation={nib.aff2axcodes(img.affine)}, clim={clim}",
        fontsize=11,
    )

    for ax, key in zip(axes, ["axis0_mid", "axis1_mid", "axis2_mid"]):
        view = np.rot90(slices[key])
        ax.imshow(view, cmap="gray", vmin=clim[0], vmax=clim[1])
        ax.set_title(key)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print("Saved QC PNG:", save_path)
    plt.close()
    # plt.show()

# Input folder
data_folder = 'DataSRR/ajay_scan_2/10mm/npy/niftymic_output'

# Get all .nii or .nii.gz files
files = [f for f in os.listdir(data_folder) if f.endswith(('.nii', '.nii.gz'))]

for i, file_name in enumerate(files):
    file_path = os.path.join(data_folder, file_name)
    img = nib.load(file_path)
    data = np.abs(img.get_fdata())

    print(f"\n[{i+1}] Processing: {file_name}")
    print(f"Shape: {data.shape}")

    # Use Nibabel OrthoSlicer3D to visualize interactively
    print("Launching OrthoSlicer3D viewer... (close the window to continue)")
    ortho_view = OrthoSlicer3D(data)
    ortho_view.show()

    ORTHOSLICER_CLIM_MODE = "percentile"
    QC_DIR = 'DataSRR/ajay_scan_2/10mm/npy/QC_Images1'

    #make directory if it doesn't exist
    os.makedirs(QC_DIR, exist_ok=True)

    # BASEPATH OF FILE_NAME
    run_name = os.path.splitext(file_name)[0]
    save_recon_qc_png(
            data,
            save_path=os.path.join(
                QC_DIR,
                run_name + ".png",
            ),
            title=run_name,
            clim_mode=ORTHOSLICER_CLIM_MODE,
        )
