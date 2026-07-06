import numpy as np
import nibabel as nib
from read_kea3d import kea3d
from kea2nifti import make_nifti
from nibabel.viewers import OrthoSlicer3D
import os
import matplotlib.pyplot as plt

import numpy as np

# def homodyne_reconstruction(kspace, axis=2, fraction=0.6):
#     """
#     Homodyne reconstruction compatible with fftn + fftshift convention.
#     """

#     # Step 1: ensure k-space is centered (your convention)
#     kspace = np.fft.fftshift(kspace, axes=axis)

#     # Step 2: create weighting mask (same size as kspace)
#     dims = kspace.shape[axis]
#     ramp = np.linspace(0, 1, dims)

#     # low-pass + high-pass split
#     lp_mask = np.zeros(dims)
#     cutoff = int(fraction * dims)

#     lp_mask[:cutoff] = 1
#     lp_mask = np.roll(lp_mask, cutoff // 2)

#     hp_mask = 1 - lp_mask

#     # reshape for broadcasting
#     shape = [1] * kspace.ndim
#     shape[axis] = dims

#     lp_mask = lp_mask.reshape(shape)
#     hp_mask = hp_mask.reshape(shape)

#     # Step 3: split k-space
#     lowpass = kspace * lp_mask
#     weighted_kspace = kspace * hp_mask

#     # Step 4: USE FFT (NOT IFFT)
#     low_img = np.fft.fftshift(
#         np.fft.fftn(np.fft.fftshift(lowpass, axes=axis)),
#         axes=axis
#     )

#     weighted_img = np.fft.fftshift(
#         np.fft.fftn(np.fft.fftshift(weighted_kspace, axes=axis)),
#         axes=axis
#     )

#     # Step 5: phase correction (homodyne core idea)
#     phase = np.exp(-1j * np.angle(low_img + 1e-8))
#     recon = np.real(weighted_img * phase)

#     return recon

import numpy as np

def homodyne_reconstruction(kspace, axis=2, fraction=0.625):
    """
    Classical Homodyne Partial Fourier Reconstruction
    (consistent with IFFT-based MRI pipeline)
    """

    if fraction <= 0.5 or fraction > 1.0:
        raise ValueError("fraction must lie in (0.5, 1].")

    N = kspace.shape[axis]

    acquired = int(np.round(fraction * N))
    overlap = 2 * acquired - N

    if overlap <= 0:
        raise ValueError("Invalid overlap region.")

    # -----------------------------
    # Weighting
    # -----------------------------
    weight = np.zeros(N)

    missing = N - acquired

    weight[missing:] = 2.0
    weight[missing:missing + overlap] = np.linspace(
        1, 2, overlap, endpoint=False
    )

    shape = [1] * kspace.ndim
    shape[axis] = N
    weight = weight.reshape(shape)

    weighted_kspace = kspace * weight

    # -----------------------------
    # Low-res phase estimation
    # -----------------------------
    lowpass = np.zeros_like(kspace)

    center = N // 2
    half = overlap // 2

    slicer = [slice(None)] * kspace.ndim
    slicer[axis] = slice(center - half, center + half)

    lowpass[tuple(slicer)] = kspace[tuple(slicer)]

    # -----------------------------
    # IFFT reconstruction (IMPORTANT)
    # -----------------------------
    weighted_img = np.fft.ifftn(np.fft.ifftshift(weighted_kspace))
    low_img = np.fft.ifftn(np.fft.ifftshift(lowpass))

    # -----------------------------
    # Phase correction
    # -----------------------------
    phase = np.exp(-1j * np.angle(low_img + 1e-8))
    corrected = weighted_img * phase

    # img1 = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace))))
    # img1 = np.abs(np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kspace)))))

    img = np.real(corrected)

    # final display shift
    img = np.fft.fftshift(img)

    return img

import numpy as np

def center_kspace(kspace):
    """
    Automatically center k-space using the maximum magnitude.

    Returns
    -------
    centered_kspace
    shifts
    """

    mag = np.abs(kspace)

    peak = np.array(np.unravel_index(np.argmax(mag), mag.shape))

    center = np.array(kspace.shape)//2

    shift = center - peak

    print("Peak :", peak)
    print("Center :", center)
    print("Shift :", shift)

    kspace = np.roll(kspace, shift[0], axis=0)
    kspace = np.roll(kspace, shift[1], axis=1)
    kspace = np.roll(kspace, shift[2], axis=2)

    return kspace, shift

def visualize_volume_slices(volume, title="Volume", n_cols=4, cmap="gray",
                     vmin=None, vmax=None):
    """
    Visualize all slices of a 3D volume.

    Parameters
    ----------
    volume : ndarray
        3D volume of shape (H, W, Z)
    title : str
        Figure title
    n_cols : int
        Number of columns (default = 4)
    cmap : str
        Colormap
    vmin, vmax : float, optional
        Display intensity limits. If None, matplotlib auto-scales.
    """
    n_slices = volume.shape[2]
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2 * n_rows),
        squeeze=False
    )

    axes = axes.ravel()

    for i in range(n_slices):
        slice_img = np.rot90(volume[:, :, i], k=-1)  # 90° clockwise
        axes[i].imshow(slice_img, cmap=cmap,
                       vmin=vmin, vmax=vmax)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_slices, len(axes)):
        axes[i].axis("off")

    # Remove ALL spacing
    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0,
        hspace=0
    )

    # Optional figure title
    # fig.suptitle(title, fontsize=14)

    plt.show()

def visualize_phase(kspace, n_cols=4):
    """
    Visualize the phase of a 3D complex k-space.
    kspace shape: (Nx, Ny, Nz)
    """
    phase = np.angle(kspace)  # values in [-pi, pi]

    n_slices = phase.shape[2]
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2*n_cols, 2*n_rows),
                             squeeze=False)
    axes = axes.ravel()

    for i in range(n_slices):
        axes[i].imshow(phase[:, :, i],
                       cmap='gray',
                       vmin=-np.pi,
                       vmax=np.pi)
        axes[i].axis('off')
        axes[i].set_title(f'{i}', fontsize=8)

    for i in range(n_slices, len(axes)):
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def read_lf_data(
    data_folder='DataSRR_AJ/SRR/ajay_training_06/',
    output_folder='./DataSRR_AJ/SRR/ajay_training_06/',
    subject='kspace',
    sub_folder='3DTSE/3',
    file_name='lf_mri.nii.gz',
    save_kspace_npy=True,
    homodyne_correction=True,
):
    try:
        file_name = f"{file_name}.nii.gz" if not file_name.endswith('.nii.gz') else file_name

        subject_folder = os.path.join(output_folder, subject)
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)

        kspace_npy_path = os.path.join(output_folder, '10mm', 'kspace')
        if not os.path.exists(kspace_npy_path):
            os.makedirs(kspace_npy_path)
        base_name = file_name.replace(".nii.gz", "")
        kspace_filename = os.path.join(kspace_npy_path, f"kspace_{base_name}.npy")

        # Include subfolder name in the output filename for differentiation
        filename = file_name
        fname_nii = os.path.join(subject_folder, filename)
        print(f"Output NIfTI file will be saved as: {fname_nii}")

        sample_data = kea3d(data_folder=data_folder, sub_folder=sub_folder)
        kspace = sample_data.kspace_gauss_filter

        phase = np.angle(kspace)

        # visualize_volume_slices(phase)

        # Center k-space automatically
        # kspace, shift = center_kspace(kspace)

        #save kspace as npy file
        if save_kspace_npy:
            np.save(kspace_filename, kspace)
            print(f"K-space data saved as: {kspace_filename}")

        im = np.abs(np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kspace)))))

        if im is None:
            print("No data found in the specified folder.")
            return None
        
        print("Displaying the RAW image using OrthoSlicer3D...")

        # visualize_volume_slices(np.abs(im))

        # s = OrthoSlicer3D(np.abs(im))
        # s.clim = [0, np.abs(1.5 * np.max(np.abs(im)))]
        # s.cmap = 'gray'
        # s.show()

        print(np.max(np.abs(im)))
        print("Min value:", np.min(np.abs(im)))
        print("Data type of np.abs(im):", np.abs(im).dtype)
        print("Shape of im:", im.shape)

        if hasattr(sample_data, 'res_dim1') and hasattr(sample_data, 'res_dim2') and hasattr(sample_data, 'res_dim3'):
            print(sample_data.res_dim1, sample_data.res_dim2, sample_data.res_dim3)
        else:
            print("sample_data does not have resolution attributes (res_dim1, res_dim2, res_dim3).")
        # Make nifti in case of need for further inputs to other software 
        make_nifti(
            im,
            fname=fname_nii,
            mask=False,
            res=[sample_data.res_dim1, sample_data.res_dim2, sample_data.res_dim3],
            dim_info=[0, 1, 2]
        )

        print(f"NIfTI file saved as: {fname_nii}")

        # # image space reconstruction to k-space
        # vol_roll_z = np.roll(im, 2, axis=2)  # shift slices (Z-axis)
        # # visualize_volume(vol_roll_z, title="Roll Z (slices)")
        # vol_roll_x = np.roll(vol_roll_z, 13, axis=1)   # shift horizontally (X-axis)
        # # visualize_volume(vol_roll_x, title="Roll X (horizontal)")
        # vol_roll_y = np.roll(vol_roll_x, 1, axis=0)  # shift vertically (Y-axis)

        # print("Displaying the RAW image using OrthoSlicer3D...")
        # s = OrthoSlicer3D(np.abs(vol_roll_y))
        # s.clim = [0, np.abs(1.5 * np.max(np.abs(vol_roll_y)))]
        # s.cmap = 'gray'
        # s.show()

        # kspace_reconstructed = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol_roll_y)))


        if homodyne_correction:
            print("Performing Homodyne reconstruction...")

            # kspace_roll_z = np.roll(kspace, 2, axis=2)
            # kspace_roll_x = np.roll(kspace_roll_z, 13, axis=1)
            # kspace_roll_y = np.roll(kspace_roll_x, 1, axis=0)

            img1 = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace))))
            # img1 = np.abs(np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kspace)))))
            
            visualize_volume_slices(np.abs(img1))

            # visualize_phase(img1)

            # img2 = np.abs(np.fft.ifftn(kspace_roll_y))
            # s = OrthoSlicer3D(np.abs(img2))
            # s.clim = [0, 1.5*np.max(np.abs(img2))]
            # s.cmap = "gray"
            # s.show()
            # img3 = np.abs(np.fft.fftshift(np.fft.ifftn(kspace_roll_y)))
            # s = OrthoSlicer3D(np.abs(img3))
            # s.clim = [0, 1.5*np.max(np.abs(img3))]
            # s.cmap = "gray"
            # s.show()
            # img4 = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace_roll_y)))
            # s = OrthoSlicer3D(np.abs(img4))
            # s.clim = [0, 1.5*np.max(np.abs(img4))]
            # s.cmap = "gray"
            # s.show()
            
            homodyne_img = homodyne_reconstruction(
                kspace,
                axis=2,
                fraction=0.60
            )

            print("Displaying the Homodyne reconstructed image using OrthoSlicer3D...")

            visualize_volume_slices(np.abs(homodyne_img))

            homodyne_fname = os.path.join(output_folder, '10mm', 'homodyne')
            if not os.path.exists(homodyne_fname):
                os.makedirs(homodyne_fname)
            homodyne_fname = os.path.join(homodyne_fname, f"homodyne_{base_name}.nii.gz")

            make_nifti(
                np.abs(homodyne_img),
                fname=homodyne_fname,
                mask=False,
                res=[
                    sample_data.res_dim1,
                    sample_data.res_dim2,
                    sample_data.res_dim3,
                ],
                dim_info=[0,1,2]
            )

            print("Saved:", homodyne_fname)

        if im is None:
            sample_data = []

        return im, sample_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
        
if __name__ == "__main__":
    im = read_lf_data()