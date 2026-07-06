import os
import nibabel as nib
import matplotlib.pyplot as plt

folder_path = "Ajay_training/Output_nifti/34507"

nifty_files = [f for f in os.listdir(folder_path) if f.endswith('.nii') or f.endswith('.nii.gz')]

print("NIFTI files in '{}':".format(folder_path))
for file in nifty_files:
    print(file)
    file_path = os.path.join(folder_path, file)
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Get middle indices for each axis
    x_mid = data.shape[0] // 2  # sagittal
    y_mid = data.shape[1] // 2  # coronal
    z_mid = data.shape[2] // 2  # axial

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Axial (xy plane, slice along z)
    axes[0].imshow(data[:, :, z_mid].T, cmap='gray', origin='lower')
    axes[0].set_title('Axial')
    axes[0].axis('off')
    # Coronal (xz plane, slice along y)
    axes[1].imshow(data[:, y_mid, :].T, cmap='gray', origin='lower')
    axes[1].set_title('Coronal')
    axes[1].axis('off')
    # Sagittal (yz plane, slice along x)
    axes[2].imshow(data[x_mid, :, :].T, cmap='gray', origin='lower')
    axes[2].set_title('Sagittal')
    axes[2].axis('off')

    plt.suptitle(file)
    plt.show()