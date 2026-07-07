import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Create a simple 3D volume
# Shape: (Y, X, Z) = (5, 5, 4)
# -------------------------------
vol = np.zeros((5, 5, 4), dtype=int)

# Center pixel indicates slice index for clarity
for z in range(4):
    vol[2, 2, z] = z + 1

# -------------------------------
# 2️⃣ Circular shifts using np.roll
# -------------------------------
vol_roll_z = np.roll(vol, -1, axis=2)  # shift slices (Z-axis)
vol_roll_x = np.roll(vol, 1, axis=1)   # shift horizontally (X-axis)
vol_roll_y = np.roll(vol, -1, axis=0)  # shift vertically (Y-axis)

# -------------------------------
# 3️⃣ Plot all slices for each circular shift
# -------------------------------
def plot_circular_shifts(volumes, volume_names):
    n_slices = volumes[0].shape[2]
    n_ops = len(volumes)
    
    plt.figure(figsize=(4*n_ops, 3*n_slices))
    
    for z in range(n_slices):
        for i, vol in enumerate(volumes):
            plt.subplot(n_slices, n_ops, z*n_ops + i + 1)
            plt.imshow(vol[:, :, z], cmap='gray', vmin=0, vmax=n_slices)
            if z == 0:
                plt.title(volume_names[i], fontsize=10)
            plt.axis('off')
            if i == 0:
                plt.ylabel(f"Slice {z}", fontsize=10)
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# 4️⃣ Call plotting function
# -------------------------------
volumes = [vol, vol_roll_z, vol_roll_x, vol_roll_y]
names = ["Original", "Roll Z (slices)", "Roll X (horizontal)", "Roll Y (vertical)"]

plot_circular_shifts(volumes, names)
