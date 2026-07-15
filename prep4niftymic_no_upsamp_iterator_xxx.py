# =============================================================================
# Native NumPy-to-NiftyMIC pipeline
# Goal:
#   Start from native 2 x 2 x 10 mm orthogonal NumPy volumes and create
#   meaningful NIfTI + mask inputs for NiftyMIC.
#
# Philosophy:
#   Validate one thing at a time:
#     array shape/intensity
#     orientation
#     physical affine
#     threshold/mask
#     image-mask consistency
#     pairwise overlap
#     NiftyMIC readiness
# =============================================================================
import os
import numpy as np
import nibabel as nib

# Module 1: Load native NumPy volumes and print sanity checks
# - Load axial, coronal, sagittal .npy files
# - Print shape, dtype, finite/NaN/Inf counts
# - Print intensity statistics and percentiles
# - Print candidate threshold voxel counts

# Module 2: Visualize raw central slices
# - Display central slices along array axes 0, 1, and 2
# - Save per-volume QC PNGs
# - Save combined 3 × 3 raw-axis QC PNG
# - Decide whether axis movement is needed

# Module 3: Apply explicit orientation recipes
# - Apply axis_order and flips for each stack
# - Current recipes are identity for all three stacks
# - Validate post-orientation central slices
# - Save combined post-orientation QC PNG

# Module 4: Create native stack-specific NIfTI affines
# - Convert fixed NumPy arrays to native NIfTI images
# - Assign stack-specific physical axes
# - Axial:    world_axes = ("R", "A", "S")
# - Coronal:  world_axes = ("R", "S", "A")
# - Sagittal: world_axes = ("A", "S", "R")
# - Set voxel sizes to 2 × 2 × 10 mm
# - Center all stacks at world coordinate [0, 0, 0]
# - Validate shape, zooms, affine, orientation, world extent

# Module 5: Apply selected threshold and create masks
# - Apply absolute threshold to each native NIfTI
# - Selected threshold: 40
# - Set background voxels to zero
# - Create binary masks
# - Validate image/mask shape, zooms, affine, dtype, and mask fraction
# - Save threshold/mask QC PNGs

# Module 5A: Threshold sweep
# - Test thresholds: 30, 40, 50, 60, 75, 100, 125, 150, 200
# - Print mask fraction, filled fraction, component count, largest connected component
# - Save threshold sweep QC PNGs
# - Select final working threshold

# Module 6: Geometry and overlap validation
# - Validate image/mask pairs in memory
# - Print geometry summary for each stack
# - Compute pairwise world-space bounding-box overlap
# - Resample masks to common 2 mm reference grid
# - Compute pairwise mask intersections, overlap fractions, and Dice scores
# - Decide whether inputs are safe to save for NiftyMIC

# Module 7: Save and reload validation
# - Save native NiftyMIC images and masks to disk
# - Reload all saved files
# - Check qform/sform codes, affines, zooms, orientation, dtype, nonzero voxels
# - Compare in-memory vs reloaded data
# - Run MD5 duplicate-file check
# - Re-run geometry and mask-overlap validation on reloaded files

# Module 8: Generate NiftyMIC commands and shell scripts
# - Generate geometry-debug NiftyMIC command
# - Generate one-cycle no-intensity-correction command
# - Write runnable shell scripts
# - Geometry-debug run uses:
#   - native 2 × 2 × 10 mm inputs
#   - 2 mm isotropic output
#   - intensity correction off
#   - two-step cycles off
#   - outlier rejection off

# Module 9: Evaluate geometry-debug reconstruction
# - Load geometry-debug SRR output
# - Print NIfTI geometry, affine, zooms, orientation, intensity stats
# - Display with OrthoSlicer3D for interactive browsing
# - Save static QC PNG of central axial/coronal/sagittal slices
# - Compare SRR world-space bounding box with input stacks
# - Optionally launch next NiftyMIC shell script from Python only after user confirms


# Expected voxel size:
#   in-plane: 2 mm
#   slice thickness: 10 mm
#
# Expected stacks:
#   axial:    thick axis should be superior-inferior
#   coronal:  thick axis should be anterior-posterior
#   sagittal: thick axis should be left-right
#
# Target NiftyMIC reconstruction:
#   isotropic-resolution = 2 mm

# =============================================================================
# Module 1 - Load native NumPy volumes and print basic sanity checks
# =============================================================================

# =============================================================================
# Matplotlib backend setup
# =============================================================================
#
# Put this near the very top of the script, BEFORE:
#   import matplotlib.pyplot as plt
#
# On macOS, avoid switching to QtAgg after macosx is already active.
# Prefer TkAgg if available; otherwise use MacOSX; otherwise fall back to Agg.
# =============================================================================

import matplotlib


def configure_matplotlib_backend(preferred=("TkAgg", "MacOSX", "QtAgg", "Agg")):
    """
    Configure a usable Matplotlib backend.

    Must be called before importing matplotlib.pyplot.

    On macOS, 'MacOSX' is often already active and works well.
    'TkAgg' is also fine if Python/Tk is installed.
    'QtAgg' requires PyQt/PySide and often fails in lightweight venvs.
    'Agg' is non-interactive but can still save PNGs.
    """
    current_backend = matplotlib.get_backend()
    print(f"Current matplotlib backend before configuration: {current_backend}")

    for backend in preferred:
        try:
            matplotlib.use(backend, force=True)
            print(f"Using matplotlib backend: {backend}")
            return backend
        except Exception as e:
            print(f"Could not use backend {backend}: {e}")

    print("WARNING: Falling back to Agg backend.")
    matplotlib.use("Agg", force=True)
    return "Agg"


MATPLOTLIB_BACKEND = configure_matplotlib_backend(
    preferred=("TkAgg", "MacOSX", "QtAgg", "Agg")
)

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# User paths
# -----------------------------------------------------------------------------

BASE_DIR = "DataSRR/volunteer_xxx/10mm/npy"
# -----------------------------------------------------------------------------
# Native acquisition geometry
# -----------------------------------------------------------------------------

IN_PLANE_MM = 2.0
SLICE_THICKNESS_MM = 10.0

axial_path = os.path.join(BASE_DIR, "axial_circshift_1zy.npy")

# Add these once you confirm the exact filenames
coronal_path = os.path.join(BASE_DIR, "coronal_circshift_1zx.npy")
sagittal_path = os.path.join(BASE_DIR, "sagittal_circshift_1yx.npy")

NIFTYMIC_INPUT_DIR = os.path.join(BASE_DIR, "native_niftymic_inputs_thr40")
# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_numpy_volume(path, label="volume"):
    """
    Load a NumPy volume and print basic metadata.
    """
    print("")
    print("=" * 80)
    print(f"Loading {label}")
    print("=" * 80)
    print("Path:", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {label} file: {path}")

    arr = np.load(path)

    print("Loaded successfully.")
    print("Shape:", arr.shape)
    print("ndim:", arr.ndim)
    print("dtype:", arr.dtype)
    print("size:", arr.size)

    return arr


def print_volume_stats(arr, label="volume"):
    """
    Print robust intensity and occupancy statistics for a 3D NumPy volume.
    """
    arr = np.asarray(arr)

    print("")
    print(f"--- Stats: {label} ---")

    if arr.ndim != 3:
        print("WARNING: Expected 3D array, but got ndim =", arr.ndim)

    finite_mask = np.isfinite(arr)
    finite_vals = arr[finite_mask]

    print("finite voxels:", finite_vals.size, "/", arr.size)
    print("nan voxels:", np.count_nonzero(np.isnan(arr)))
    print("inf voxels:", np.count_nonzero(np.isinf(arr)))

    if finite_vals.size == 0:
        print("ERROR: No finite values found.")
        return

    nonzero_vals = finite_vals[finite_vals != 0]

    print("min:", np.min(finite_vals))
    print("max:", np.max(finite_vals))
    print("mean:", np.mean(finite_vals))
    print("std:", np.std(finite_vals))
    print("nonzero voxels:", nonzero_vals.size)
    print("nonzero fraction:", nonzero_vals.size / arr.size)

    percentiles = [0, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 100]
    pvals = np.percentile(finite_vals, percentiles)

    print("")
    print("Percentiles, all finite voxels:")
    for p, v in zip(percentiles, pvals):
        print(f"  p{p:>5}: {v:.3f}")

    if nonzero_vals.size > 0:
        pvals_nz = np.percentile(nonzero_vals, percentiles)

        print("")
        print("Percentiles, nonzero finite voxels:")
        for p, v in zip(percentiles, pvals_nz):
            print(f"  p{p:>5}: {v:.3f}")

    # Threshold-specific quick check for your planned background threshold
    threshold = 40
    above_thr = finite_vals > threshold

    print("")
    print(f"Voxels > {threshold}: {np.count_nonzero(above_thr)}")
    print(f"Fraction > {threshold}: {np.count_nonzero(above_thr) / arr.size:.6f}")


def load_and_inspect_numpy_volume(path, label="volume"):
    """
    Convenience wrapper.
    """
    arr = load_numpy_volume(path, label=label)
    print_volume_stats(arr, label=label)
    return arr


# -----------------------------------------------------------------------------
# Load currently available arrays
# -----------------------------------------------------------------------------

axial_np = load_and_inspect_numpy_volume(axial_path, label="axial")
coronal_np = load_and_inspect_numpy_volume(coronal_path, label="coronal")
sagittal_np = load_and_inspect_numpy_volume(sagittal_path, label="sagittal")


# -----------------------------------------------------------------------------
# Expected acquisition note
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("Module 1 complete")
print("=" * 80)
print("Next module will visualize central slices along array axes 0, 1, and 2.")
print("No orientation, affine, thresholding, or saving has been performed yet.")


# =============================================================================
# Module 2 - Visualize raw NumPy volumes along all three array axes
# =============================================================================
#
# Purpose:
#   For each raw NumPy volume, show central slices along array axis 0, 1, and 2.
#
# We are NOT reorienting yet.
# We are NOT thresholding yet.
# We are NOT creating NIfTI affines yet.
#
# This module helps determine:
#   1. Which displayed panel corresponds to the anatomical in-plane view.
#   2. Whether the brain/skull is upside down.
#   3. Which axis should become the thick-slice axis.
#   4. What axis_order/flips we need in Module 3.
# =============================================================================




# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

BASE_DIR = "DataSRR/volunteer_xxx/10mm/npy"

axial_path = os.path.join(BASE_DIR, "axial_circshift_1zy.npy")
coronal_path = os.path.join(BASE_DIR, "coronal_circshift_1zx.npy")
sagittal_path = os.path.join(BASE_DIR, "sagittal_circshift_1yx.npy")

OUTPUT_DIR = os.path.join(BASE_DIR, "module2_qc")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Load arrays
# -----------------------------------------------------------------------------

axial_np = np.load(axial_path)
coronal_np = np.load(coronal_path)
sagittal_np = np.load(sagittal_path)

volumes = {
    "axial_raw": axial_np,
    "coronal_raw": coronal_np,
    "sagittal_raw": sagittal_np,
}


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def get_central_slices(arr):
    """
    Return central slices along array axis 0, axis 1, and axis 2.

    For arr shape (X, Y, Z):
        axis 0 slice = arr[Xmid, :, :]
        axis 1 slice = arr[:, Ymid, :]
        axis 2 slice = arr[:, :, Zmid]
    """
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    cx, cy, cz = [s // 2 for s in arr.shape]

    slices = {
        "axis0_mid": arr[cx, :, :],
        "axis1_mid": arr[:, cy, :],
        "axis2_mid": arr[:, :, cz],
    }

    indices = {
        "axis0_mid": cx,
        "axis1_mid": cy,
        "axis2_mid": cz,
    }

    return slices, indices


def display_slice_for_human_view(slice_2d, rotate_k=1, flip_ud=False, flip_lr=False):
    """
    Apply display-only transformations for visualization.

    These do not modify the underlying array.
    They only make matplotlib display easier to interpret.

    rotate_k=1 uses np.rot90 once, which often makes array slices look
    closer to radiological viewer convention.
    """
    view = np.asarray(slice_2d)

    if rotate_k is not None and rotate_k != 0:
        view = np.rot90(view, k=rotate_k)

    if flip_ud:
        view = np.flipud(view)

    if flip_lr:
        view = np.fliplr(view)

    return view


def robust_clim(arr, low=1, high=99.5):
    """
    Robust display limits.
    """
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0, 1

    vmin, vmax = np.percentile(vals, [low, high])
    return float(vmin), float(vmax)


def plot_axis_slices(
    arr,
    label,
    clim=None,
    rotate_k=1,
    save_path=None,
    show=True,
):
    """
    Plot central slices along all three array axes for one volume.
    """
    slices, indices = get_central_slices(arr)

    if clim is None:
        clim = robust_clim(arr, low=1, high=99.5)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"{label} | shape={arr.shape} | display only, no reorientation applied",
        fontsize=12,
    )

    for ax, key in zip(axes, ["axis0_mid", "axis1_mid", "axis2_mid"]):
        view = display_slice_for_human_view(
            slices[key],
            rotate_k=rotate_k,
            flip_ud=False,
            flip_lr=False,
        )

        ax.imshow(view, cmap="gray", vmin=clim[0], vmax=clim[1])
        ax.set_title(f"{key}\nindex={indices[key]}, raw slice shape={slices[key].shape}")
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_volumes_axis_slices(
    volumes,
    save_path=None,
    rotate_k=1,
    show=True,
):
    """
    Plot all three volumes in a 3 x 3 grid.

    Rows:
        axial_raw
        coronal_raw
        sagittal_raw

    Columns:
        central slice along axis 0
        central slice along axis 1
        central slice along axis 2
    """
    names = list(volumes.keys())

    fig, axes = plt.subplots(len(names), 3, figsize=(13, 4 * len(names)))

    if len(names) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        "Module 2 QC: raw central slices along array axes\n"
        "Rows = volumes, Columns = axis0 / axis1 / axis2 central slices",
        fontsize=14,
    )

    for row, name in enumerate(names):
        arr = volumes[name]
        slices, indices = get_central_slices(arr)
        clim = robust_clim(arr, low=1, high=99.5)

        for col, key in enumerate(["axis0_mid", "axis1_mid", "axis2_mid"]):
            view = display_slice_for_human_view(
                slices[key],
                rotate_k=rotate_k,
                flip_ud=False,
                flip_lr=False,
            )

            axes[row, col].imshow(view, cmap="gray", vmin=clim[0], vmax=clim[1])
            axes[row, col].axis("off")

            axes[row, col].set_title(
                f"{name}\n{key}, idx={indices[key]}\n"
                f"raw slice={slices[key].shape}"
            )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Run Module 2 visualization
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("Module 2 - Raw central-slice visualization")
print("=" * 80)

for name, arr in volumes.items():
    print("")
    print(f"{name}")
    print("shape:", arr.shape)
    print("central indices:", tuple(s // 2 for s in arr.shape))

    out_png = os.path.join(OUTPUT_DIR, f"{name}_axis_slices.png")

    plot_axis_slices(
        arr,
        label=name,
        clim=None,
        rotate_k=1,
        save_path=out_png,
        show=True,
    )


combined_png = os.path.join(OUTPUT_DIR, "all_raw_axis_slices_3x3.png")

plot_all_volumes_axis_slices(
    volumes,
    save_path=combined_png,
    rotate_k=1,
    show=True,
)


print("")
print("=" * 80)
print("Module 2 complete")
print("=" * 80)
print("QC images saved in:")
print(OUTPUT_DIR)
print("")
print("Next step: inspect the 3 x 3 figure and decide Module 3 orientation recipes:")
print("  axial:    axis_order=?, flips=?")
print("  coronal:  axis_order=?, flips=?")
print("  sagittal: axis_order=?, flips=?")

# =============================================================================
# Module 3 - Apply explicit array orientation recipes and validate visually
# =============================================================================
#
# Purpose:
#   Apply only array-level operations:
#       np.transpose
#       optional np.flip
#
# Current decision from Module 2:
#   All three raw arrays are shape 110 x 110 x 16.
#   All three have their best anatomical in-plane view at axis2_mid.
#   Therefore, start with no axis movement.
#
# We are NOT creating NIfTI affines yet.
# We are NOT thresholding yet.
# We are NOT saving final NiftyMIC files yet.
# =============================================================================




# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

BASE_DIR = "DataSRR/volunteer_xxx/10mm/npy"

axial_path = os.path.join(BASE_DIR, "axial_circshift_1zy.npy")
coronal_path = os.path.join(BASE_DIR, "coronal_circshift_1zx.npy")
sagittal_path = os.path.join(BASE_DIR, "sagittal_circshift_1yx.npy")

OUTPUT_DIR = os.path.join(BASE_DIR, "module3_qc")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Load raw arrays
# -----------------------------------------------------------------------------

raw_volumes = {
    "axial": np.load(axial_path),
    "coronal": np.load(coronal_path),
    "sagittal": np.load(sagittal_path),
}


# -----------------------------------------------------------------------------
# Orientation recipes
# -----------------------------------------------------------------------------
#
# axis_order:
#   Passed to np.transpose(arr, axes=axis_order)
#
# flips:
#   Applied after transpose.
#   flips=(True, False, False) means flip axis 0.
#
# From Module 2, start with identity for all three.

ORIENTATION_CONFIG = {
    "axial": {
        "axis_order": (0, 1, 2),
        "flips": (False, False, False),
    },
    "coronal": {
        "axis_order": (0, 1, 2),
        "flips": (False, False, False),
    },
    "sagittal": {
        "axis_order": (0, 1, 2),
        "flips": (False, False, False),
    },
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def apply_orientation_recipe(arr, axis_order=(0, 1, 2), flips=(False, False, False), label="volume"):
    """
    Apply explicit array orientation recipe.

    This does not know anything about anatomy or NIfTI affines.
    It only permutes and flips the NumPy array.
    """
    arr = np.asarray(arr)

    print("")
    print("=" * 80)
    print(f"Applying orientation recipe: {label}")
    print("=" * 80)
    print("Input shape:", arr.shape)
    print("axis_order:", axis_order)
    print("flips:", flips)

    if arr.ndim != 3:
        raise ValueError(f"{label}: expected 3D array, got shape {arr.shape}")

    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError(f"{label}: axis_order must be a permutation of (0, 1, 2)")

    if len(flips) != 3:
        raise ValueError(f"{label}: flips must have length 3")

    out = np.transpose(arr, axes=axis_order)

    for ax, do_flip in enumerate(flips):
        if do_flip:
            out = np.flip(out, axis=ax)

    out = np.ascontiguousarray(out.astype(np.float32))

    print("Output shape:", out.shape)

    return out


def robust_clim(arr, low=1, high=99.5):
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0, 1

    return tuple(np.percentile(vals, [low, high]).astype(float))


def get_central_slices(arr):
    arr = np.asarray(arr)
    cx, cy, cz = [s // 2 for s in arr.shape]

    slices = {
        "axis0_mid": arr[cx, :, :],
        "axis1_mid": arr[:, cy, :],
        "axis2_mid": arr[:, :, cz],
    }

    indices = {
        "axis0_mid": cx,
        "axis1_mid": cy,
        "axis2_mid": cz,
    }

    return slices, indices


def display_slice(slice_2d, rotate_k=1):
    """
    Display-only rotation for matplotlib.
    Does not affect saved arrays.
    """
    return np.rot90(slice_2d, k=rotate_k)


def plot_all_fixed_axis_slices(volumes, save_path=None, rotate_k=1, show=True):
    """
    Plot all fixed volumes in a 3 x 3 grid.
    """
    names = list(volumes.keys())

    fig, axes = plt.subplots(len(names), 3, figsize=(13, 4 * len(names)))

    if len(names) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        "Module 3 QC: post-orientation central slices\n"
        "Rows = volumes, Columns = axis0 / axis1 / axis2 central slices",
        fontsize=14,
    )

    for row, name in enumerate(names):
        arr = volumes[name]
        slices, indices = get_central_slices(arr)
        clim = robust_clim(arr, low=1, high=99.5)

        for col, key in enumerate(["axis0_mid", "axis1_mid", "axis2_mid"]):
            view = display_slice(slices[key], rotate_k=rotate_k)

            axes[row, col].imshow(view, cmap="gray", vmin=clim[0], vmax=clim[1])
            axes[row, col].axis("off")
            axes[row, col].set_title(
                f"{name}_fixed\n{key}, idx={indices[key]}\n"
                f"raw slice={slices[key].shape}"
            )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_fixed_summary(volumes):
    print("")
    print("=" * 80)
    print("Module 3 fixed-array summary")
    print("=" * 80)

    for name, arr in volumes.items():
        print("")
        print(f"{name}_fixed")
        print("shape:", arr.shape)
        print("dtype:", arr.dtype)
        print("min/max/mean:", np.min(arr), np.max(arr), np.mean(arr))
        print("voxels > 200:", np.count_nonzero(arr > 200))

import SimpleITK as sitk
import numpy as np

def n4_bias_correct_numpy(volume, mask=None):
    img = sitk.GetImageFromArray(volume.astype(np.float32))

    if mask is None:
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
    else:
        mask = sitk.GetImageFromArray(mask.astype(np.uint8))

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask)

    return sitk.GetArrayFromImage(corrected)

# -----------------------------------------------------------------------------
# Run Module 3
# -----------------------------------------------------------------------------

fixed_volumes = {}

for name, arr in raw_volumes.items():
    cfg = ORIENTATION_CONFIG[name]

    fixed_volumes[name] = apply_orientation_recipe(
        arr,
        axis_order=cfg["axis_order"],
        flips=cfg["flips"],
        label=name,
    )


# Apply N4 bias field correction to each fixed volume;
# this is optional and can be commented if not desired.

# corrected_volumes = {}

# for name, arr in fixed_volumes.items():
#     print(f"Running N4 for {name}...")
#     corrected_volumes[name] = n4_bias_correct_numpy(arr)

# fixed_volumes = corrected_volumes

print_fixed_summary(fixed_volumes)

combined_png = os.path.join(OUTPUT_DIR, "all_fixed_axis_slices_3x3.png")

plot_all_fixed_axis_slices(
    fixed_volumes,
    save_path=combined_png,
    rotate_k=1,
    show=True,
)

print("")
print("=" * 80)
print("Module 3 complete")
print("=" * 80)
print("QC image saved at:")
print(combined_png)
print("")
print("Current orientation recipes:")
for name, cfg in ORIENTATION_CONFIG.items():
    print(f"{name}: axis_order={cfg['axis_order']}, flips={cfg['flips']}")
print("")
print("Next module: create stack-specific native affines.")

# =============================================================================
# Module 4 - Create native stack-specific NIfTI images with explicit affines
# =============================================================================
#
# Purpose:
#   Convert fixed NumPy arrays into NIfTI images with correct physical geometry.
#
# We are NOT thresholding yet.
# We are NOT creating masks yet.
# We are NOT saving final NiftyMIC inputs yet.
#
# Current fixed arrays:
#   axial_fixed:    shape 110 x 110 x 16
#   coronal_fixed:  shape 110 x 110 x 16
#   sagittal_fixed: shape 110 x 110 x 16
#
# Native voxel size:
#   in-plane:        2 mm
#   slice thickness: 10 mm
#
# Physical stack meaning:
#   axial:
#       axis 0 = R/L
#       axis 1 = A/P
#       axis 2 = S/I thick slice
#
#   coronal:
#       axis 0 = R/L
#       axis 1 = S/I
#       axis 2 = A/P thick slice
#
#   sagittal:
#       axis 0 = A/P
#       axis 1 = S/I
#       axis 2 = R/L thick slice
# =============================================================================






# -----------------------------------------------------------------------------
# Stack-specific affine definitions
# -----------------------------------------------------------------------------

STACK_GEOMETRY = {
    "axial": {
        "world_axes": ("R", "A", "S"),
        "voxel_sizes": (IN_PLANE_MM, IN_PLANE_MM, SLICE_THICKNESS_MM),
    },

    "coronal": {
        "world_axes": ("R", "S", "A"),
        "voxel_sizes": (IN_PLANE_MM, IN_PLANE_MM, SLICE_THICKNESS_MM),
    },

    "sagittal": {
        "world_axes": ("A", "S", "R"),
        "voxel_sizes": (IN_PLANE_MM, IN_PLANE_MM, SLICE_THICKNESS_MM),
    },
}


# -----------------------------------------------------------------------------
# Affine utilities
# -----------------------------------------------------------------------------

def anatomical_axis_to_vector(axis_label):
    """
    Convert an anatomical axis label into a RAS-world direction vector.

    NIfTI convention here:
        +x = R
        +y = A
        +z = S

    Supported labels:
        R, L, A, P, S, I
    """
    axis_to_vector = {
        "R": np.array([ 1.0,  0.0,  0.0]),
        "L": np.array([-1.0,  0.0,  0.0]),

        "A": np.array([ 0.0,  1.0,  0.0]),
        "P": np.array([ 0.0, -1.0,  0.0]),

        "S": np.array([ 0.0,  0.0,  1.0]),
        "I": np.array([ 0.0,  0.0, -1.0]),
    }

    if axis_label not in axis_to_vector:
        raise ValueError(f"Unknown anatomical axis label: {axis_label}")

    return axis_to_vector[axis_label]


def make_centered_stack_affine(shape, world_axes, voxel_sizes):
    """
    Create a centered NIfTI affine for a native thick-slice stack.

    Parameters
    ----------
    shape : tuple
        Array shape, e.g. (110, 110, 16).

    world_axes : tuple[str, str, str]
        Physical direction of array axes in RAS space.

        Example:
            axial    = ("R", "A", "S")
            coronal  = ("R", "S", "A")
            sagittal = ("A", "S", "R")

    voxel_sizes : tuple[float, float, float]
        Voxel size along each array axis.

        Example:
            (2.0, 2.0, 10.0)

    Returns
    -------
    affine : np.ndarray
        4 x 4 affine mapping voxel indices to RAS mm.
    """
    shape = np.asarray(shape[:3], dtype=float)
    voxel_sizes = np.asarray(voxel_sizes, dtype=float)

    if len(world_axes) != 3:
        raise ValueError("world_axes must contain exactly 3 entries.")

    if len(voxel_sizes) != 3:
        raise ValueError("voxel_sizes must contain exactly 3 entries.")

    affine = np.eye(4, dtype=float)

    for array_axis in range(3):
        direction = anatomical_axis_to_vector(world_axes[array_axis])
        affine[:3, array_axis] = direction * voxel_sizes[array_axis]

    # Center volume around RAS coordinate [0, 0, 0].
    center_voxel = 0.5 * (shape - 1.0)
    affine[:3, 3] = -affine[:3, :3] @ center_voxel

    return affine


def make_native_nifti_from_array(arr, world_axes, voxel_sizes, dtype=np.float32):
    """
    Convert a NumPy array into a native-stack NIfTI image.
    """
    arr = np.asarray(arr).astype(dtype)

    affine = make_centered_stack_affine(
        shape=arr.shape,
        world_axes=world_axes,
        voxel_sizes=voxel_sizes,
    )

    img = nib.Nifti1Image(arr, affine)
    img.header.set_data_dtype(dtype)
    img.header.set_zooms(tuple(voxel_sizes))

    # Explicitly set both qform and sform.
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)

    return img

def image_world_corners(img):
    """
    Return the 8 world-space corners of a NIfTI image.
    """
    shape = np.array(img.shape[:3], dtype=int)

    corners_ijk = np.array([
        [0, 0, 0],
        [shape[0] - 1, 0, 0],
        [0, shape[1] - 1, 0],
        [0, 0, shape[2] - 1],
        [shape[0] - 1, shape[1] - 1, 0],
        [shape[0] - 1, 0, shape[2] - 1],
        [0, shape[1] - 1, shape[2] - 1],
        [shape[0] - 1, shape[1] - 1, shape[2] - 1],
    ], dtype=float)

    corners_h = np.c_[corners_ijk, np.ones(8)]
    corners_xyz = (img.affine @ corners_h.T).T[:, :3]

    return corners_xyz


def image_world_bbox(img):
    """
    Return world-space bounding box min/max.
    """
    corners = image_world_corners(img)
    return corners.min(axis=0), corners.max(axis=0)


def inspect_native_nifti(img, label="image"):
    """
    Print shape, zooms, affine, orientation, and world-space extent.
    """
    data = img.get_fdata()
    bmin, bmax = image_world_bbox(img)
    center = 0.5 * (bmin + bmax)
    extent = bmax - bmin

    print("")
    print("=" * 80)
    print(f"Inspect native NIfTI: {label}")
    print("=" * 80)

    print("shape:", img.shape[:3])
    print("dtype:", img.get_data_dtype())
    print("header zooms:", img.header.get_zooms()[:3])
    print("orientation codes:", nib.aff2axcodes(img.affine))

    print("")
    print("affine:")
    print(img.affine)

    print("")
    print("world min:", bmin)
    print("world max:", bmax)
    print("world center:", center)
    print("world extent mm:", extent)

    print("")
    print("intensity min/max/mean:", np.min(data), np.max(data), np.mean(data))
    print("nonzero voxels:", np.count_nonzero(data))

    # Sanity checks
    if not np.allclose(center, np.zeros(3), atol=1e-5):
        print("WARNING: world center is not close to [0, 0, 0].")

    expected_extent = np.array(img.shape[:3], dtype=float) - 1
    expected_extent = expected_extent * np.array(img.header.get_zooms()[:3])

    if not np.allclose(np.sort(extent), np.sort(expected_extent), atol=1e-5):
        print("WARNING: world extent does not match expected voxel geometry.")


# -----------------------------------------------------------------------------
# Run Module 4
# -----------------------------------------------------------------------------
#
# This assumes Module 3 has already created:
#
#   fixed_volumes["axial"]
#   fixed_volumes["coronal"]
#   fixed_volumes["sagittal"]
#
# If running Module 4 standalone, uncomment the fallback loader below.

# -----------------------------------------------------------------------------
# Optional standalone fallback loader
# -----------------------------------------------------------------------------
# BASE_DIR = "DataSRR/ajay_scan_2/10mm/npy"
# fixed_volumes = {
#     "axial": np.load(os.path.join(BASE_DIR, "axial_circshift_1zy.npy")).astype(np.float32),
#     "coronal": np.load(os.path.join(BASE_DIR, "coronal_circshift_1zx.npy")).astype(np.float32),
#     "sagittal": np.load(os.path.join(BASE_DIR, "sagittal_circshift_1yx.npy")).astype(np.float32),
# }

native_imgs = {}

for name, arr in fixed_volumes.items():
    geom = STACK_GEOMETRY[name]

    native_imgs[name] = make_native_nifti_from_array(
        arr,
        world_axes=geom["world_axes"],
        voxel_sizes=geom["voxel_sizes"],
        dtype=np.float32,
    )

    inspect_native_nifti(native_imgs[name], label=name)


print("")
print("=" * 80)
print("Module 4 complete")
print("=" * 80)

print("")
print("Created native NIfTI images in memory:")
for name, img in native_imgs.items():
    print(f"{name}: shape={img.shape}, zooms={img.header.get_zooms()[:3]}, orientation={nib.aff2axcodes(img.affine)}")

print("")
print("No thresholding, masks, saving, or NiftyMIC input writing has been performed yet.")
print("Next module: threshold at 200 and create simple masks.")

# =============================================================================
# Module 5 - Threshold native NIfTI images and create simple masks
# =============================================================================
#
# Purpose:
#   Create NiftyMIC-friendly images and masks from the native-stack NIfTIs.
#
# Input from Module 4:
#   native_imgs["axial"]
#   native_imgs["coronal"]
#   native_imgs["sagittal"]
#
# Output:
#   thresholded_imgs["axial"]
#   thresholded_imgs["coronal"]
#   thresholded_imgs["sagittal"]
#
#   masks["axial"]
#   masks["coronal"]
#   masks["sagittal"]
#
# Current strategy:
#   mask = image > 200
#   thresholded_image = image with background set to 0
#
# Important:
#   For the first NiftyMIC test, we avoid aggressive morphology.
#   A slightly loose mask is safer than a tight mask that clips anatomy.
# =============================================================================

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import (
    binary_fill_holes,
    binary_opening,
    binary_closing,
    binary_dilation,
    binary_erosion,
    label,
)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

THRESHOLD = 65.0

USE_MASK_CLEANUP = False

# These are only used if USE_MASK_CLEANUP = True
MASK_KEEP_LARGEST_COMPONENT = True
MASK_FILL_HOLES = True
MASK_OPENING_ITER = 0
MASK_CLOSING_ITER = 2
MASK_DILATION_ITER = 0
MASK_EROSION_ITER = 0

BASE_DIR = "DataSRR/volunteer_xxx/10mm/npy"
OUTPUT_DIR = os.path.join(BASE_DIR, "module5_qc")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Mask utilities
# -----------------------------------------------------------------------------

def largest_connected_component(mask):
    """
    Keep the largest connected component of a binary mask.
    """
    mask = mask.astype(bool)

    labeled, nlab = label(mask)
    if nlab == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    largest_label = np.argmax(counts)
    return labeled == largest_label


def cleanup_binary_mask(
    mask,
    keep_largest=True,
    fill_holes=True,
    opening_iter=0,
    closing_iter=1,
    dilation_iter=0,
    erosion_iter=0,
):
    """
    Optional light mask cleanup.
    For the first NiftyMIC run, keep USE_MASK_CLEANUP=False.
    """
    mask = mask.astype(bool)

    if fill_holes:
        mask = binary_fill_holes(mask)

    if opening_iter > 0:
        mask = binary_opening(mask, iterations=opening_iter)

    if closing_iter > 0:
        mask = binary_closing(mask, iterations=closing_iter)

    if keep_largest:
        mask = largest_connected_component(mask)

    if erosion_iter > 0:
        mask = binary_erosion(mask, iterations=erosion_iter)

    if dilation_iter > 0:
        mask = binary_dilation(mask, iterations=dilation_iter)

    return mask.astype(bool)


def apply_absolute_threshold_to_nifti(
    img,
    threshold=40.0,
    cleanup=False,
    dtype=np.float32,
):
    """
    Apply an absolute threshold to a NIfTI image.

    Returns
    -------
    thresholded_img : nib.Nifti1Image
        Image with voxels <= threshold set to 0.

    mask_img : nib.Nifti1Image
        Binary uint8 mask where original image > threshold.
    """
    data = img.get_fdata().astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    mask = data > float(threshold)

    if cleanup:
        mask = cleanup_binary_mask(
            mask,
            keep_largest=MASK_KEEP_LARGEST_COMPONENT,
            fill_holes=MASK_FILL_HOLES,
            opening_iter=MASK_OPENING_ITER,
            closing_iter=MASK_CLOSING_ITER,
            dilation_iter=MASK_DILATION_ITER,
            erosion_iter=MASK_EROSION_ITER,
        )

    thresholded = data.copy()
    thresholded[~mask] = 0
    thresholded = thresholded.astype(dtype)

    # Image header
    img_header = img.header.copy()
    img_header.set_data_dtype(dtype)

    thresholded_img = nib.Nifti1Image(
        thresholded,
        img.affine.copy(),
        img_header,
    )
    thresholded_img.set_qform(img.affine.copy(), code=1)
    thresholded_img.set_sform(img.affine.copy(), code=1)

    # Mask header
    mask_header = img.header.copy()
    mask_header.set_data_dtype(np.uint8)

    mask_img = nib.Nifti1Image(
        mask.astype(np.uint8),
        img.affine.copy(),
        mask_header,
    )
    mask_img.set_qform(img.affine.copy(), code=1)
    mask_img.set_sform(img.affine.copy(), code=1)

    return thresholded_img, mask_img

# -----------------------------------------------------------------------------
# Validation utilities
# -----------------------------------------------------------------------------
def validate_image_mask_pair(img, mask_img, label="volume"):
    """
    Validate that image and mask are compatible.
    """
    img_data = img.get_fdata()
    mask_data = mask_img.get_fdata()

    mask_unique = np.unique(mask_data)

    print("")
    print("=" * 80)
    print(f"Validate image/mask pair: {label}")
    print("=" * 80)

    print("image shape:", img.shape[:3])
    print("mask shape: ", mask_img.shape[:3])
    print("same shape:", img.shape[:3] == mask_img.shape[:3])

    print("image zooms:", img.header.get_zooms()[:3])
    print("mask zooms: ", mask_img.header.get_zooms()[:3])
    print("same zooms:", np.allclose(img.header.get_zooms()[:3], mask_img.header.get_zooms()[:3]))

    print("same affine:", np.allclose(img.affine, mask_img.affine))

    print("image dtype:", img.get_data_dtype())
    print("mask dtype: ", mask_img.get_data_dtype())

    print("mask unique values:", mask_unique[:10])
    print("mask is binary:", np.all(np.isin(mask_unique, [0, 1])))

    print("image nonzero voxels:", np.count_nonzero(img_data))
    print("mask nonzero voxels: ", np.count_nonzero(mask_data))
    print("mask fraction:", np.count_nonzero(mask_data) / mask_data.size)

    print("image min/max/mean:", np.min(img_data), np.max(img_data), np.mean(img_data))

    if img.shape[:3] != mask_img.shape[:3]:
        print("ERROR: image and mask shapes do not match.")

    if not np.allclose(img.affine, mask_img.affine):
        print("ERROR: image and mask affines do not match.")

    if np.count_nonzero(mask_data) == 0:
        print("ERROR: mask is empty.")

    if not np.all(np.isin(mask_unique, [0, 1])):
        print("ERROR: mask is not binary.")


def print_threshold_summary(original_img, thresholded_img, mask_img, label="volume"):
    """
    Print threshold effect.
    """
    orig = original_img.get_fdata()
    thr = thresholded_img.get_fdata()
    mask = mask_img.get_fdata() > 0

    print("")
    print(f"--- Threshold summary: {label} ---")
    print("threshold:", THRESHOLD)
    print("original nonzero:", np.count_nonzero(orig))
    print("thresholded nonzero:", np.count_nonzero(thr))
    print("mask nonzero:", np.count_nonzero(mask))
    print("fraction retained:", np.count_nonzero(mask) / mask.size)

    if np.count_nonzero(mask) > 0:
        vals_inside = orig[mask]
        print("inside-mask original min/max/mean:", vals_inside.min(), vals_inside.max(), vals_inside.mean())

    vals_outside = orig[~mask]
    if vals_outside.size > 0:
        print("outside-mask original min/max/mean:", vals_outside.min(), vals_outside.max(), vals_outside.mean())


# -----------------------------------------------------------------------------
# Visualization utilities
# -----------------------------------------------------------------------------

def robust_clim(arr, low=1, high=99.5):
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0, 1

    return tuple(np.percentile(vals, [low, high]).astype(float))


def get_central_slices(arr):
    arr = np.asarray(arr)
    cx, cy, cz = [s // 2 for s in arr.shape]

    slices = {
        "axis0_mid": arr[cx, :, :],
        "axis1_mid": arr[:, cy, :],
        "axis2_mid": arr[:, :, cz],
    }

    indices = {
        "axis0_mid": cx,
        "axis1_mid": cy,
        "axis2_mid": cz,
    }

    return slices, indices


def display_slice(slice_2d, rotate_k=1):
    return np.rot90(slice_2d, k=rotate_k)


def plot_thresholded_and_mask(
    thresholded_imgs,
    masks,
    save_path=None,
    rotate_k=1,
    show=True,
):
    """
    Plot central in-plane slice and corresponding mask for each stack.

    Since all current stacks are 110 x 110 x 16, the in-plane anatomical view is axis2_mid.
    """
    names = list(thresholded_imgs.keys())

    fig, axes = plt.subplots(len(names), 2, figsize=(8, 4 * len(names)))

    if len(names) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        f"Module 5 QC: thresholded images and masks | threshold={THRESHOLD}",
        fontsize=14,
    )

    for row, name in enumerate(names):
        img_data = thresholded_imgs[name].get_fdata()
        mask_data = masks[name].get_fdata()

        img_slices, idx = get_central_slices(img_data)
        mask_slices, _ = get_central_slices(mask_data)

        key = "axis2_mid"

        img_view = display_slice(img_slices[key], rotate_k=rotate_k)
        mask_view = display_slice(mask_slices[key], rotate_k=rotate_k)

        clim = robust_clim(img_data[img_data > 0], low=1, high=99.5) if np.any(img_data > 0) else (0, 1)

        axes[row, 0].imshow(img_view, cmap="gray", vmin=clim[0], vmax=clim[1])
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{name} thresholded\n{key}, idx={idx[key]}")

        axes[row, 1].imshow(mask_view, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")
        axes[row, 1].set_title(f"{name} mask\n{key}, idx={idx[key]}")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_mask_axis_slices(
    masks,
    save_path=None,
    rotate_k=1,
    show=True,
):
    """
    Plot central slices along all three axes for each mask.
    """
    names = list(masks.keys())

    fig, axes = plt.subplots(len(names), 3, figsize=(13, 4 * len(names)))

    if len(names) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        f"Module 5 QC: mask central slices along all axes | threshold={THRESHOLD}",
        fontsize=14,
    )

    for row, name in enumerate(names):
        mask_data = masks[name].get_fdata()
        slices, indices = get_central_slices(mask_data)

        for col, key in enumerate(["axis0_mid", "axis1_mid", "axis2_mid"]):
            view = display_slice(slices[key], rotate_k=rotate_k)

            axes[row, col].imshow(view, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{name} mask\n{key}, idx={indices[key]}")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Run Module 5
# -----------------------------------------------------------------------------
#
# This assumes Module 4 has already created:
#   native_imgs["axial"]
#   native_imgs["coronal"]
#   native_imgs["sagittal"]

thresholded_imgs = {}
masks = {}

print("")
print("=" * 80)
print("Module 5 - Threshold images and create masks")
print("=" * 80)
print("Threshold:", THRESHOLD)
print("Use mask cleanup:", USE_MASK_CLEANUP)

for name, img in native_imgs.items():
    thresholded_img, mask_img = apply_absolute_threshold_to_nifti(
        img,
        threshold=THRESHOLD,
        cleanup=USE_MASK_CLEANUP,
        dtype=np.float32,
    )

    thresholded_imgs[name] = thresholded_img
    masks[name] = mask_img

    print_threshold_summary(
        original_img=img,
        thresholded_img=thresholded_img,
        mask_img=mask_img,
        label=name,
    )

    validate_image_mask_pair(
        img=thresholded_img,
        mask_img=mask_img,
        label=name,
    )


# -----------------------------------------------------------------------------
# QC plots
# -----------------------------------------------------------------------------

qc_png = os.path.join(OUTPUT_DIR, "thresholded_images_and_masks_axis2.png")
mask_axes_png = os.path.join(OUTPUT_DIR, "all_mask_axis_slices_3x3.png")

plot_thresholded_and_mask(
    thresholded_imgs,
    masks,
    save_path=qc_png,
    rotate_k=1,
    show=True,
)

plot_all_mask_axis_slices(
    masks,
    save_path=mask_axes_png,
    rotate_k=1,
    show=True,
)


print("")
print("=" * 80)
print("Module 5 complete")
print("=" * 80)
print("QC images saved:")
print(qc_png)
print(mask_axes_png)

print("")
print("Created in-memory outputs:")
for name in ["axial", "coronal", "sagittal"]:
    print(
        f"{name}: image shape={thresholded_imgs[name].shape}, "
        f"mask shape={masks[name].shape}, "
        f"mask voxels={np.count_nonzero(masks[name].get_fdata())}"
    )

print("")
print("No files have been saved for NiftyMIC yet.")
print("Next module: geometry and overlap validation.")

# =============================================================================
# Module 5A - Threshold sweep for choosing a meaningful mask threshold
# =============================================================================
#
# Purpose:
#   Compare multiple absolute thresholds before creating final NiftyMIC masks.
#
# Input from Module 4:
#   native_imgs["axial"]
#   native_imgs["coronal"]
#   native_imgs["sagittal"]
#
# Output:
#   QC PNGs showing masks at different thresholds.
#
# Recommendation:
#   For NiftyMIC, choose the lowest threshold that removes most background
#   but keeps the full brain/skull outline continuous.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label


BASE_DIR = "DataSRR/volunteer_xxx/10mm/npy"
OUTPUT_DIR = os.path.join(BASE_DIR, "module5A_threshold_sweep_qc")
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLDS_TO_TEST = [30, 40, 50, 60, 75, 100, 125, 150, 200]


def largest_connected_component(mask):
    """
    Optional helper for measuring the largest component size.
    This does not modify the final mask unless explicitly used.
    """
    mask = mask.astype(bool)

    labeled, nlab = label(mask)
    if nlab == 0:
        return mask, 0, 0

    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    largest_label = np.argmax(counts)
    largest = labeled == largest_label

    return largest, int(counts[largest_label]), int(nlab)


def threshold_mask_stats(img, threshold):
    """
    Compute useful mask statistics for a candidate absolute threshold.
    """
    data = img.get_fdata().astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    mask = data > threshold
    mask_filled = binary_fill_holes(mask)

    largest, largest_count, n_components = largest_connected_component(mask_filled)

    total_voxels = data.size
    mask_count = int(np.count_nonzero(mask))
    filled_count = int(np.count_nonzero(mask_filled))

    stats = {
        "threshold": threshold,
        "mask_voxels": mask_count,
        "mask_fraction": mask_count / total_voxels,
        "filled_voxels": filled_count,
        "filled_fraction": filled_count / total_voxels,
        "n_components": n_components,
        "largest_component_voxels": largest_count,
        "largest_component_fraction_of_mask": largest_count / filled_count if filled_count > 0 else 0,
    }

    return stats, mask, mask_filled, largest


def get_axis2_mid(arr):
    """
    For these native stacks, axis2_mid is the in-plane anatomical view.
    """
    return arr[:, :, arr.shape[2] // 2]


def display_slice(slice_2d, rotate_k=1):
    return np.rot90(slice_2d, k=rotate_k)


def plot_threshold_sweep_for_stack(
    img,
    label,
    thresholds,
    save_path=None,
    rotate_k=1,
    show=True,
):
    """
    Plot axis2_mid mask for each threshold for one stack.
    """
    data = img.get_fdata().astype(np.float32)

    n = len(thresholds)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()

    fig.suptitle(f"Threshold sweep: {label}", fontsize=14)

    for i, threshold in enumerate(thresholds):
        stats, mask, mask_filled, largest = threshold_mask_stats(img, threshold)

        # For visualization only: show filled mask because holes are distracting.
        view = display_slice(get_axis2_mid(mask_filled), rotate_k=rotate_k)

        axes[i].imshow(view, cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")
        axes[i].set_title(
            f"thr={threshold}\n"
            f"frac={stats['mask_fraction']:.3f}\n"
            f"LCC/mask={stats['largest_component_fraction_of_mask']:.2f}"
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_threshold_sweep_table(native_imgs, thresholds):
    """
    Print threshold statistics across stacks.
    """
    print("")
    print("=" * 100)
    print("Module 5A - Threshold sweep table")
    print("=" * 100)

    for name, img in native_imgs.items():
        print("")
        print(f"--- {name} ---")
        print(
            f"{'thr':>8} | {'mask_frac':>10} | {'filled_frac':>11} | "
            f"{'n_comp':>7} | {'LCC_vox':>10} | {'LCC/fill':>9}"
        )
        print("-" * 75)

        for threshold in thresholds:
            stats, _, _, _ = threshold_mask_stats(img, threshold)

            print(
                f"{threshold:8.1f} | "
                f"{stats['mask_fraction']:10.4f} | "
                f"{stats['filled_fraction']:11.4f} | "
                f"{stats['n_components']:7d} | "
                f"{stats['largest_component_voxels']:10d} | "
                f"{stats['largest_component_fraction_of_mask']:9.3f}"
            )


# -----------------------------------------------------------------------------
# Run Module 5A
# -----------------------------------------------------------------------------

print_threshold_sweep_table(native_imgs, THRESHOLDS_TO_TEST)

for name, img in native_imgs.items():
    out_png = os.path.join(OUTPUT_DIR, f"{name}_threshold_sweep.png")

    plot_threshold_sweep_for_stack(
        img,
        label=name,
        thresholds=THRESHOLDS_TO_TEST,
        save_path=out_png,
        rotate_k=1,
        show=True,
    )

print("")
print("=" * 100)
print("Module 5A complete")
print("=" * 100)
print("Threshold sweep QC saved in:")
print(OUTPUT_DIR)
# This needs to be updated based on each subject's data quality
THRESHOLD = 40.0
USE_MASK_CLEANUP = False

# =============================================================================
# Module 6 - Geometry and overlap validation before saving for NiftyMIC
# =============================================================================
#
# Purpose:
#   Validate that the native NIfTI images and masks are physically meaningful
#   before passing them to NiftyMIC.
#
# Checks:
#   1. Image/mask shape match
#   2. Image/mask affine match
#   3. Image/mask zoom match
#   4. Mask is binary and non-empty
#   5. World-space bounding boxes overlap
#   6. Masks overlap after resampling to a common 2 mm grid
#
# Inputs from Module 5:
#   thresholded_imgs
#   masks
#
# Outputs:
#   Printed validation tables.
#   No files are saved in this module.
# =============================================================================

import itertools
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


# -----------------------------------------------------------------------------
# Basic image/mask validation
# -----------------------------------------------------------------------------

def validate_image_mask_geometry(img, mask_img, label="volume"):
    """
    Validate that image and mask are compatible.
    """
    img_data = img.get_fdata()
    mask_data = mask_img.get_fdata()

    same_shape = img.shape[:3] == mask_img.shape[:3]
    same_affine = np.allclose(img.affine, mask_img.affine)
    same_zooms = np.allclose(
        img.header.get_zooms()[:3],
        mask_img.header.get_zooms()[:3],
    )

    mask_unique = np.unique(mask_data)
    mask_binary = np.all(np.isin(mask_unique, [0, 1]))
    mask_nonzero = np.count_nonzero(mask_data)

    print("")
    print("=" * 80)
    print(f"Image/mask validation: {label}")
    print("=" * 80)

    print("image shape:", img.shape[:3])
    print("mask shape: ", mask_img.shape[:3])
    print("same shape:", same_shape)

    print("image zooms:", img.header.get_zooms()[:3])
    print("mask zooms: ", mask_img.header.get_zooms()[:3])
    print("same zooms:", same_zooms)

    print("same affine:", same_affine)

    print("image orientation:", nib.aff2axcodes(img.affine))
    print("mask orientation: ", nib.aff2axcodes(mask_img.affine))

    print("image nonzero voxels:", np.count_nonzero(img_data))
    print("mask nonzero voxels: ", mask_nonzero)
    print("mask fraction:", mask_nonzero / mask_data.size)

    print("mask unique values:", mask_unique[:10])
    print("mask binary:", mask_binary)

    if not same_shape:
        print("ERROR: image and mask shapes do not match.")

    if not same_affine:
        print("ERROR: image and mask affines do not match.")

    if not same_zooms:
        print("ERROR: image and mask zooms do not match.")

    if not mask_binary:
        print("ERROR: mask is not binary.")

    if mask_nonzero == 0:
        print("ERROR: mask is empty.")

    passed = (
        same_shape
        and same_affine
        and same_zooms
        and mask_binary
        and mask_nonzero > 0
    )

    print("PAIR VALIDATION PASSED:", passed)

    return passed


# -----------------------------------------------------------------------------
# World-space geometry helpers
# -----------------------------------------------------------------------------

def image_world_corners(img):
    """
    Return the 8 world-space corners of a NIfTI image.
    """
    shape = np.asarray(img.shape[:3], dtype=int)

    corners_ijk = np.array(
        [
            [0, 0, 0],
            [shape[0] - 1, 0, 0],
            [0, shape[1] - 1, 0],
            [0, 0, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, 0],
            [shape[0] - 1, 0, shape[2] - 1],
            [0, shape[1] - 1, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, shape[2] - 1],
        ],
        dtype=float,
    )

    corners_h = np.c_[corners_ijk, np.ones(8)]
    corners_xyz = (img.affine @ corners_h.T).T[:, :3]

    return corners_xyz


def image_world_bbox(img):
    """
    Return world-space bounding-box min/max.
    """
    corners_xyz = image_world_corners(img)
    return corners_xyz.min(axis=0), corners_xyz.max(axis=0)


def bbox_volume_mm3(bmin, bmax):
    """
    Compute axis-aligned bbox volume in mm^3.
    """
    extent = np.maximum(0, bmax - bmin)
    return float(np.prod(extent))


def bbox_overlap(bmin1, bmax1, bmin2, bmax2):
    """
    Compute physical overlap between two world-space bounding boxes.
    """
    overlap_min = np.maximum(bmin1, bmin2)
    overlap_max = np.minimum(bmax1, bmax2)

    overlap_extent = np.maximum(0, overlap_max - overlap_min)
    overlap_volume = float(np.prod(overlap_extent))

    vol1 = bbox_volume_mm3(bmin1, bmax1)
    vol2 = bbox_volume_mm3(bmin2, bmax2)

    frac1 = overlap_volume / vol1 if vol1 > 0 else 0.0
    frac2 = overlap_volume / vol2 if vol2 > 0 else 0.0

    return {
        "overlap_min": overlap_min,
        "overlap_max": overlap_max,
        "overlap_extent": overlap_extent,
        "overlap_volume_mm3": overlap_volume,
        "frac_of_img1_bbox": frac1,
        "frac_of_img2_bbox": frac2,
    }


def print_geometry_summary(imgs, masks=None):
    """
    Print shape, zooms, affine, orientation, center, and world bbox for each image.
    """
    print("")
    print("=" * 80)
    print("Module 6A - Geometry summary")
    print("=" * 80)

    for name, img in imgs.items():
        data = img.get_fdata()
        bmin, bmax = image_world_bbox(img)
        center = 0.5 * (bmin + bmax)
        extent = bmax - bmin

        print("")
        print(f"--- {name} ---")
        print("shape:", img.shape[:3])
        print("zooms:", img.header.get_zooms()[:3])
        print("orientation codes:", nib.aff2axcodes(img.affine))
        print("world min:", bmin)
        print("world max:", bmax)
        print("world center:", center)
        print("world extent mm:", extent)
        print("image nonzero voxels:", np.count_nonzero(data))

        if masks is not None and name in masks:
            mask_data = masks[name].get_fdata() > 0
            print("mask nonzero voxels:", np.count_nonzero(mask_data))
            print("mask fraction:", np.count_nonzero(mask_data) / mask_data.size)
            print("mask same affine:", np.allclose(img.affine, masks[name].affine))

        if not np.allclose(center, np.zeros(3), atol=1e-5):
            print("WARNING: world center is not close to [0, 0, 0].")


def print_bbox_overlap_matrix(imgs):
    """
    Print pairwise world-space bounding-box overlap matrix.
    """
    names = list(imgs.keys())
    n = len(names)

    bboxes = {
        name: image_world_bbox(img)
        for name, img in imgs.items()
    }

    overlap_vol = np.zeros((n, n), dtype=float)
    frac_i = np.zeros((n, n), dtype=float)

    print("")
    print("=" * 80)
    print("Module 6B - Pairwise world-space bounding-box overlap")
    print("=" * 80)

    for i, name_i in enumerate(names):
        bmin_i, bmax_i = bboxes[name_i]

        for j, name_j in enumerate(names):
            bmin_j, bmax_j = bboxes[name_j]

            ov = bbox_overlap(bmin_i, bmax_i, bmin_j, bmax_j)

            overlap_vol[i, j] = ov["overlap_volume_mm3"]
            frac_i[i, j] = ov["frac_of_img1_bbox"]

    print("")
    print("Order:", names)

    print("")
    print("Overlap volume matrix, mm^3:")
    print(np.array2string(overlap_vol, precision=1, suppress_small=True))

    print("")
    print("Overlap fraction matrix:")
    print("Rows are reference images; columns are comparison images.")
    print("Entry [i,j] = overlap volume / bbox volume of image i")
    print(np.array2string(frac_i, precision=3, suppress_small=True))

    print("")
    print("Pairwise bbox details:")

    all_passed = True

    for name_i, name_j in itertools.combinations(names, 2):
        bmin_i, bmax_i = bboxes[name_i]
        bmin_j, bmax_j = bboxes[name_j]

        ov = bbox_overlap(bmin_i, bmax_i, bmin_j, bmax_j)

        print("")
        print(f"{name_i} vs {name_j}")
        print("overlap extent mm:", ov["overlap_extent"])
        print("overlap volume mm^3:", ov["overlap_volume_mm3"])
        print(f"fraction of {name_i} bbox:", ov["frac_of_img1_bbox"])
        print(f"fraction of {name_j} bbox:", ov["frac_of_img2_bbox"])

        if ov["overlap_volume_mm3"] <= 0:
            print("ERROR: No physical bbox overlap. NiftyMIC/ITK will likely fail.")
            all_passed = False
        elif ov["frac_of_img1_bbox"] < 0.20 or ov["frac_of_img2_bbox"] < 0.20:
            print("WARNING: Low bbox overlap. Inspect affine/world-axis definitions.")
        else:
            print("OK: substantial bbox overlap.")

    print("")
    print("BBOX OVERLAP VALIDATION PASSED:", all_passed)

    return all_passed


# -----------------------------------------------------------------------------
# Common-grid mask overlap helpers
# -----------------------------------------------------------------------------

def make_common_reference_grid(imgs, resolution_mm=2.0, margin_mm=5.0):
    """
    Make a common RAS world-space grid covering all image bounding boxes.
    """
    all_mins = []
    all_maxs = []

    for img in imgs.values():
        bmin, bmax = image_world_bbox(img)
        all_mins.append(bmin)
        all_maxs.append(bmax)

    world_min = np.min(np.vstack(all_mins), axis=0) - margin_mm
    world_max = np.max(np.vstack(all_maxs), axis=0) + margin_mm

    extent = world_max - world_min
    ref_shape = np.ceil(extent / resolution_mm).astype(int) + 1

    ref_affine = np.eye(4, dtype=float)
    ref_affine[0, 0] = resolution_mm
    ref_affine[1, 1] = resolution_mm
    ref_affine[2, 2] = resolution_mm
    ref_affine[:3, 3] = world_min

    return tuple(ref_shape), ref_affine


def resample_mask_to_reference(mask_img, ref_shape, ref_affine):
    """
    Resample a binary mask into a common reference grid using nearest-neighbor.
    """
    mask_data = mask_img.get_fdata() > 0

    ii, jj, kk = np.meshgrid(
        np.arange(ref_shape[0]),
        np.arange(ref_shape[1]),
        np.arange(ref_shape[2]),
        indexing="ij",
    )

    ref_ijk = np.vstack(
        [
            ii.ravel(),
            jj.ravel(),
            kk.ravel(),
            np.ones(ii.size),
        ]
    )

    # reference voxel indices -> world coordinates -> source voxel indices
    world_xyz = ref_affine @ ref_ijk
    src_ijk = np.linalg.inv(mask_img.affine) @ world_xyz

    coords = [
        src_ijk[0, :],
        src_ijk[1, :],
        src_ijk[2, :],
    ]

    sampled = map_coordinates(
        mask_data.astype(np.float32),
        coords,
        order=0,
        mode="constant",
        cval=0,
    )

    sampled = sampled.reshape(ref_shape) > 0.5

    return sampled


def print_mask_overlap_matrix(imgs, masks, resolution_mm=2.0):
    """
    Resample masks into a common reference space and compute pairwise overlaps.
    """
    names = list(masks.keys())

    ref_shape, ref_affine = make_common_reference_grid(
        imgs,
        resolution_mm=resolution_mm,
        margin_mm=5.0,
    )

    print("")
    print("=" * 80)
    print("Module 6C - Common-grid mask overlap")
    print("=" * 80)

    print("reference grid shape:", ref_shape)
    print("reference resolution mm:", resolution_mm)
    print("reference affine:")
    print(ref_affine)

    masks_ref = {}

    for name in names:
        masks_ref[name] = resample_mask_to_reference(
            masks[name],
            ref_shape=ref_shape,
            ref_affine=ref_affine,
        )

    n = len(names)
    dice = np.zeros((n, n), dtype=float)
    intersect_counts = np.zeros((n, n), dtype=int)
    frac_i = np.zeros((n, n), dtype=float)
    mask_counts = np.zeros(n, dtype=int)

    for i, name_i in enumerate(names):
        mi = masks_ref[name_i]
        ni = np.count_nonzero(mi)
        mask_counts[i] = ni

        for j, name_j in enumerate(names):
            mj = masks_ref[name_j]
            nj = np.count_nonzero(mj)

            inter = np.count_nonzero(mi & mj)
            intersect_counts[i, j] = inter

            dice[i, j] = (2.0 * inter / (ni + nj)) if (ni + nj) > 0 else 0.0
            frac_i[i, j] = inter / ni if ni > 0 else 0.0

    print("")
    print("Order:", names)

    print("")
    print("Mask voxel counts in common grid:")
    for name, count in zip(names, mask_counts):
        print(f"{name}: {count}")

    print("")
    print("Mask intersection voxel count matrix:")
    print(intersect_counts)

    print("")
    print("Mask overlap fraction matrix:")
    print("Rows are reference masks; columns are comparison masks.")
    print("Entry [i,j] = intersection / voxels in mask i")
    print(np.array2string(frac_i, precision=3, suppress_small=True))

    print("")
    print("Mask Dice matrix:")
    print(np.array2string(dice, precision=3, suppress_small=True))

    print("")
    print("Pairwise mask details:")

    all_passed = True

    for name_i, name_j in itertools.combinations(names, 2):
        i = names.index(name_i)
        j = names.index(name_j)

        print("")
        print(f"{name_i} vs {name_j}")
        print("intersection voxels:", intersect_counts[i, j])
        print(f"fraction of {name_i} mask:", frac_i[i, j])
        print(f"fraction of {name_j} mask:", frac_i[j, i])
        print("Dice:", dice[i, j])

        if intersect_counts[i, j] == 0:
            print("ERROR: No mask overlap in common grid. Registration will likely fail.")
            all_passed = False
        elif dice[i, j] < 0.05:
            print("WARNING: Very low mask overlap. Check affine/axis geometry or threshold.")
        elif dice[i, j] < 0.15:
            print("CAUTION: Low mask overlap. May still register, but inspect geometry.")
        else:
            print("OK: nonzero mask overlap.")

    print("")
    print("MASK OVERLAP VALIDATION PASSED:", all_passed)

    return all_passed, {
        "names": names,
        "ref_shape": ref_shape,
        "ref_affine": ref_affine,
        "intersections": intersect_counts,
        "fractions": frac_i,
        "dice": dice,
        "mask_counts": mask_counts,
    }


# -----------------------------------------------------------------------------
# Run Module 6
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("Module 6 - Geometry and overlap validation")
print("=" * 80)

# Inputs from Module 5.
imgs_for_check = {
    "axial": thresholded_imgs["axial"],
    "coronal": thresholded_imgs["coronal"],
    "sagittal": thresholded_imgs["sagittal"],
}

masks_for_check = {
    "axial": masks["axial"],
    "coronal": masks["coronal"],
    "sagittal": masks["sagittal"],
}

pair_results = {}

for name in ["axial", "coronal", "sagittal"]:
    pair_results[name] = validate_image_mask_geometry(
        imgs_for_check[name],
        masks_for_check[name],
        label=name,
    )

print_geometry_summary(imgs_for_check, masks_for_check)

bbox_passed = print_bbox_overlap_matrix(imgs_for_check)

mask_passed, mask_overlap_results = print_mask_overlap_matrix(
    imgs_for_check,
    masks_for_check,
    resolution_mm=2.0,
)

all_pairs_passed = all(pair_results.values())
all_module6_passed = all_pairs_passed and bbox_passed and mask_passed

print("")
print("=" * 80)
print("Module 6 complete")
print("=" * 80)

print("Image/mask pair validation passed:", all_pairs_passed)
print("Bounding-box overlap validation passed:", bbox_passed)
print("Common-grid mask overlap validation passed:", mask_passed)
print("MODULE 6 PASSED:", all_module6_passed)

if all_module6_passed:
    print("")
    print("Good to proceed to Module 7: save/reload validation.")
else:
    print("")
    print("Do not proceed to NiftyMIC yet. Inspect the warnings/errors above.")
    
# =============================================================================
# Module 7 - Save NiftyMIC inputs and reload-validate from disk
# =============================================================================
#
# Purpose:
#   Save thresholded native NIfTI images and masks after Module 6 validation,
#   then reload them from disk and verify that nothing changed during saving.
#
# Inputs from Module 5/6:
#   thresholded_imgs["axial"]
#   thresholded_imgs["coronal"]
#   thresholded_imgs["sagittal"]
#
#   masks["axial"]
#   masks["coronal"]
#   masks["sagittal"]
#
# Outputs:
#   Native NIfTI images:
#       axial_native.nii.gz
#       coronal_native.nii.gz
#       sagittal_native.nii.gz
#
#   Native masks:
#       axial_native_mask.nii.gz
#       coronal_native_mask.nii.gz
#       sagittal_native_mask.nii.gz
#
# Validation:
#   1. Reload saved files.
#   2. Confirm shape, zooms, affine, orientation, nonzero voxels.
#   3. Confirm image/mask pair consistency.
#   4. Confirm files are not duplicated by MD5.
#   5. Re-run overlap validation from the reloaded files.
# =============================================================================

import os
import hashlib
import numpy as np
import nibabel as nib


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------



os.makedirs(NIFTYMIC_INPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------

OUTPUT_PATHS = {
    "axial": {
        "image": os.path.join(NIFTYMIC_INPUT_DIR, "axial_native.nii.gz"),
        "mask":  os.path.join(NIFTYMIC_INPUT_DIR, "axial_native_mask.nii.gz"),
    },
    "coronal": {
        "image": os.path.join(NIFTYMIC_INPUT_DIR, "coronal_native.nii.gz"),
        "mask":  os.path.join(NIFTYMIC_INPUT_DIR, "coronal_native_mask.nii.gz"),
    },
    "sagittal": {
        "image": os.path.join(NIFTYMIC_INPUT_DIR, "sagittal_native.nii.gz"),
        "mask":  os.path.join(NIFTYMIC_INPUT_DIR, "sagittal_native_mask.nii.gz"),
    },
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def file_md5(path, block_size=65536):
    """
    Compute MD5 hash of a file to detect accidental duplicated outputs.
    """
    h = hashlib.md5()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)

    return h.hexdigest()


def save_nifti_with_forms(img, path):
    """
    Save NIfTI while explicitly preserving qform/sform as img.affine.
    """
    out = nib.Nifti1Image(
        img.get_fdata().astype(img.get_data_dtype()),
        img.affine.copy(),
        img.header.copy(),
    )

    out.set_qform(img.affine.copy(), code=1)
    out.set_sform(img.affine.copy(), code=1)

    nib.save(out, path)


def inspect_saved_nifti(path, label="image"):
    """
    Print basic info for a saved/reloaded NIfTI file.
    """
    img = nib.load(path)
    data = img.get_fdata()

    print("")
    print("=" * 80)
    print(f"Saved file inspection: {label}")
    print("=" * 80)
    print("path:", path)
    print("exists:", os.path.exists(path))
    print("file size bytes:", os.path.getsize(path))
    print("md5:", file_md5(path))

    print("shape:", img.shape[:3])
    print("dtype:", img.get_data_dtype())
    print("zooms:", img.header.get_zooms()[:3])
    print("orientation:", nib.aff2axcodes(img.affine))
    print("qform code:", img.header["qform_code"])
    print("sform code:", img.header["sform_code"])
    print("affine:")
    print(img.affine)

    print("min/max/mean:", np.nanmin(data), np.nanmax(data), np.nanmean(data))
    print("nonzero voxels:", np.count_nonzero(data))

    return img


def compare_in_memory_vs_reloaded(img_memory, img_reloaded, label="image"):
    """
    Verify that saved/reloaded image still matches the in-memory object.
    """
    data_memory = img_memory.get_fdata()
    data_reloaded = img_reloaded.get_fdata()

    same_shape = img_memory.shape[:3] == img_reloaded.shape[:3]
    same_affine = np.allclose(img_memory.affine, img_reloaded.affine)
    same_zooms = np.allclose(
        img_memory.header.get_zooms()[:3],
        img_reloaded.header.get_zooms()[:3],
    )
    same_data = np.allclose(data_memory, data_reloaded)

    print("")
    print(f"--- In-memory vs reloaded check: {label} ---")
    print("same shape:", same_shape)
    print("same zooms:", same_zooms)
    print("same affine:", same_affine)
    print("same data:", same_data)

    passed = same_shape and same_zooms and same_affine and same_data
    print("PASSED:", passed)

    return passed


def check_for_duplicate_outputs(output_paths):
    """
    Check whether any saved files are accidentally byte-identical.
    """
    print("")
    print("=" * 80)
    print("Duplicate-file MD5 check")
    print("=" * 80)

    entries = []

    for name, paths in output_paths.items():
        for kind, path in paths.items():
            entries.append((f"{name}_{kind}", path, file_md5(path)))

    for label, path, digest in entries:
        print(f"{label:20s} {digest}  {path}")

    print("")
    print("Potential duplicates:")

    found_duplicate = False

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            label_i, path_i, md5_i = entries[i]
            label_j, path_j, md5_j = entries[j]

            if md5_i == md5_j:
                found_duplicate = True
                print(f"WARNING: {label_i} and {label_j} are byte-identical.")
                print(f"  {path_i}")
                print(f"  {path_j}")

    if not found_duplicate:
        print("No byte-identical saved files detected.")

    return not found_duplicate

# -----------------------------------------------------------------------------
# Run Module 7 save
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("Module 7 - Save NiftyMIC native inputs")
print("=" * 80)
print("Output directory:")
print(NIFTYMIC_INPUT_DIR)

for name in ["axial", "coronal", "sagittal"]:
    img_path = OUTPUT_PATHS[name]["image"]
    mask_path = OUTPUT_PATHS[name]["mask"]

    save_nifti_with_forms(thresholded_imgs[name], img_path)
    save_nifti_with_forms(masks[name], mask_path)

    print("")
    print(f"Saved {name}:")
    print("image:", img_path)
    print("mask: ", mask_path)


# -----------------------------------------------------------------------------
# Reload saved files
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("Reload saved files")
print("=" * 80)

reloaded_imgs = {}
reloaded_masks = {}

for name in ["axial", "coronal", "sagittal"]:
    reloaded_imgs[name] = inspect_saved_nifti(
        OUTPUT_PATHS[name]["image"],
        label=f"{name} image",
    )

    reloaded_masks[name] = inspect_saved_nifti(
        OUTPUT_PATHS[name]["mask"],
        label=f"{name} mask",
    )

# -----------------------------------------------------------------------------
# Compare in-memory objects to reloaded files
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("In-memory vs reloaded validation")
print("=" * 80)

compare_results = []

for name in ["axial", "coronal", "sagittal"]:
    compare_results.append(
        compare_in_memory_vs_reloaded(
            thresholded_imgs[name],
            reloaded_imgs[name],
            label=f"{name} image",
        )
    )

    compare_results.append(
        compare_in_memory_vs_reloaded(
            masks[name],
            reloaded_masks[name],
            label=f"{name} mask",
        )
    )

# -----------------------------------------------------------------------------
# Check for accidental duplicated files
# -----------------------------------------------------------------------------

no_duplicates = check_for_duplicate_outputs(OUTPUT_PATHS)


# -----------------------------------------------------------------------------
# Re-run critical Module 6 checks on reloaded files
# -----------------------------------------------------------------------------
#
# This assumes Module 6 functions are still defined:
#   validate_image_mask_geometry
#   print_geometry_summary
#   print_bbox_overlap_matrix
#   print_mask_overlap_matrix

print("")
print("=" * 80)
print("Re-run geometry/overlap validation on reloaded files")
print("=" * 80)

pair_results_reloaded = {}

for name in ["axial", "coronal", "sagittal"]:
    pair_results_reloaded[name] = validate_image_mask_geometry(
        reloaded_imgs[name],
        reloaded_masks[name],
        label=f"{name} reloaded",
    )

print_geometry_summary(reloaded_imgs, reloaded_masks)

bbox_passed_reloaded = print_bbox_overlap_matrix(reloaded_imgs)

mask_passed_reloaded, mask_overlap_results_reloaded = print_mask_overlap_matrix(
    reloaded_imgs,
    reloaded_masks,
    resolution_mm=2.0,
)


# -----------------------------------------------------------------------------
# Module 7 final status
# -----------------------------------------------------------------------------

all_compare_passed = all(compare_results)
all_pairs_passed_reloaded = all(pair_results_reloaded.values())

module7_passed = (
    all_compare_passed
    and all_pairs_passed_reloaded
    and bbox_passed_reloaded
    and mask_passed_reloaded
    and no_duplicates
)

print("")
print("=" * 80)
print("Module 7 complete")
print("=" * 80)

print("In-memory vs reloaded passed:", all_compare_passed)
print("Reloaded image/mask pair validation passed:", all_pairs_passed_reloaded)
print("Reloaded bbox overlap validation passed:", bbox_passed_reloaded)
print("Reloaded mask overlap validation passed:", mask_passed_reloaded)
print("No duplicate output files:", no_duplicates)
print("MODULE 7 PASSED:", module7_passed)

if module7_passed:
    print("")
    print("Good to proceed to Module 8: print and run NiftyMIC command.")
    print("")
    print("NiftyMIC input images:")
    print(OUTPUT_PATHS["axial"]["image"])
    print(OUTPUT_PATHS["coronal"]["image"])
    print(OUTPUT_PATHS["sagittal"]["image"])

    print("")
    print("NiftyMIC input masks:")
    print(OUTPUT_PATHS["axial"]["mask"])
    print(OUTPUT_PATHS["coronal"]["mask"])
    print(OUTPUT_PATHS["sagittal"]["mask"])
else:
    print("")
    print("Do not proceed to NiftyMIC yet. Inspect the failed validation above.")
    
    
# =============================================================================
# Module 8 - Generate NiftyMIC commands and shell scripts
# =============================================================================
#
# Purpose:
#   Create copy-paste-ready NiftyMIC Docker commands using the validated,
#   saved native NIfTI files from Module 7.
#
# Inputs from Module 7:
#   OUTPUT_PATHS
#   NIFTYMIC_INPUT_DIR
#
# This module writes:
#   run_niftymic_geometry_debug.sh
#   run_niftymic_1cycle_no_intensity.sh
#
# Recommended execution order:
#   1. Run geometry debug first.
#   2. Inspect output.
#   3. Then run 1-cycle version.
#
# Important:
#   Keep --intensity-correction 0 for now because the container previously
#   failed with:
#       libcurl.so.4: cannot open shared object file
# =============================================================================

import os
import textwrap

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

DOCKER_IMAGE = "renbem/niftymic"

# Reconstruction target resolution
ISOTROPIC_RESOLUTION_MM = 2

# Reconstruction type
RECONSTRUCTION_TYPE = "HuberL2"

# Regularization
ALPHA = 0.02

# Use 0 until the Docker libcurl / ITK-SNAP issue is resolved.
INTENSITY_CORRECTION = 0

# Output directory from Module 7.
# This should already exist:
#   DataSRR/ajay_scan_2/10mm/npy/native_niftymic_inputs_thr40
SCRIPT_DIR = NIFTYMIC_INPUT_DIR


# -----------------------------------------------------------------------------
# Paths from Module 7
# -----------------------------------------------------------------------------

AXIAL_IMG = OUTPUT_PATHS["axial"]["image"]
CORONAL_IMG = OUTPUT_PATHS["coronal"]["image"]
SAGITTAL_IMG = OUTPUT_PATHS["sagittal"]["image"]

AXIAL_MASK = OUTPUT_PATHS["axial"]["mask"]
CORONAL_MASK = OUTPUT_PATHS["coronal"]["mask"]
SAGITTAL_MASK = OUTPUT_PATHS["sagittal"]["mask"]


# -----------------------------------------------------------------------------
# Command builders
# -----------------------------------------------------------------------------

def make_niftymic_command(
    output_path,
    two_step_cycles=0,
    intensity_correction=0,
    outlier_rejection=0,
    threshold_first=None,
    threshold=None,
    alpha=0.02,
    isotropic_resolution=2,
    reconstruction_type="HuberL2",
    verbose=1,
):
    """
    Build a NiftyMIC Docker command.

    threshold_first and threshold are optional. For the first native geometry
    debug run, I prefer omitting them.
    """
    lines = [
        'docker run --rm -v "$PWD:$PWD" -w "$PWD" ' + DOCKER_IMAGE + " \\",
        "  niftymic_reconstruct_volume \\",
        "  --filenames \\",
        f"    {AXIAL_IMG} \\",
        f"    {CORONAL_IMG} \\",
        f"    {SAGITTAL_IMG} \\",
        "  --filenames-masks \\",
        f"    {AXIAL_MASK} \\",
        f"    {CORONAL_MASK} \\",
        f"    {SAGITTAL_MASK} \\",
        f"  --alpha {alpha} \\",
        f"  --outlier-rejection {outlier_rejection} \\",
        f"  --intensity-correction {intensity_correction} \\",
        f"  --two-step-cycles {two_step_cycles} \\",
        f"  --isotropic-resolution {isotropic_resolution} \\",
        f"  --output {output_path} \\",
        f"  --verbose {verbose} \\",
        f"  --reconstruction-type {reconstruction_type}",
    ]

    # Add optional NiftyMIC threshold arguments before output if requested.
    if threshold_first is not None or threshold is not None:
        insert_at = 11  # after outlier-rejection line approximately
        optional_lines = []

        if threshold_first is not None:
            optional_lines.append(f"  --threshold-first {threshold_first} \\")

        if threshold is not None:
            optional_lines.append(f"  --threshold {threshold} \\")

        for line in reversed(optional_lines):
            lines.insert(insert_at, line)

    return "\n".join(lines)


def write_shell_script(path, command, header_comment):
    """
    Write a runnable shell script.
    """
    script_text = f"""#!/usr/bin/env bash
set -euo pipefail

# {header_comment}
# Generated by Module 8.
# Run from the project directory that contains DataSRR/.

{command}
"""

    with open(path, "w") as f:
        f.write(script_text)

    os.chmod(path, 0o755)

    print(f"Saved shell script: {path}")


# -----------------------------------------------------------------------------
# Build recommended commands
# -----------------------------------------------------------------------------

geometry_debug_output = os.path.join(
    SCRIPT_DIR,
    "srr_native_thr40_geometry_debug_2mm.nii.gz",
)

onecycle_output = os.path.join(
    SCRIPT_DIR,
    "srr_native_thr40_1cycle_no_intensity_2mm.nii.gz",
)


# 1. Cleanest geometry-debug run.
# No intensity correction, no two-step refinement, no NiftyMIC internal threshold.
cmd_geometry_debug = make_niftymic_command(
    output_path=geometry_debug_output,
    two_step_cycles=0,
    intensity_correction=0,
    outlier_rejection=0,
    threshold_first=None,
    threshold=None,
    alpha=ALPHA,
    isotropic_resolution=ISOTROPIC_RESOLUTION_MM,
    reconstruction_type=RECONSTRUCTION_TYPE,
    verbose=1,
)


# 2. First refined run.
# Add one two-step cycle, still no intensity correction.
cmd_onecycle = make_niftymic_command(
    output_path=onecycle_output,
    two_step_cycles=1,
    intensity_correction=0,
    outlier_rejection=0,
    threshold_first=None,
    threshold=None,
    alpha=ALPHA,
    isotropic_resolution=ISOTROPIC_RESOLUTION_MM,
    reconstruction_type=RECONSTRUCTION_TYPE,
    verbose=1,
)


# -----------------------------------------------------------------------------
# Print commands
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("Module 8 - NiftyMIC command generation")
print("=" * 80)

print("")
print("NiftyMIC input images:")
print(AXIAL_IMG)
print(CORONAL_IMG)
print(SAGITTAL_IMG)

print("")
print("NiftyMIC input masks:")
print(AXIAL_MASK)
print(CORONAL_MASK)
print(SAGITTAL_MASK)

print("")
print("=" * 80)
print("Command 1: geometry-debug reconstruction")
print("=" * 80)
print(cmd_geometry_debug)

print("")
print("=" * 80)
print("Command 2: one-cycle reconstruction, no intensity correction")
print("=" * 80)
print(cmd_onecycle)


# -----------------------------------------------------------------------------
# Write shell scripts
# -----------------------------------------------------------------------------

geometry_debug_script = os.path.join(
    SCRIPT_DIR,
    "run_niftymic_geometry_debug.sh",
)

onecycle_script = os.path.join(
    SCRIPT_DIR,
    "run_niftymic_1cycle_no_intensity.sh",
)

write_shell_script(
    geometry_debug_script,
    cmd_geometry_debug,
    header_comment="NiftyMIC geometry-debug run: no intensity correction, no two-step cycles.",
)

write_shell_script(
    onecycle_script,
    cmd_onecycle,
    header_comment="NiftyMIC one-cycle run: two-step-cycles=1, intensity-correction=0.",
)


print("")
print("=" * 80)
print("Module 8 complete")
print("=" * 80)

print("")
print("Run these in order:")
print("")
print(f"bash {geometry_debug_script}")
print(f"bash {onecycle_script}")

print("")
print("First inspect:")
print(geometry_debug_output)

print("")
print("Then inspect:")
print(onecycle_output)

print("")
print("Recommended next step:")
print("Run the geometry-debug command first. If it reconstructs without split-plane artifacts,")
print("then run the one-cycle command. Keep intensity correction off until the Docker libcurl")
print("issue is fixed.")

# =============================================================================
# Module 9 - Evaluate NiftyMIC geometry-debug reconstruction
# =============================================================================
#
# Purpose:
#   Inspect the geometry-debug SRR output before running the next NiftyMIC script.
#
# This module:
#   1. Loads the geometry-debug SRR output.
#   2. Prints shape, zooms, affine, orientation, intensity stats.
#   3. Opens OrthoSlicer3D for interactive browsing.
#   4. Saves a static central-slice QC PNG.
#   5. Compares SRR bbox against input stack bboxes.
#   6. Optionally launches the next shell script from Python.
#
# Assumes Module 7/8 variables exist:
#   NIFTYMIC_INPUT_DIR
#   OUTPUT_PATHS
#
# If running standalone, set NIFTYMIC_INPUT_DIR manually below.
# =============================================================================

import os
import subprocess
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

# If needed, uncomment this standalone path:
# NIFTYMIC_INPUT_DIR = "DataSRR/ajay_scan_2/10mm/npy/native_niftymic_inputs_thr40"

GEOMETRY_DEBUG_RECON = os.path.join(
    NIFTYMIC_INPUT_DIR,
    "srr_native_thr40_geometry_debug_2mm.nii.gz",
)

ONECYCLE_SCRIPT = os.path.join(
    NIFTYMIC_INPUT_DIR,
    "run_niftymic_1cycle_no_intensity.sh",
)

QC_DIR = os.path.join(NIFTYMIC_INPUT_DIR, "module9_recon_qc")
os.makedirs(QC_DIR, exist_ok=True)

# RUN_NEXT_SCRIPT_FROM_PYTHON = False
# Set to True only after visual inspection looks acceptable.
RUN_NEXT_SCRIPT_FROM_PYTHON = True


# -----------------------------------------------------------------------------
# Geometry utilities
# -----------------------------------------------------------------------------

def image_world_corners(img):
    """
    Return the 8 world-space corners of a NIfTI image.
    """
    shape = np.asarray(img.shape[:3], dtype=int)

    corners_ijk = np.array(
        [
            [0, 0, 0],
            [shape[0] - 1, 0, 0],
            [0, shape[1] - 1, 0],
            [0, 0, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, 0],
            [shape[0] - 1, 0, shape[2] - 1],
            [0, shape[1] - 1, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, shape[2] - 1],
        ],
        dtype=float,
    )

    corners_h = np.c_[corners_ijk, np.ones(8)]
    corners_xyz = (img.affine @ corners_h.T).T[:, :3]

    return corners_xyz


def image_world_bbox(img):
    """
    Return world-space bbox min/max.
    """
    corners = image_world_corners(img)
    return corners.min(axis=0), corners.max(axis=0)


def inspect_nifti(img, label="image"):
    """
    Print NIfTI summary.
    """
    data = img.get_fdata()
    finite = data[np.isfinite(data)]

    bmin, bmax = image_world_bbox(img)
    center = 0.5 * (bmin + bmax)
    extent = bmax - bmin

    print("")
    print("=" * 80)
    print(f"Inspect NIfTI: {label}")
    print("=" * 80)

    print("shape:", img.shape[:3])
    print("dtype:", img.get_data_dtype())
    print("zooms:", img.header.get_zooms()[:3])
    print("orientation:", nib.aff2axcodes(img.affine))
    print("qform code:", img.header["qform_code"])
    print("sform code:", img.header["sform_code"])

    print("")
    print("affine:")
    print(img.affine)

    print("")
    print("world min:", bmin)
    print("world max:", bmax)
    print("world center:", center)
    print("world extent mm:", extent)

    print("")
    print("finite voxels:", finite.size, "/", data.size)
    print("nan voxels:", np.count_nonzero(np.isnan(data)))
    print("inf voxels:", np.count_nonzero(np.isinf(data)))
    print("nonzero voxels:", np.count_nonzero(data))

    if finite.size > 0:
        print("min:", np.min(finite))
        print("max:", np.max(finite))
        print("mean:", np.mean(finite))
        print("std:", np.std(finite))

        percentiles = [0, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 100]
        pvals = np.percentile(finite, percentiles)

        print("")
        print("Percentiles:")
        for p, v in zip(percentiles, pvals):
            print(f"  p{p:>5}: {v:.6f}")


def bbox_overlap_summary(img_a, img_b, label_a="A", label_b="B"):
    """
    Print simple bbox overlap between two images.
    """
    amin, amax = image_world_bbox(img_a)
    bmin, bmax = image_world_bbox(img_b)

    overlap_min = np.maximum(amin, bmin)
    overlap_max = np.minimum(amax, bmax)
    overlap_extent = np.maximum(0, overlap_max - overlap_min)

    overlap_volume = float(np.prod(overlap_extent))
    a_volume = float(np.prod(np.maximum(0, amax - amin)))
    b_volume = float(np.prod(np.maximum(0, bmax - bmin)))

    frac_a = overlap_volume / a_volume if a_volume > 0 else 0
    frac_b = overlap_volume / b_volume if b_volume > 0 else 0

    print("")
    print(f"{label_a} vs {label_b}")
    print("overlap extent mm:", overlap_extent)
    print("overlap volume mm^3:", overlap_volume)
    print(f"fraction of {label_a} bbox:", frac_a)
    print(f"fraction of {label_b} bbox:", frac_b)

    if overlap_volume <= 0:
        print("ERROR: no physical overlap.")
    elif frac_a < 0.2 or frac_b < 0.2:
        print("WARNING: low physical overlap.")
    else:
        print("OK: physical overlap present.")


# -----------------------------------------------------------------------------
# Visualization utilities
# -----------------------------------------------------------------------------

def robust_clim(data, low=1, high=99.5, nonzero_only=True):
    """
    Robust display limits.
    """
    vals = np.asarray(data)

    vals = vals[np.isfinite(vals)]

    if nonzero_only:
        vals = vals[vals != 0]

    if vals.size == 0:
        return 0, 1

    return tuple(np.percentile(vals, [low, high]).astype(float))


def central_slices_ras_like(data):
    """
    Return central slices along array axes.

    These are array-space central views, not reoriented to canonical RAS.
    For QC, this is usually enough. OrthoSlicer3D gives interactive browsing.
    """
    cx, cy, cz = [s // 2 for s in data.shape[:3]]

    return {
        "axis0_mid": data[cx, :, :],
        "axis1_mid": data[:, cy, :],
        "axis2_mid": data[:, :, cz],
    }


def plot_recon_central_slices(img, save_path, title="SRR reconstruction"):
    """
    Save static QC image with central slices along all three array axes.
    """
    data = img.get_fdata().astype(np.float32)
    clim = robust_clim(data, low=1, high=99.5, nonzero_only=True)

    slices = central_slices_ras_like(data)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"{title}\nshape={img.shape[:3]}, zooms={img.header.get_zooms()[:3]}, orientation={nib.aff2axcodes(img.affine)}",
        fontsize=12,
    )

    for ax, key in zip(axes, ["axis0_mid", "axis1_mid", "axis2_mid"]):
        view = np.rot90(slices[key])
        ax.imshow(view, cmap="gray", vmin=clim[0], vmax=clim[1])
        ax.set_title(key)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved static QC PNG: {save_path}")
    plt.show()


def browse_with_orthoslicer(img, title="SRR reconstruction"):
    """
    Open interactive OrthoSlicer3D viewer.
    """
    data = img.get_fdata().astype(np.float32)
    # clim = robust_clim(data, low=1, high=125, nonzero_only=True)
    
    clim = (0, 1.5 * np.max(data))

    print("")
    print(f"Opening OrthoSlicer3D: {title}")
    print("clim:", clim)

    slicer = OrthoSlicer3D(data)
    slicer.clim = clim
    slicer.show()


# -----------------------------------------------------------------------------
# Subprocess helper for next NiftyMIC command
# -----------------------------------------------------------------------------

def run_shell_script(script_path, cwd=None):
    """
    Run a shell script from Python and stream output.

    This keeps the workflow inside Python while still using Docker through shell.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Could not find script: {script_path}")

    if cwd is None:
        cwd = os.getcwd()

    print("")
    print("=" * 80)
    print("Running shell script")
    print("=" * 80)
    print("script:", script_path)
    print("cwd:", cwd)

    process = subprocess.Popen(
        ["bash", script_path],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_lines = []

    for line in process.stdout:
        print(line, end="")
        log_lines.append(line)

    return_code = process.wait()

    print("")
    print("=" * 80)
    print("Shell script finished")
    print("=" * 80)
    print("return code:", return_code)

    log_path = script_path.replace(".sh", ".log")
    with open(log_path, "w") as f:
        f.writelines(log_lines)

    print("Saved log:", log_path)

    if return_code != 0:
        raise RuntimeError(f"Shell script failed with return code {return_code}: {script_path}")

    return log_path


# -----------------------------------------------------------------------------
# Run Module 9 assessment
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("Module 9 - Evaluate geometry-debug reconstruction")
print("=" * 80)

print("Expected geometry-debug output:")
print(GEOMETRY_DEBUG_RECON)

if not os.path.exists(GEOMETRY_DEBUG_RECON):
    raise FileNotFoundError(
        f"Geometry-debug reconstruction not found:\n{GEOMETRY_DEBUG_RECON}"
    )

recon_img = nib.load(GEOMETRY_DEBUG_RECON)

inspect_nifti(recon_img, label="geometry-debug SRR")


# -----------------------------------------------------------------------------
# Compare SRR bbox to input stack bboxes if reloaded_imgs exists from Module 7
# -----------------------------------------------------------------------------

if "reloaded_imgs" in globals():
    print("")
    print("=" * 80)
    print("SRR vs input-stack bbox overlap")
    print("=" * 80)

    for name, input_img in reloaded_imgs.items():
        bbox_overlap_summary(
            recon_img,
            input_img,
            label_a="SRR",
            label_b=name,
        )
else:
    print("")
    print("Note: reloaded_imgs not found in globals; skipping SRR/input bbox comparison.")


# -----------------------------------------------------------------------------
# Static central slice QC
# -----------------------------------------------------------------------------

static_qc_png = os.path.join(
    QC_DIR,
    "geometry_debug_recon_central_slices.png",
)

plot_recon_central_slices(
    recon_img,
    save_path=static_qc_png,
    title="Geometry-debug SRR",
)


# -----------------------------------------------------------------------------
# Interactive OrthoSlicer
# -----------------------------------------------------------------------------

browse_with_orthoslicer(
    recon_img,
    title="Geometry-debug SRR",
)


# -----------------------------------------------------------------------------
# Human decision gate
# -----------------------------------------------------------------------------

print("")
print("=" * 80)
print("Module 9 visual decision gate")
print("=" * 80)
print("Inspect the OrthoSlicer3D output before proceeding.")
print("")
print("Proceed only if:")
print("  1. No split-plane / duplicate-plane artifact.")
print("  2. Brain appears as one coherent 3D object.")
print("  3. Axial/coronal/sagittal browsing looks physically consistent.")
print("  4. The output is not empty or dominated by background.")
print("")
print("Current next script:")
print(ONECYCLE_SCRIPT)
print("")
print("To launch the next script from Python, set:")
print("  RUN_NEXT_SCRIPT_FROM_PYTHON = True")
print("and rerun the final block below.")


# -----------------------------------------------------------------------------
# Optional: run next NiftyMIC script from Python
# -----------------------------------------------------------------------------

if RUN_NEXT_SCRIPT_FROM_PYTHON:
    onecycle_log = run_shell_script(
        ONECYCLE_SCRIPT,
        cwd=os.getcwd(),
    )

    print("")
    print("One-cycle NiftyMIC run completed.")
    print("Log:", onecycle_log)
else:
    print("")
    print("RUN_NEXT_SCRIPT_FROM_PYTHON is False.")
    print("The next NiftyMIC script was not launched.")
    
# =============================================================================
# Module 10 - Run refined NiftyMIC reconstruction and evaluate output
# =============================================================================
#
# Purpose:
#   Run the next NiftyMIC reconstruction after geometry-debug succeeded.
#
# Recommended next run:
#   reconstruction-type: HuberL2
#   alpha: 0.03
#   two-step-cycles: 2
#   outlier-rejection: 1
#   intensity-correction: 0
#
# This module:
#   1. Builds the Docker command.
#   2. Runs it from Python using subprocess.
#   3. Saves stdout/stderr to a log file.
#   4. Loads the refined SRR output.
#   5. Prints geometry/intensity QC.
#   6. Saves central-slice PNG.
#   7. Opens OrthoSlicer3D for browsing.
#
# Assumes these variables exist from previous modules:
#   NIFTYMIC_INPUT_DIR
#   OUTPUT_PATHS
#
# If running standalone, set them manually.
# =============================================================================

import os
import subprocess
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D


# =============================================================================
# Backend-safe display helper
# =============================================================================

def safe_set_interactive_backend():
    """
    Avoid forcing QtAgg after another backend is already active.
    On macOS, MacOSX or TkAgg is usually safest.
    """
    backend = matplotlib.get_backend()
    print("Current matplotlib backend:", backend)
    return backend


safe_set_interactive_backend()


# =============================================================================
# User settings
# =============================================================================

# If needed for standalone execution:
# NIFTYMIC_INPUT_DIR = "DataSRR/ajay_scan_2/10mm/npy/native_niftymic_inputs_thr40"

DOCKER_IMAGE = "renbem/niftymic"

# REFINED_RECON_NAME = "srr_native_thr40_2cycle_outlier_HuberL2_alpha003_2mm.nii.gz"
# REFINED_RECON_PATH = os.path.join(NIFTYMIC_INPUT_DIR, REFINED_RECON_NAME)

# REFINED_LOG_PATH = os.path.join(
#     NIFTYMIC_INPUT_DIR,
#     "run_niftymic_2cycle_outlier_HuberL2_alpha003.log",
# )

# REFINED_SCRIPT_PATH = os.path.join(
#     NIFTYMIC_INPUT_DIR,
#     "run_niftymic_2cycle_outlier_HuberL2_alpha003.sh",
# )

def make_run_name(p):

    return (
        f"srr"
        f"_alpha{str(p['alpha']).replace('.','')}"
        f"_out{p['outlier_rejection']}"
        f"_ic{p['intensity_correction']}"
        f"_cycle{p['two_step_cycles']}"
        f"_iso{p['isotropic_resolution']}"
        f"_{p['reconstruction_type']}"
    )

QC_DIR = os.path.join(NIFTYMIC_INPUT_DIR, "module10_refined_recon_qc")
os.makedirs(QC_DIR, exist_ok=True)

# Recommended next-run parameters
# NIFTYMIC_PARAMS = {
#     "alpha": 0.03,
#     "outlier_rejection": 1,
#     "intensity_correction": 0,
#     "two_step_cycles": 2,
#     "isotropic_resolution": 2,
#     "reconstruction_type": "HuberL2",
#     "verbose": 1,
# }

from itertools import product

PARAM_GRID = {
    "alpha": [0.01],
    "outlier_rejection": [1],
    "intensity_correction": [0],
    "two_step_cycles": [2],
    "isotropic_resolution": [2],
    "reconstruction_type": [
        "HuberL2"
    ],
    "verbose": [1],
}

keys = list(PARAM_GRID.keys())

PARAM_COMBINATIONS = [
    dict(zip(keys, vals))
    for vals in product(*(PARAM_GRID[k] for k in keys))
]

print(f"Running {len(PARAM_COMBINATIONS)} combinations")

# Optional dynamic range choice for visualization.
# You mentioned using 1.25 * max; this is okay for browsing,
# but percentile-based clim is usually better for QC.
ORTHOSLICER_CLIM_MODE = "percentile"
# Options:
#   "percentile"
#   "max_1p25"


# =============================================================================
# Inputs from Module 7
# =============================================================================

AXIAL_IMG = OUTPUT_PATHS["axial"]["image"]
CORONAL_IMG = OUTPUT_PATHS["coronal"]["image"]
SAGITTAL_IMG = OUTPUT_PATHS["sagittal"]["image"]

AXIAL_MASK = OUTPUT_PATHS["axial"]["mask"]
CORONAL_MASK = OUTPUT_PATHS["coronal"]["mask"]
SAGITTAL_MASK = OUTPUT_PATHS["sagittal"]["mask"]


# =============================================================================
# Command construction
# =============================================================================

def build_niftymic_refined_command():
    """
    Build the refined NiftyMIC Docker command.
    """
    p = NIFTYMIC_PARAMS

    cmd = f"""
docker run --rm -v "$PWD:$PWD" -w "$PWD" {DOCKER_IMAGE} \\
  niftymic_reconstruct_volume \\
  --filenames \\
    {AXIAL_IMG} \\
    {CORONAL_IMG} \\
    {SAGITTAL_IMG} \\
  --filenames-masks \\
    {AXIAL_MASK} \\
    {CORONAL_MASK} \\
    {SAGITTAL_MASK} \\
  --alpha {p["alpha"]} \\
  --outlier-rejection {p["outlier_rejection"]} \\
  --intensity-correction {p["intensity_correction"]} \\
  --two-step-cycles {p["two_step_cycles"]} \\
  --isotropic-resolution {p["isotropic_resolution"]} \\
  --output {REFINED_RECON_PATH} \\
  --verbose {p["verbose"]} \\
  --reconstruction-type {p["reconstruction_type"]}
""".strip()

    return cmd


def write_shell_script(script_path, command):
    """
    Write shell script for reproducibility.
    """
    script_text = f"""#!/usr/bin/env bash
set -euo pipefail

# Refined NiftyMIC reconstruction.
# Parameters:
#   alpha={NIFTYMIC_PARAMS["alpha"]}
#   outlier-rejection={NIFTYMIC_PARAMS["outlier_rejection"]}
#   intensity-correction={NIFTYMIC_PARAMS["intensity_correction"]}
#   two-step-cycles={NIFTYMIC_PARAMS["two_step_cycles"]}
#   reconstruction-type={NIFTYMIC_PARAMS["reconstruction_type"]}
#   isotropic-resolution={NIFTYMIC_PARAMS["isotropic_resolution"]}

{command}
"""

    with open(script_path, "w") as f:
        f.write(script_text)

    os.chmod(script_path, 0o755)
    print("Saved script:", script_path)


def run_command_streaming(command, log_path, cwd=None):
    """
    Run command through shell and stream output into Python console.
    Also save full log.
    """
    if cwd is None:
        cwd = os.getcwd()

    print("")
    print("=" * 80)
    print("Running NiftyMIC refined reconstruction")
    print("=" * 80)
    print("cwd:", cwd)
    print("log:", log_path)
    print("")
    print(command)
    print("")

    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_lines = []

    for line in process.stdout:
        print(line, end="")
        log_lines.append(line)

    return_code = process.wait()

    with open(log_path, "w") as f:
        f.writelines(log_lines)

    print("")
    print("=" * 80)
    print("NiftyMIC command finished")
    print("=" * 80)
    print("return code:", return_code)
    print("saved log:", log_path)

    if return_code != 0:
        raise RuntimeError(f"NiftyMIC failed with return code {return_code}")

    return log_path

# =============================================================================
# NIfTI inspection utilities
# =============================================================================

def image_world_corners(img):
    shape = np.asarray(img.shape[:3], dtype=int)

    corners_ijk = np.array(
        [
            [0, 0, 0],
            [shape[0] - 1, 0, 0],
            [0, shape[1] - 1, 0],
            [0, 0, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, 0],
            [shape[0] - 1, 0, shape[2] - 1],
            [0, shape[1] - 1, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, shape[2] - 1],
        ],
        dtype=float,
    )

    corners_h = np.c_[corners_ijk, np.ones(8)]
    return (img.affine @ corners_h.T).T[:, :3]


def image_world_bbox(img):
    corners = image_world_corners(img)
    return corners.min(axis=0), corners.max(axis=0)


def inspect_nifti(img, label="image"):
    data = img.get_fdata()
    finite = data[np.isfinite(data)]

    bmin, bmax = image_world_bbox(img)
    center = 0.5 * (bmin + bmax)
    extent = bmax - bmin

    print("")
    print("=" * 80)
    print(f"Inspect NIfTI: {label}")
    print("=" * 80)

    print("shape:", img.shape[:3])
    print("dtype:", img.get_data_dtype())
    print("zooms:", img.header.get_zooms()[:3])
    print("orientation:", nib.aff2axcodes(img.affine))
    print("qform code:", img.header["qform_code"])
    print("sform code:", img.header["sform_code"])

    print("")
    print("affine:")
    print(img.affine)

    print("")
    print("world min:", bmin)
    print("world max:", bmax)
    print("world center:", center)
    print("world extent mm:", extent)

    print("")
    print("finite voxels:", finite.size, "/", data.size)
    print("nan voxels:", np.count_nonzero(np.isnan(data)))
    print("inf voxels:", np.count_nonzero(np.isinf(data)))
    print("nonzero voxels:", np.count_nonzero(data))

    if finite.size > 0:
        print("min:", np.min(finite))
        print("max:", np.max(finite))
        print("mean:", np.mean(finite))
        print("std:", np.std(finite))

        percentiles = [0, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 100]
        pvals = np.percentile(finite, percentiles)

        print("")
        print("Percentiles:")
        for p, v in zip(percentiles, pvals):
            print(f"  p{p:>5}: {v:.6f}")


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


# =============================================================================
# Visualization utilities
# =============================================================================

def central_slices(data):
    cx, cy, cz = [s // 2 for s in data.shape[:3]]

    return {
        "axis0_mid": data[cx, :, :],
        "axis1_mid": data[:, cy, :],
        "axis2_mid": data[:, :, cz],
    }

def save_recon_qc_png(img, save_path, title="Refined SRR", clim_mode="percentile"):
    data = img.get_fdata().astype(np.float32)
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
    #close the figure to free memory

def browse_with_orthoslicer(img, title="SRR", clim_mode="percentile"):
    data = img.get_fdata().astype(np.float32)
    clim = choose_clim(data, mode=clim_mode)

    print("")
    print(f"Opening OrthoSlicer3D: {title}")
    print("backend:", matplotlib.get_backend())
    print("clim:", clim)

    try:
        slicer = OrthoSlicer3D(data)
        slicer.clim = clim
        slicer.show()
    except Exception as e:
        print("WARNING: OrthoSlicer3D failed.")
        print("Reason:", e)


def compare_two_recons(img_a, img_b, label_a="geometry-debug", label_b="refined"):
    """
    Basic numeric comparison if both reconstructions exist.
    """
    a = img_a.get_fdata().astype(np.float32)
    b = img_b.get_fdata().astype(np.float32)

    print("")
    print("=" * 80)
    print(f"Compare reconstructions: {label_a} vs {label_b}")
    print("=" * 80)

    print(f"{label_a} shape:", a.shape)
    print(f"{label_b} shape:", b.shape)

    if a.shape != b.shape:
        print("Shapes differ; skipping voxelwise difference.")
        return

    diff = b - a

    print("mean abs diff:", np.mean(np.abs(diff)))
    print("std diff:", np.std(diff))
    print("max abs diff:", np.max(np.abs(diff)))

    corr_mask = np.isfinite(a) & np.isfinite(b) & ((a != 0) | (b != 0))

    if np.count_nonzero(corr_mask) > 10:
        corr = np.corrcoef(a[corr_mask].ravel(), b[corr_mask].ravel())[0, 1]
        print("voxelwise correlation over nonzero union:", corr)


# =============================================================================
# Run Module 10
# =============================================================================

print("")
print("=" * 80)
print("Module 10 - Refined NiftyMIC reconstruction")
print("=" * 80)

VISUALIZE = False
SAVE_QC = True

results = []

for run_id, NIFTYMIC_PARAMS in enumerate(PARAM_COMBINATIONS, 1):

    run_name = make_run_name(NIFTYMIC_PARAMS)

    print("="*80)
    print(f"Run {run_id}/{len(PARAM_COMBINATIONS)}")
    print(run_name)
    print("="*80)

    REFINED_RECON_PATH = os.path.join(
        NIFTYMIC_INPUT_DIR,
        run_name + ".nii.gz",
    )

    REFINED_LOG_PATH = os.path.join(
        NIFTYMIC_INPUT_DIR,
        run_name + ".log",
    )

    REFINED_SCRIPT_PATH = os.path.join(
        NIFTYMIC_INPUT_DIR,
        run_name + ".sh",
    )

    QC_DIR = os.path.join(
        NIFTYMIC_INPUT_DIR,
        run_name + "_QC",
    )

    os.makedirs(QC_DIR, exist_ok=True)

    # --------------------------------------------------
    # ADD THIS BLOCK HERE
    # --------------------------------------------------
    if os.path.exists(REFINED_RECON_PATH):
        print(f"{run_name} already exists. Skipping...")
        continue

    cmd = build_niftymic_refined_command()

    write_shell_script(
        REFINED_SCRIPT_PATH,
        cmd,
    )

    run_command_streaming(
        cmd,
        REFINED_LOG_PATH,
        cwd=os.getcwd(),
    )

    if not os.path.exists(REFINED_RECON_PATH):
        print("FAILED")
        continue

    refined_img = nib.load(REFINED_RECON_PATH)

    inspect_nifti(
        refined_img,
        label=run_name,
    )

    if SAVE_QC:
        save_recon_qc_png(
            refined_img,
            save_path=os.path.join(
                QC_DIR,
                run_name + ".png",
            ),
            title=run_name,
            clim_mode=ORTHOSLICER_CLIM_MODE,
        )

    if VISUALIZE:

        browse_with_orthoslicer(
            refined_img,
            title=run_name,
            clim_mode=ORTHOSLICER_CLIM_MODE,
        )