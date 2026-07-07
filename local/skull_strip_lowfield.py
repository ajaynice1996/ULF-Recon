import numpy as np
from scipy import ndimage


def _largest_component(mask):
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == np.argmax(counts)


def _remove_border_components(mask, border_width):
    if border_width <= 0:
        return mask

    border = np.zeros(mask.shape, dtype=bool)
    border[:border_width, :, :] = True
    border[-border_width:, :, :] = True
    border[:, :border_width, :] = True
    border[:, -border_width:, :] = True
    border[:, :, :border_width] = True
    border[:, :, -border_width:] = True

    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask

    border_labels = np.unique(labels[border])
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return mask

    cleaned = mask.copy()
    cleaned[np.isin(labels, border_labels)] = False
    return cleaned


def _remove_small_components(mask, min_voxels):
    if min_voxels <= 0:
        return mask

    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask

    counts = np.bincount(labels.ravel())
    keep = counts >= min_voxels
    keep[0] = False
    return keep[labels]


def create_lowfield_brain_mask(
    volume,
    threshold_fraction=0.12,
    smooth_sigma=1.2,
    erosion_iters=1,
    closing_iters=5,
    opening_iters=1,
    final_dilation_iters=1,
    min_voxels=1000,
    trim_border=True,
    border_width=2,
):
    """Build a robust low-field MRI foreground/brain mask from one 3D volume."""
    image = np.nan_to_num(np.abs(volume).astype(np.float32), copy=False)
    nonzero = image[image > 0]
    if nonzero.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    p_low, p_high = np.percentile(nonzero, [2, 98])
    threshold = p_low + threshold_fraction * (p_high - p_low)

    smoothed = ndimage.gaussian_filter(image, sigma=smooth_sigma)
    mask = smoothed > threshold

    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    if opening_iters > 0:
        mask = ndimage.binary_opening(mask, structure=structure, iterations=opening_iters)

    mask = ndimage.binary_closing(mask, structure=structure, iterations=closing_iters)
    mask = ndimage.binary_fill_holes(mask)

    if trim_border:
        mask = _remove_border_components(mask, border_width=border_width)

    mask = _remove_small_components(mask, min_voxels=min_voxels)
    mask = _largest_component(mask)

    if erosion_iters > 0:
        mask = ndimage.binary_erosion(mask, structure=structure, iterations=erosion_iters)
        mask = _largest_component(mask)

    dilation_iters = erosion_iters + final_dilation_iters
    if dilation_iters > 0:
        mask = ndimage.binary_dilation(mask, structure=structure, iterations=dilation_iters)
        mask = ndimage.binary_fill_holes(mask)

    mask = _remove_small_components(mask, min_voxels=min_voxels)

    return mask.astype(np.uint8)


def apply_mask(volume, mask):
    return np.asarray(volume) * mask.astype(volume.dtype, copy=False)
