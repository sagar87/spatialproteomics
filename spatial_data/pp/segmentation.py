import numpy as np


def sum_intensity(regionmask, intensity_image):
    return np.sum(intensity_image[regionmask])


def mean_intensity(regionmask, intensity_image):
    return np.mean(intensity_image[regionmask])


def arcsinh_mean_intensity(regionmask, intensity_image, cofactor=5):
    return np.arcsinh(np.mean(intensity_image[regionmask]) / cofactor)


def arcsinh_sum_intensity(regionmask, intensity_image, cofactor=5):
    return np.arcsinh(np.sum(intensity_image[regionmask]) / cofactor)


def _remove_unlabeled_cells(segmentation: np.ndarray, cells: np.ndarray, copy: bool = True) -> np.ndarray:
    """Removes all cells from the segmentation that are not in cells."""
    if copy:
        segmentation = segmentation.copy()
    bool_mask = ~np.isin(segmentation, cells)
    segmentation[bool_mask] = 0

    return segmentation
