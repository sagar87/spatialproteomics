import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure


def sum_intensity(regionmask, intensity_image):
    return np.sum(intensity_image[regionmask])


def mean_intensity(regionmask, intensity_image):
    return np.mean(intensity_image[regionmask])


def arcsinh_mean_intensity(regionmask, intensity_image, cofactor=5):
    return np.arcsinh(np.mean(intensity_image[regionmask]) / cofactor)


def arcsinh_var_intensity(regionmask, intensity_image, cofactor=5):
    return np.arcsinh(np.var(intensity_image[regionmask]) / cofactor)


def arcsinh_sum_intensity(regionmask, intensity_image, cofactor=5):
    return np.arcsinh(np.sum(intensity_image[regionmask]) / cofactor)


def detect_peaks_num(regionmask, intensity_image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    image = intensity_image  # [regionmask]
    # import pdb; pdb.set_trace()
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = image == 0

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks[regionmask].sum()


def _remove_unlabeled_cells(segmentation: np.ndarray, cells: np.ndarray, copy: bool = True) -> np.ndarray:
    """Removes all cells from the segmentation that are not in cells."""
    if copy:
        segmentation = segmentation.copy()
    bool_mask = ~np.isin(segmentation, cells)
    segmentation[bool_mask] = 0

    return segmentation
