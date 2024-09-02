import numpy as np


def is_positive(regionmask: np.ndarray, intensity_image: np.ndarray, threshold: float) -> float:
    """
    Determines whether a cell is positive based on the provided intensity image and threshold.

    Parameters
    ----------
    regionmask : numpy.ndarray
        Binary mask representing the region of interest (ROI) where cells are located.
    intensity_image : numpy.ndarray
        Intensity image representing the fluorescence signal of the cells.
    threshold : float
        Threshold value used for determining positivity. Cells with intensity values greater
        than this threshold are considered positive.

    Returns
    -------
    bool
        A boolean value indicating whether the cell is positive or not. Returns True if the fraction of
        positive pixels within the region of interest exceeds the provided threshold, otherwise returns False.
    """

    return (intensity_image[regionmask] > 0).sum() / (regionmask == 1).sum() > threshold


def percentage_positive(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    """
    Computes the percentage of positive pixels per label on the provided intensity image and region mask.

    Parameters
    ----------
    regionmask : numpy.ndarray
        Binary mask representing the region of interest (ROI) where cells are located.
    intensity_image : numpy.ndarray
        Intensity image representing the fluorescence signal of the cells.

    Returns
    -------
    float
        The percentage of positive cells within the region of interest. This is calculated as the ratio of the
        number of positive pixels (intensity greater than 0) to the total number of pixels in the region mask.
    """
    return (intensity_image[regionmask] > 0).sum() / (regionmask == 1).sum()
