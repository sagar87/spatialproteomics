import numpy as np


def sum_intensity(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    """
    Calculate the sum of intensity values within the specified regionmask.

    Parameters
    ----------
    regionmask : numpy.ndarray
        Binary mask representing the region of interest.
    intensity_image : numpy.ndarray
        Array containing the intensity values of the corresponding image.

    Returns
    -------
    numpy.float64
        The sum of intensity values within the specified regionmask.
    """
    return np.sum(intensity_image[regionmask])


def mean_intensity(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    """
    Calculate the mean of intensity values within the specified regionmask.

    Parameters
    ----------
    regionmask : numpy.ndarray
        Binary mask representing the region of interest.
    intensity_image : numpy.ndarray
        Array containing the intensity values of the corresponding image.

    Returns
    -------
    numpy.float64
        The mean of intensity values within the specified regionmask.
    """
    return np.mean(intensity_image[regionmask])


def arcsinh_mean_intensity(regionmask: np.ndarray, intensity_image: np.ndarray, cofactor: float = 5.0) -> float:
    """
    Calculate the arcsinh-transformed mean of intensity values within the specified regionmask.

    Parameters
    ----------
    regionmask : numpy.ndarray
        Binary mask representing the region of interest.
    intensity_image : numpy.ndarray
        Array containing the intensity values of the corresponding image.
    cofactor : numpy.float64, optional
        The cofactor used for the arcsinh transformation. Default is 5.

    Returns
    -------
    numpy.float64
        The arcsinh-transformed mean of intensity values within the specified regionmask.
    """
    return np.arcsinh(np.mean(intensity_image[regionmask]) / cofactor)


def arcsinh_median_intensity(regionmask: np.ndarray, intensity_image: np.ndarray, cofactor: float = 5.0) -> float:
    """
    Calculate the arcsinh-transformed median of intensity values within the specified regionmask.

    Parameters
    ----------
    regionmask : numpy.ndarray
        Binary mask representing the region of interest.
    intensity_image : numpy.ndarray
        Array containing the intensity values of the corresponding image.
    cofactor : numpy.float64, optional
        The cofactor used for the arcsinh transformation. Default is 5.

    Returns
    -------
    numpy.float64
        The arcsinh-transformed median of intensity values within the specified regionmask.
    """
    return np.arcsinh(np.median(intensity_image[regionmask]) / cofactor)


def arcsinh_var_intensity(regionmask: np.ndarray, intensity_image: np.ndarray, cofactor: float = 5.0) -> float:
    """
    Calculate the arcsinh-transformed variance of intensity values within the specified regionmask.

    Parameters
    ----------
    regionmask : numpy.ndarray
        Binary mask representing the region of interest.
    intensity_image : numpy.ndarray
        Array containing the intensity values of the corresponding image.
    cofactor : numpy.float64, optional
        The cofactor used for the arcsinh transformation. Default is 5.

    Returns
    -------
    numpy.float64
        The arcsinh-transformed variance of intensity values within the specified regionmask.
    """
    return np.arcsinh(np.var(intensity_image[regionmask]) / cofactor)


def arcsinh_sum_intensity(regionmask: np.ndarray, intensity_image: np.ndarray, cofactor: float = 5.0) -> float:
    """
    Calculate the arcsinh-transformed sum of intensity values within the specified regionmask.

    Parameters
    ----------
    regionmask : numpy.ndarray
        Binary mask representing the region of interest.
    intensity_image : numpy.ndarray
        Array containing the intensity values of the corresponding image.
    cofactor : numpy.float64, optional
        The cofactor used for the arcsinh transformation. Default is 5.

    Returns
    -------
    numpy.float64
        The arcsinh-transformed sum of intensity values within the specified regionmask.
    """
    return np.arcsinh(np.sum(intensity_image[regionmask]) / cofactor)


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
