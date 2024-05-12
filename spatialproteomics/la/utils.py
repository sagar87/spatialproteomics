import numpy as np
from skimage.segmentation import relabel_sequential

from ..base_logger import logger


def _format_labels(labels):
    """
    Format the labels array to ensure consecutive numbering.

    Parameters
    ----------
    labels : numpy.ndarray
        The input array of labels.

    Returns
    -------
    numpy.ndarray
        The formatted array of labels with consecutive numbering.

    Notes
    -----
    This function checks if the input labels array contains non-consecutive numbers. If it does, it relabels the array
    to ensure consecutive numbering. A warning message is logged if relabeling is performed.
    """

    formatted_labels = labels.copy()
    unique_labels = np.unique(labels)

    if ~np.all(np.diff(unique_labels) == 1):
        logger.warning("Labels are non-consecutive. Relabeling...")
        formatted_labels, _, _ = relabel_sequential(formatted_labels)

    return formatted_labels
