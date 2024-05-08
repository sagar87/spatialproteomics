import numpy as np
from skimage.segmentation import relabel_sequential

from ..base_logger import logger


def _format_labels(labels):
    """Formats a label list."""
    formatted_labels = labels.copy()
    unique_labels = np.unique(labels)

    if ~np.all(np.diff(unique_labels) == 1):
        logger.warning("Labels are non-consecutive. Relabeling...")
        formatted_labels, _, _ = relabel_sequential(formatted_labels)

    return formatted_labels
