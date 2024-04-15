from typing import List, Union

import numpy as np
from skimage.segmentation import find_boundaries

from ..pl import _get_linear_colormap


def _render_label(mask, cmap_mask, img=None, alpha=0.2, alpha_boundary=1.0, mode="inner"):
    colored_mask = cmap_mask(mask)

    mask_bool = mask > 0
    mask_bound = np.bitwise_and(mask_bool, find_boundaries(mask, mode=mode))

    # blend
    if img is None:
        img = np.zeros(mask.shape + (4,), np.float32)
        img[..., -1] = 1

    im = img.copy()

    im[mask_bool] = alpha * colored_mask[mask_bool] + (1 - alpha) * img[mask_bool]
    im[mask_bound] = alpha_boundary * colored_mask[mask_bound] + (1 - alpha_boundary) * img[mask_bound]

    return im


def _label_segmentation_mask(segmentation: np.ndarray, annotations: dict) -> np.ndarray:
    """
    Relabels a segmentation according to the annotations df (contains the columns type, cell).
    """
    labeled_segmentation = segmentation.copy()
    all_cells = []

    for k, v in annotations.items():
        mask = np.isin(segmentation, v)
        labeled_segmentation[mask] = k
        all_cells.extend(v)

    # remove cells that are not indexed
    neg_mask = ~np.isin(segmentation, all_cells)
    labeled_segmentation[neg_mask] = 0

    return labeled_segmentation


def merge(images, colors=["C1", "C2", "C3", "C4", "C5"], proj="sum", alpha=0.5):

    if proj == "sum":
        im_combined = np.sum(np.stack(images, axis=3), axis=3)
        im_combined[im_combined > 1] = 1
    elif proj == "blend":
        im_base = images[0].copy()
        for i in range(1, len(images)):
            alpha_a = images[i][:, :, 3][:, :, np.newaxis]
            alpha_a[alpha_a > 0] = alpha
            alpha_b = im_base[:, :, 3][:, :, np.newaxis]
            alpha_0 = alpha_a + alpha_b * (1 - alpha_a)
            im_combined = np.ones_like(images[0])
            im_combined[:, :, 0:3] = (
                images[i][:, :, 0:3] * alpha_a + im_base[:, :, 0:3] * alpha_b * (1 - alpha_a)
            ) / alpha_0
            im_base = im_combined

    return im_combined


def _remove_unlabeled_cells(segmentation: np.ndarray, cells: np.ndarray, copy: bool = True) -> np.ndarray:
    """Removes all cells from the segmentation that are not in cells."""
    if copy:
        segmentation = segmentation.copy()
    bool_mask = ~np.isin(segmentation, cells)
    segmentation[bool_mask] = 0

    return segmentation


def _relabel_cells(segmentation: np.ndarray):
    """
    Relabels cells in a segmentation array.

    Parameters:
    ----------
    segmentation : np.ndarray
        The input segmentation array.

    Returns:
    -------
    tuple[np.ndarray, dict]
        A tuple containing the relabeled segmentation array and a mapping dictionary.

    Notes:
    ------
    This method relabels cells in the segmentation array, so that non-consecutive labels are turned into labels from 1 to n again.
    This is important since CellSeg's mask growing relies on this assumption.

    The mapping dictionary provides a mapping from the original values to the new values.
    """
    unique_values = np.unique(segmentation)  # Find unique values in the array
    # num_unique_values = len(unique_values)  # Get the number of unique values

    # Create a mapping from original values to new values
    value_map = {value: i for i, value in enumerate(unique_values)}

    # Map the original array to the new values using the mapping
    segmentation_relabeled = np.vectorize(lambda x: value_map[x])(segmentation)

    return segmentation_relabeled, value_map
