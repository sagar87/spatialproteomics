from typing import List, Union

import numpy as np
from skimage.segmentation import find_boundaries
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist

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


def _remove_segmentation_mask_labels(segmentation: np.ndarray, labels: Union[list, np.ndarray]) -> np.ndarray:
    """
    Relabels a segmentation according to the labels df (contains the columns type, cell).
    """
    labeled_segmentation = segmentation.copy()
    mask = ~np.isin(segmentation, labels)
    labeled_segmentation[mask] = 0

    return labeled_segmentation


def _normalize(
    img: np.ndarray,
    pmin: float = 3.0,
    pmax: float = 99.8,
    eps: float = 1e-20,
    clip: bool = False,
    name: str = "normed",
) -> np.ndarray:
    """Performs a min max normalisation.

    This function was adapted from the csbdeep package.

    Parameters
    ----------
    dataarray: xr.DataArray
        A xarray DataArray with an image field.
    pmin: float
        Lower quantile (min value) used to perform qunatile normalization.
    pmax: float
        Upper quantile (max value) used to perform qunatile normalization.
    eps: float
        Epsilon float added to prevent 0 division.
    clip: bool
        Ensures that normed image array contains no values greater than 1.

    Returns
    -------
    xr.DataArray
        A min-max normalized image.
    """
    perc = np.percentile(img, [pmin, pmax], axis=(1, 2)).T

    norm = (img - np.expand_dims(perc[:, 0], (1, 2))) / (np.expand_dims(perc[:, 1] - perc[:, 0], (1, 2)) + eps)

    if clip:
        norm = np.clip(norm, 0, 1)

    return norm


def _colorize(
    img: np.ndarray,
    colors: List[str] = ["C1", "C2", "C3", "C4", "C5"],
    background: str = "black",
    normalize: bool = True,
    name: str = "colored",
) -> np.ndarray:
    """Colorizes a stack of images

    Parameters
    ----------
    dataarray: xr.DataArray
        A xarray DataArray with an image field.
    clors: List[str]
        A list of strings that denote the color of each channel
    background: float
        Background color of the colorized image.
    normalize: bool
        Normalizes the image prior to colorizing it.

    Returns
    -------
    np.ndarray
        A colorized image
    """
    num_channels = img.shape[0]

    assert (
        len(colors) >= num_channels
    ), "Length of colors must at least be greater or equal the number of channels of the image."

    cmaps = _get_linear_colormap(colors[:num_channels], background)

    if normalize:
        img = _normalize(img)

    colored = np.stack([cmaps[i](img[i]) for i in range(num_channels)], 0)

    return colored


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
    num_unique_values = len(unique_values)  # Get the number of unique values

    # Create a mapping from original values to new values
    value_map = {value: i for i, value in enumerate(unique_values)}

    # Map the original array to the new values using the mapping
    segmentation_relabeled = np.vectorize(lambda x: value_map[x])(segmentation)

    return segmentation_relabeled, value_map


def _remove_overlaps_nearest_neighbors(original_masks, masks, centroids):
    """
    Remove overlaps between masks by assigning each overlapping pixel to mask with the closest centroid.

    Args:
        original_masks (ndarray): The original masks before growing.
        masks (ndarray): The grown masks with overlaps.
        centroids (ndarray): The centroids of the masks.

    Returns:
        ndarray: The masks with overlaps removed.

    """
    final_masks = np.max(masks, axis=2)
    collisions = np.nonzero(np.sum(masks > 0, axis=2) > 1)
    collision_masks = masks[collisions]
    collision_index = np.nonzero(collision_masks)
    collision_masks = collision_masks[collision_index]
    collision_frame = pd.DataFrame(np.transpose(np.array([collision_index[0], collision_masks]))).rename(
        columns={0: "collis_idx", 1: "mask_id"}
    )
    grouped_frame = collision_frame.groupby("collis_idx")
    for collis_idx, group in grouped_frame:
        collis_pos = np.expand_dims(np.array([collisions[0][collis_idx], collisions[1][collis_idx]]), axis=0)
        mask_ids = list(group["mask_id"])
        curr_centroids = np.array([centroids[mask_id - 1] for mask_id in mask_ids])
        dists = cdist(curr_centroids, collis_pos)
        closest_mask = mask_ids[np.argmin(dists)]
        final_masks[collis_pos[0, 0], collis_pos[0, 1]] = closest_mask

    # setting all values to the original masks so no masks get overwritten
    # get all pixels that were background in the original masks
    background_pixels = original_masks == 0
    # only reassigning cells which were previously background
    final_masks = np.array(final_masks * background_pixels, dtype=original_masks.dtype)
    # adding this growth to the original masks
    final_masks += original_masks
    return final_masks


def _grow_masks(flatmasks, centroids, growth, num_neighbors=30):
    """
    Grow the masks by performing dilation on each mask in the given flatmasks.

    Parameters
    ----------
    flatmasks : numpy.ndarray
        2D array representing the flat masks.
    centroids : numpy.ndarray
        Array of centroids corresponding to each mask.
    growth : int
        Number of times to perform dilation on the masks.
    num_neighbors : int, optional
        Number of nearest neighbors to consider for connectivity, by default 30.

    Returns
    -------
    numpy.ndarray
        2D array representing the grown masks.

    """
    masks = flatmasks
    num_masks = len(np.unique(masks)) - 1
    num_neighbors = min(num_neighbors, num_masks - 1)

    for _ in range(growth):
        # getting neighboring cells
        indices = np.where(masks != 0)
        values = masks[indices[0], indices[1]]
        maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(
            columns={0: "x", 1: "y", 2: "id"}
        )
        cent_array = maskframe.groupby("id").agg({"x": "mean", "y": "mean"}).to_numpy()
        connectivity_matrix = kneighbors_graph(cent_array, num_neighbors).toarray() * np.arange(1, num_masks + 1)
        connectivity_matrix = connectivity_matrix.astype(int)
        labels = {}
        for n in range(num_masks):
            connections = list(connectivity_matrix[n, :])
            connections.remove(0)
            layers_used = [labels[i] for i in connections if i in labels]
            layers_used.sort()
            currlayer = 0
            for layer in layers_used:
                if currlayer != layer:
                    break
                currlayer += 1
            labels[n + 1] = currlayer

        possible_layers = len(list(set(labels.values())))
        label_frame = pd.DataFrame(list(labels.items()), columns=["maskid", "layer"])
        image_h, image_w = masks.shape
        expanded_masks = np.zeros((image_h, image_w, possible_layers), dtype=np.uint32)

        grouped_frame = label_frame.groupby("layer")
        for layer, group in grouped_frame:
            currids = list(group["maskid"])
            masklocs = np.isin(masks, currids)
            expanded_masks[masklocs, layer] = masks[masklocs]

        dilation_mask = disk(1)
        grown_masks = np.copy(expanded_masks)
        for i in range(possible_layers):
            grown_masks[:, :, i] = dilation(grown_masks[:, :, i], dilation_mask)
        masks = _remove_overlaps_nearest_neighbors(masks, grown_masks, centroids)

    return masks
