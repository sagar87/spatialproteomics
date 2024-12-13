from typing import List, Tuple

import numpy as np
import scipy.ndimage
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential

from ..base_logger import logger


def merge(images: List[np.ndarray], proj: str = "sum", alpha: float = 0.5):
    """
    Merge multiple images into a single image using different projection methods.

    Parameters:
    - images (List[np.ndarray]): A list of images to be merged.
    - proj (str, optional): The projection method to be used. Default is "sum".
        - "sum": Sum the pixel values of the images.
        - "blend": Blend the images using alpha blending.
    - alpha (float, optional): The alpha value used in blending. Default is 0.5.

    Returns:
    - im_combined (np.ndarray): The merged image.
    """
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


def _remove_unlabeled_cells(segmentation: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """
    Remove unlabeled cells from the segmentation mask.

    Parameters:
    ----------
    segmentation : np.ndarray
        The segmentation array representing the labeled cells.
    cells : np.ndarray
        The array of cell labels to keep.

    Returns:
    -------
    np.ndarray
        The updated segmentation array with unlabeled cells removed.
    """
    segmentation_copy = segmentation.copy()
    bool_mask = ~np.isin(segmentation_copy, cells)
    segmentation_copy[bool_mask] = 0

    return segmentation_copy


def _relabel_cells(segmentation: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    This method relabels cells in the segmentation array, so that non-consecutive labels are turned into labels from 1 to n again.

    Parameters:
    ----------
    segmentation : np.ndarray
        The input segmentation array.

    Returns:
    -------
    Tuple[np.ndarray, dict]
        A tuple containing the relabeled segmentation array and a mapping dictionary.
    """
    # find unique cell IDs
    unique_values = np.unique(segmentation)

    # create a mapping from original values to new values
    value_map = {value: i for i, value in enumerate(unique_values)}

    # map the original array to the new values using the mapping
    segmentation_relabeled = np.vectorize(lambda x: value_map[x])(segmentation)

    return segmentation_relabeled, value_map


def _merge_segmentation(s1: np.ndarray, s2: np.ndarray, label1: int = 1, label2: int = 2, threshold: float = 1.0):
    """
    Merge two segmentation masks based on specified criteria.

    Parameters
    ----------
    s1 : numpy.ndarray
        First segmentation mask.
    s2 : numpy.ndarray
        Second segmentation mask.
    label1 : int
        Label for regions from the first mask in the final merged mask. Default is 1.
    label2 : int
        Label for regions from the second mask in the final merged mask. Default is 2.
    threshold : float, optional
        Threshold for area ratio of intersection over union for merging regions.
        Default is 1.0, meaning all regions from the second mask are merged.

    Returns
    -------
    numpy.ndarray
        Merged segmentation mask.
    dict
        Mapping of labels from the merged mask to the original labels.

    Notes
    -----
    This function assumes that `s1` and `s2` are 2D segmentation masks with integer labels.
    """
    s1 = s1.squeeze()
    s2 = s2.squeeze()

    n2, fmap, bmap = relabel_sequential(s2, offset=s1.max() + 1)
    s3 = s1.copy()
    s3[np.logical_and(n2 > 0, s1 == 0)] = n2[np.logical_and(n2 > 0, s1 == 0)]

    p1 = regionprops(s1)
    p2 = regionprops(n2)
    p3 = regionprops(s3)

    l1 = [p.label for p in p1]
    l2 = [p.label for p in p2]
    l3 = [p.label for p in p3]

    # compute intersections
    i1 = list(set(l1) & set(l3))
    i2 = list(set(l2) & set(l3))

    area_ratio = np.array([p.area for p in p3 if p.label in i2]) / np.array([p.area for p in p2 if p.label in i2])
    i2 = np.array(i2)[area_ratio >= threshold]
    selected_cells = np.concatenate([np.array(i1), i2])

    # make final mask
    clean_mask = _remove_unlabeled_cells(s3, selected_cells)
    final_mask, fmap, bmap = relabel_sequential(clean_mask)
    mapping = dict(zip([fmap[i] for i in selected_cells.astype(int)], [label1] * len(i1) + [label2] * len(i2)))

    return final_mask, mapping


def _normalize(
    img: np.ndarray,
    pmin: float = 3.0,
    pmax: float = 99.8,
    eps: float = 1e-20,
    clip: bool = False,
) -> np.ndarray:
    """
    Performs a min max normalisation.
    This function was adapted from the csbdeep package.

    Parameters
    ----------
    dataarray: xr.DataArray
        A xarray DataArray with an image field.
    pmin: float
        Lower quantile (min value) used to perform quantile normalization.
    pmax: float
        Upper quantile (max value) used to perform quantile normalization.
    eps: float
        Epsilon float added to prevent 0 division.
    clip: bool
        Ensures that normed image array contains no values greater than 1.

    Returns
    -------
    np.ndarray
        A min-max normalized image.
    """
    perc = np.percentile(img, [pmin, pmax], axis=(1, 2)).T

    norm = (img - np.expand_dims(perc[:, 0], (1, 2))) / (np.expand_dims(perc[:, 1] - perc[:, 0], (1, 2)) + eps)

    if clip:
        norm = np.clip(norm, 0, 1)

    return norm


def _check_for_disconnected_cells(segmentation: np.ndarray, handle: str = "error") -> bool:
    """
    Check for disconnected cells in a segmentation mask.

    Parameters:
    ----------
    segmentation : np.ndarray
        The segmentation mask to check for disconnected cells.
    handle : str, optional
        The handling option for disconnected cells. Can be 'error', 'warning', or 'ignore'.

    Returns:
    -------
    bool
        True if there are disconnected cells, False otherwise.

    Raises:
    ------
    ValueError
        If disconnected cells are found and the handle is set to 'error'.

    Warnings:
    ---------
    If disconnected cells are found and the handle is set to 'warning'.

    Notes:
    ------
    This method checks for disconnected cells in a segmentation mask. It returns True if there are disconnected cells,
    and False otherwise. The handle parameter determines how disconnected cells are handled. If handle is set to 'error',
    a ValueError is raised. If handle is set to 'warning', a warning is logged. If handle is set to 'ignore', the method
    returns True without raising an error or warning.
    """
    relabeled_mask = label(segmentation)
    num_cells = len(np.unique(segmentation))
    num_cells_relabeled = len(np.unique(relabeled_mask))

    if num_cells == num_cells_relabeled:
        return False
    else:
        if handle == "error":
            raise ValueError(
                "Found disconnected masks in the segmentation. Use pp.get_disconnected_cell() to get an example of a disconnected cell."
            )
        elif handle == "warning":
            logger.warning(
                "Found disconnected masks in the segmentation. Use pp.get_disconnected_cell() to to get an example of a disconnected cell."
            )
        return True


def handle_disconnected_cells(segmentation: np.ndarray, mode: str = "ignore"):
    """
    Handle disconnected cells in a segmentation mask.

    Parameters:
        segmentation (np.ndarray): The input segmentation mask.
        mode (str, optional): The mode to handle disconnected cells.
            - "ignore": Do nothing and keep the original segmentation mask.
            - "remove": Remove disconnected cells from the segmentation mask.
            - "relabel": Relabel cells to avoid disconnected cells.
            - "keep_largest": Keep only the largest component of each disconnected cell.

    Returns:
        np.ndarray: The updated segmentation mask.

    Raises:
        AssertionError: If the mode is not one of 'ignore', 'remove', 'relabel', 'keep_largest'.

    Notes:
        - This method checks if there are any disconnected cells in the segmentation mask.
        - If there are no disconnected cells, it returns the original segmentation mask.
        - The behavior depends on the chosen mode:
            - "ignore": Do nothing and return the original segmentation mask.
            - "remove": Remove disconnected cells by setting their values to 0.
            - "relabel": Relabel cells to avoid disconnected cells.
            - "keep_largest": Keep only the largest component of each disconnected cell.
    """
    assert mode in [
        "ignore",
        "remove",
        "relabel",
        "keep_largest",
    ], f"Could not recognize mode {mode}. Please choose one of 'ignore', 'remove', 'relabel', 'keep_largest'."

    # checking if there are any disconnected cells
    # if not, we simply do nothing
    contains_disconnected_components = _check_for_disconnected_cells(segmentation, "warning")
    if not contains_disconnected_components:
        return segmentation

    if mode == "ignore":
        return segmentation

    elif mode == "remove":
        num_removed_cells = 0
        for cell in sorted(np.unique(segmentation))[1:]:
            binary_mask = np.where(segmentation == cell, 1, 0)
            _, num_features = scipy.ndimage.label(binary_mask, structure=np.ones((3, 3)))
            if num_features != 1:
                segmentation[segmentation == cell] = 0
                num_removed_cells += 1
        logger.warning(f"Removed {num_removed_cells} disconnected cells from the segmentation mask.")

    elif mode == "keep_largest":
        for cell in sorted(np.unique(segmentation))[1:]:
            binary_mask = np.where(segmentation == cell, 1, 0)
            new_labels, num_features = scipy.ndimage.label(binary_mask, structure=np.ones((3, 3)))
            if num_features != 1:
                # Count the occurrences of each label excluding 0
                unique_labels, counts = np.unique(new_labels[new_labels != 0], return_counts=True)
                # Find the label with the most occurrences
                most_common_label = unique_labels[np.argmax(counts)]
                # Get all labels except the most common one
                other_labels = [label for label in unique_labels if label != most_common_label]
                # Create a mask to set every entity with this label to 0 and everything else to 1
                mask = np.where(np.isin(new_labels, other_labels), 1, 0)
                # Set segmentation to 0 for all positions where mask is 1
                segmentation = np.where(mask == 1, 0, segmentation)
        logger.warning("Kept largest component of each disconnected cell.")

    elif mode == "relabel":
        max_cell_id = np.max(segmentation)
        for cell in sorted(np.unique(segmentation))[1:]:
            binary_mask = np.where(segmentation == cell, 1, 0)
            new_labels, num_features = scipy.ndimage.label(binary_mask, structure=np.ones((3, 3)))
            if num_features != 1:
                # Count the occurrences of each label excluding 0
                unique_labels, counts = np.unique(new_labels[new_labels != 0], return_counts=True)
                for i, unique_label in enumerate(unique_labels):
                    # for the first entity, we simply keep the original label
                    if i == 0:
                        continue
                    # for any other entity, we give it a new ID
                    max_cell_id += 1
                    segmentation = np.where(new_labels == unique_label, max_cell_id, segmentation)
        logger.warning("Relabeled all cells to avoid disconnected cells.")

    return segmentation


def _get_disconnected_cell(segmentation: np.ndarray) -> int:
    """
    Find and return the first disconnected cell in the given segmentation.

    Parameters:
    segmentation (np.ndarray): The segmentation array representing the cells.

    Returns:
    int: The label of the first disconnected cell found, or None if no disconnected cell is found.
    """
    for cell in sorted(np.unique(segmentation))[1:]:
        binary_mask = np.where(segmentation == cell, 1, 0)
        _, num_features = scipy.ndimage.label(binary_mask)
        if num_features != 1:
            return cell


def _convert_to_8bit(image):
    """
    Convert an image to 8-bit format.

    Parameters:
    ----------
    image : np.ndarray
        The input image.

    Returns:
    -------
    np.ndarray
        The 8-bit image.
    """
    # if the image is already uint8, nothing happens
    if image.dtype == np.uint8:
        return image

    # checking that there are no negative values in the image
    assert np.min(image) >= 0, "The image contains negative values. Please make sure that the image is non-negative."

    # if the image is of type float, we check if the values are already in the range [0, 1]
    if image.dtype == np.float32 or image.dtype == np.float64:
        if np.max(image) <= 1:
            return (image * 255).astype(np.uint8)
        else:
            raise ValueError(
                "The image is of type float, but the values are not in the range [0, 1]. Please normalize the image first."
            )
    # checking if the integers are signed
    elif image.dtype == np.uint16:
        assert (
            np.max(image) <= 65535
        ), "The image contains values larger than 65535. Please make sure that the image is in the range [0, 65535]."
        # normalizing to the highest possible value
        return (image / 65535 * 255).astype(np.uint8)
    elif image.dtype == np.uint32:
        assert (
            np.max(image) <= 4294967295
        ), "The image contains values larger than 4294967295. Please make sure that the image is in the range [0, 4294967295]."
        # normalizing to the highest possible value
        return (image / 4294967295 * 255).astype(np.uint8)
    else:
        raise ValueError(
            f"Could not convert image of type {image.dtype} to 8-bit. Please make sure that the image is of type uint8, uint16, uint32, float32, or float64. If the image is of type float, make sure that the values are in the range [0, 1]."
        )


def _get_dtype_for_quantile(img_dtype):
    """
    Determines the appropriate dtype for the quantile result based on the image dtype.

    If img_dtype is unsigned integer, use the next higher signed integer precision.
    If img_dtype exceeds int32, use float64.
    Handles both NumPy and non-NumPy types.

    Parameters:
    ----------
    img_dtype : dtype
        The dtype of the input image.

    Returns:
    -------
    dtype
        The dtype for the quantile result.
    """
    # Ensure the dtype is resolved to a NumPy dtype
    img_dtype = np.dtype(img_dtype)

    if np.issubdtype(img_dtype, np.unsignedinteger):
        # Mapping from unsigned to the next higher signed type
        dtype_mapping = {np.uint8: np.int16, np.uint16: np.int32, np.uint32: np.int64}
        return dtype_mapping.get(img_dtype.type, np.float64)
    elif np.issubdtype(img_dtype, np.integer):
        # For signed integers, retain the same type
        return img_dtype
    elif np.issubdtype(img_dtype, np.floating):
        # For floating-point, retain the same type
        return img_dtype
    else:
        # Default to float64 for unknown or unsupported dtypes
        return np.float64
