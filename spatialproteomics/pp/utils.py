import warnings
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import scipy.ndimage
from scipy.stats import norm, zscore
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import relabel_sequential

from ..base_logger import logger
from ..constants import Dims


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
    segmentation_relabeled = np.vectorize(lambda x: value_map[x], otypes=[segmentation.dtype.char])(segmentation)

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
    amin: Optional[float] = None,
    amax: Optional[float] = None,
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
    amin: float
        Absolute value to perform normalization. If set, this overrides pmin.
    amax: float
        Absolute value to perform normalization. If set, this overrides pmax.

    Returns
    -------
    np.ndarray
        A min-max normalized image.
    """
    perc = np.percentile(img, [pmin, pmax], axis=(1, 2)).T

    if amin is not None:
        perc[:, 0] = amin
    if amax is not None:
        perc[:, 1] = amax

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


def _remove_outlying_cells(segmentation, dilation_size, threshold):
    # Create a binary mask where non-zero pixels are 1
    binary_mask = (segmentation != 0).astype(np.uint8)

    # Create the dilation kernel (elliptical shape)
    kernel_size = 2 * dilation_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply dilation to the binary mask
    dilated_mask = cv2.dilate(binary_mask, kernel)

    # Find connected components in the dilated mask
    num_clusters, cluster_labels = cv2.connectedComponents(dilated_mask, connectivity=8)

    # Flatten the original image and cluster labels for efficient indexing
    flat_image = segmentation.ravel()
    flat_cluster = cluster_labels.ravel()

    # Get unique cell labels and their first occurrence indices
    unique_cells, cell_indices = np.unique(flat_image, return_index=True)

    # Filter out the background (label 0)
    mask = unique_cells != 0
    unique_cells = unique_cells[mask]
    cell_indices = cell_indices[mask]

    # Map each cell to its corresponding cluster label
    cell_clusters = flat_cluster[cell_indices]
    cell_to_cluster = dict(zip(unique_cells, cell_clusters))

    # Group cells by their cluster
    cluster_to_cells = defaultdict(list)
    for cell in unique_cells:
        cluster = cell_to_cluster[cell]
        cluster_to_cells[cluster].append(cell)

    # Collect cells from clusters that are too small
    outlying_cells = set()
    for cluster, cells in cluster_to_cells.items():
        if len(cells) < threshold:
            outlying_cells.update(cells)

    # Identify the retained cells (cells not in outlying_cells)
    retained_cells = set(unique_cells) - outlying_cells

    # Return the indices of the retained cells
    return sorted(retained_cells)


def _compute_quantification(image, segmentation, func):
    image = np.rollaxis(image, 0, 3)
    measurements = []

    # Check if the input is a string (referring to a default skimage property)
    if isinstance(func, str):
        # Use regionprops to get the available property names
        try:
            props = regionprops_table(segmentation, intensity_image=image, properties=["label", func])
        except AttributeError:
            raise AttributeError(
                f"Invalid regionprop: {func}. Please provide a valid property or a custom function. Check skimage.measure.regionprops_table for available properties."
            )

        cell_idx = props.pop("label")
        for k in sorted(props.keys(), key=lambda x: int(x.split("-")[-1])):
            if k.startswith(func):
                measurements.append(props[k])
    # If the input is a callable (function)
    elif callable(func):
        props = regionprops_table(segmentation, intensity_image=image, extra_properties=(func,))
        cell_idx = props.pop("label")

        for k in sorted(props.keys(), key=lambda x: int(x.split("-")[-1])):
            if k.startswith(func.__name__):
                measurements.append(props[k])
    else:
        raise ValueError(
            "The func parameter should be either a string for default skimage properties or a callable function."
        )

    return np.array(measurements), np.array(cell_idx)


def _apply(image, func, **kwargs):
    # Apply the function independently across all channels
    # initially, I tried to vectorize this using xr.apply_ufunc(), but the results were spurious, esp. when applying a median filter
    processed_layers = []
    for channel in range(image.shape[0]):
        channel_data = image[channel]
        processed_channel_data = func(channel_data, **kwargs)
        processed_layers.append(processed_channel_data)

    # Stack the processed layers back into a single numpy array
    processed_layer = np.stack(processed_layers, 0)

    return processed_layer


def _threshold(
    image,
    quantile: Union[float, list] = None,
    intensity: Union[int, list] = None,
    channels: Optional[Union[str, list]] = None,
    shift: bool = True,
    channel_coord: str = Dims.CHANNELS,
):
    # note that image is an xarray object here, which has named channels
    if (quantile is None and intensity is None) or (quantile is not None and intensity is not None):
        raise ValueError("Please provide a quantile or absolute intensity cut off.")

    if isinstance(quantile, (float, int)):
        quantile = np.array([quantile])
    if isinstance(quantile, list):
        quantile = np.array(quantile)

    if isinstance(intensity, (float, int)):
        intensity = np.array([intensity])
    if isinstance(intensity, list):
        intensity = np.array(intensity)

    # if a channels argument is provided, the thresholds for all other channels are set to 0 (i. e. no thresholding)
    all_channels = image.coords[channel_coord].values.tolist()

    if channels is not None:
        if isinstance(channels, str):
            channels = [channels]

        assert all(
            channel in all_channels for channel in channels
        ), f"The following channels are not present in the image layer: {set(channels) - set(all_channels)}."

        if quantile is not None:
            assert len(channels) == len(quantile), "The number of channels must match the number of quantile values."
            quantile_dict = dict(zip(channels, quantile))
            quantile = np.array([quantile_dict.get(channel, 0) for channel in all_channels])
        if intensity is not None:
            assert len(channels) == len(intensity), "The number of channels must match the number of intensity values."
            intensity_dict = dict(zip(channels, intensity))
            intensity = np.array([intensity_dict.get(channel, 0) for channel in all_channels])
    else:
        # If no channels provided, assume the threshold applies to all channels
        if quantile is not None:
            assert len(quantile) == 1 or len(quantile) == len(
                all_channels
            ), "Quantile threshold must be a single value or a list of values with the same length as the number of channels. If you only want to threshold a subset of channels, you can use the channels argument."
            quantile = np.full(len(all_channels), quantile.item() if quantile.size == 1 else quantile)
        if intensity is not None:
            assert len(intensity) == 1 or len(intensity) == len(
                all_channels
            ), "Intensity threshold must be a single value or a list of values with the same length as the number of channels. If you only want to threshold a subset of channels, you can use the channels argument."
            assert np.all(intensity >= 0), "Intensity values must be positive."
            assert np.all(
                intensity <= np.max(image.values)
            ), "Intensity values must be smaller than the maximum intensity."
            intensity = np.full(len(all_channels), intensity.item() if intensity.size == 1 else intensity)

    if quantile is not None:
        assert np.all(quantile >= 0) and np.all(quantile <= 1), "Quantile values must be between 0 and 1."

        if shift:
            # calculate quantile (and ensure the correct dtypes in order to be more memory-efficient)
            # this is done by first clipping the values below the lower value, and subsequently subtracting the lower value from the result, which allows us to use the original dtype throughout
            lower = np.quantile(image.values.reshape(image.values.shape[0], -1), quantile, axis=1).astype(image.dtype)

            filtered = np.clip(
                image, a_min=np.expand_dims(np.diag(lower) if lower.ndim > 1 else lower, (1, 2)), a_max=None
            ).astype(image.dtype) - np.expand_dims(np.diag(lower) if lower.ndim > 1 else lower, (1, 2)).astype(
                image.dtype
            )
        else:
            # Calculate the quantile-based intensity threshold for each channel.
            flattened_values = image.values.reshape(
                image.values.shape[0], -1
            )  # Flatten height and width for each channel.
            lower = np.array(
                [np.quantile(flattened_values[i], q) for i, q in enumerate(quantile)]
            )  # Compute quantile per channel.

            # Reshape lower to match the broadcasting requirements.
            lower = lower[:, np.newaxis, np.newaxis]  # Reshape to add height and width dimensions.

            # Use np.where to apply the quantile threshold without shifting.
            filtered = np.where(image.values >= lower, image.values, 0)

    if intensity is not None:
        if shift:
            # calculate intensity
            filtered = (image - intensity.reshape(-1, 1, 1)).clip(min=0)
        else:
            # Reshape intensity to broadcast correctly across all dimensions.
            if len(intensity) == 1:
                intensity = intensity[0]  # This will make it a scalar for simple broadcasting.
            else:
                intensity = intensity[:, np.newaxis, np.newaxis]  # Add two new axes for broadcasting.
            # Apply thresholding: set all values below the intensity threshold to 0.
            filtered = np.where(image.values >= intensity, image.values, 0)

    return filtered


def _transform_expression_matrix(
    expression_matrix,
    method: str = "arcsinh",
    cofactor: float = 5.0,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
):
    # applying the appropriate transform
    if method == "arcsinh":
        transformed_matrix = np.arcsinh(expression_matrix / cofactor)
    elif method == "zscore":
        # z-scoring along each channel
        transformed_matrix = zscore(expression_matrix, axis=0)
    elif method == "minmax":
        # applying min max scaling, so that the lowest value is 0 and the highest is 1
        transformed_matrix = (expression_matrix - np.min(expression_matrix, axis=0)) / (
            np.max(expression_matrix, axis=0) - np.min(expression_matrix, axis=0)
        )
    elif method == "double_zscore":
        # z-scoring along each channel
        transformed_matrix = zscore(expression_matrix, axis=0)
        # z-scoring along each cell
        transformed_matrix = zscore(transformed_matrix, axis=1)
        # turning the z-scores into probabilities using the cumulative density function
        transformed_matrix = norm.cdf(transformed_matrix)
        # taking the negative log of the inverse probability to amplify positive values
        transformed_matrix = -np.log(1 - transformed_matrix)
    elif method == "clip":
        min_value, max_value = np.percentile(expression_matrix, [min_percentile, max_percentile])
        transformed_matrix = np.clip(expression_matrix, min_value, max_value)
    else:
        raise ValueError(f"Unknown transformation method: {method}")

    return transformed_matrix


def _validate_and_clamp_slice(start, stop, dim, slice_name="x_slice"):
    """
    Validate a slice [start, stop) against the coordinate axis 'dim'.

    - Assumes dim.values[0] .. dim.values[-1] are the valid (inclusive) coordinates.
    - Treats `start` inclusive, `stop` exclusive (so valid stop <= max + 1).
    - Raises ValueError if the slice is completely outside the image bounds.
    - Warns and clamps if the slice is partially outside.
    - Raises ValueError if start >= stop (empty/invalid slice).

    Returns (clamped_start, clamped_stop).
    """
    minv = dim.values[0]
    maxv = dim.values[-1]

    # sanity check on slice ordering
    if start >= stop:
        raise ValueError(f"{slice_name} is invalid: start ({start}) >= stop ({stop}).")

    # slice is entirely to the left if its last index (stop-1) < minv
    entirely_left = (stop - 1) < minv
    # slice is entirely to the right if its first index (start) > maxv
    entirely_right = start > maxv

    if entirely_left or entirely_right:
        raise ValueError(
            f"{slice_name} is out of bounds. You are trying to access coordinates "
            f"{start}:{stop} but the image has coordinates from {minv} to {maxv}."
        )

    # Now check for partial out-of-bounds and clamp
    clamped_start = max(start, minv)
    clamped_stop = min(stop, maxv + 1)  # stop is exclusive, so allow maxv+1

    if clamped_start != start or clamped_stop != stop:
        warnings.warn(
            f"{slice_name} is partially out of bounds. Requested {start}:{stop}, "
            f"image bounds are {minv}:{maxv}. Defaulting to {clamped_start}:{clamped_stop}."
        )

    return clamped_start, clamped_stop


def _merge_channels(arr: np.ndarray, method: Union[str, callable] = "max", normalize: bool = False) -> np.ndarray:
    dtype = arr.dtype
    # normalize if specified
    if normalize:
        # this normalization step squeezes all channels into values between 0 and 1
        arr = _normalize(arr, pmin=1, pmax=99, clip=True)
        # after normalization, we want to map this back to the original dtype, while also using the full range of the dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            arr = (arr * info.max).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            arr = arr.astype(dtype)
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")

    #  merge channels based on method
    if method == "max":
        merged = np.max(arr, axis=0)
    elif method == "sum":
        merged = np.sum(arr, axis=0)
    elif method == "mean":
        merged = np.mean(arr, axis=0)
    elif callable(method):
        merged = method(arr)
    else:
        raise ValueError(
            f"Unknown merging method: {method}. Please use 'max', 'sum', 'mean', or provide a callable function."
        )

    # ensure the merged array has the correct dtype (if the input was integer, output should also be integer, but we need to check for overflow)
    # if there is overflow, we change the dtype to int64
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if merged.max() > info.max:
            merged = merged.astype(np.uint64)
    else:
        merged = merged.astype(dtype)

    return merged
