from typing import List, Union

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border, find_boundaries, relabel_sequential

from ..constants import Dims
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
    # num_unique_values = len(unique_values)  # Get the number of unique values

    # Create a mapping from original values to new values
    value_map = {value: i for i, value in enumerate(unique_values)}

    # Map the original array to the new values using the mapping
    segmentation_relabeled = np.vectorize(lambda x: value_map[x])(segmentation)

    return segmentation_relabeled, value_map


def merge_segmentation(s1, s2, label1=1, label2=2, threshold=1.0):
    """
    Merge two segmentation masks based on specified criteria.

    Parameters
    ----------
    s1 : numpy.ndarray
        First segmentation mask.
    s2 : numpy.ndarray
        Second segmentation mask.
    label1 : int, optional
        Label for regions from the first mask in the final merged mask. Default is 1.
    label2 : int, optional
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
    mapping = dict(zip([fmap[i] for i in selected_cells], [label1] * len(i1) + [label2] * len(i2)))

    return final_mask, mapping


def _autocrop(sdata, channel=None, downsample=10):
    if channel is None:
        channel = sdata.coords[Dims.CHANNELS].values.tolist()[0]
    image = sdata.pp[channel].pp.downsample(downsample)._image.values.squeeze()

    bw = closing(image > np.quantile(image, 0.8), square(20))
    cleared = clear_border(bw)
    label_image = label(cleared)
    props = regionprops(label_image)
    if len(props) == 0:
        maxr, maxc = image.shape
        minr, minc = 0, 0
        downsample = 1
    else:
        max_idx = np.argmax([p.area for p in props])
        region = props[max_idx]
        minr, minc, maxr, maxc = region.bbox

    return slice(downsample * minc, downsample * maxc), slice(downsample * minr, downsample * maxr)
