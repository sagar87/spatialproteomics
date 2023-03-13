import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from skimage.segmentation import find_boundaries

from ..constants import COLORS


def render_label(mask, cmap_mask, img=None, alpha=0.2, alpha_boundary=0, mode="inner"):
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


def sum_intensity(regionmask, intensity_image):
    return np.sum(intensity_image[regionmask])


def label_segmentation_mask(
    segmentation: np.ndarray,
    annotation: pd.DataFrame,
    label_col: str = "type",
    cell_col: str = "id",
) -> np.ndarray:
    """
    Relabels a segmentation according to the annotations df (contains the columns type, cell).
    """
    labeled_segmentation = segmentation.copy()
    cell_types = annotation.loc[:, label_col].values.astype(int)
    cell_ids = annotation.loc[:, cell_col].values

    if 0 in cell_types:
        cell_types += 1

    for t in np.unique(cell_types):
        mask = np.isin(segmentation, cell_ids[cell_types == t])
        labeled_segmentation[mask] = t

    # remove cells that are not indexed
    neg_mask = ~np.isin(segmentation, cell_ids)
    labeled_segmentation[neg_mask] = 0

    return labeled_segmentation


def label_cells(raw_image, labeled_segmentation, cmap, **kwargs):
    return render_label(labeled_segmentation, img=raw_image, cmap=cmap, **kwargs)


def generate_cmap(num_cell_types, colors=COLORS, labels=None):
    cmap = ListedColormap(colors, N=num_cell_types)
    if labels is None:
        labels = ["BG"] + [f"Cell type {i}" for i in range(num_cell_types)]

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=t, markerfacecolor=c, markersize=15)
        for c, t in zip(colors, labels)
    ]
    return cmap, legend_elements
