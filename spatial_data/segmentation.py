from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

# from matplotlib.colors import LinearSegmentedColormap,
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from .base_logger import logger
from .constants import COLORS, Dims, Layers


def render_label(mask, img, cmap_mask, alpha=0.2, alpha_boundary=0):
    colored_mask = cmap_mask(mask)

    mask_bool = mask > 0
    mask_bound = np.bitwise_and(mask_bool, find_boundaries(mask, mode="thick"))

    # blend
    im = img.copy()

    im[mask_bool] = alpha * colored_mask[mask_bool] + (1 - alpha) * img[mask_bool]
    im[mask_bound] = (
        alpha_boundary * colored_mask[mask_bound]
        + (1 - alpha_boundary) * img[mask_bound]
    )

    return im


def sum_intensity(regionmask, intensity_image):
    return np.sum(intensity_image[regionmask])


def label_segmentation_mask(segmentation, annotation, label_col="type", cell_col="id"):
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
        Line2D(
            [0], [0], marker="o", color="w", label=t, markerfacecolor=c, markersize=15
        )
        for c, t in zip(colors, labels)
    ]
    return cmap, legend_elements


@xr.register_dataset_accessor("se")
class SegmentationAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def get_cell_num(self):
        return self._obj.coords[Dims.COORDINATES[0]].shape[0]

    def get_coordinates(self):
        """
        Returns a xr.DataArray with the coordinates of each cell.
        """
        table = regionprops_table(
            self._obj[Layers.SEGMENTATION].values, properties=("label", "centroid")
        )
        dataset = xr.DataArray(
            np.stack([table["centroid-1"], table["centroid-0"]], -1),
            coords=[table["label"], ["x", "y"]],
            dims=Dims.COORDINATES,
        )
        return dataset

    def quantify(self, channels: Union[str, list] = "all", func=sum_intensity):
        """
        Adds channel(s) to an existing image container.
        """

        measurements = []
        all_channels = self._obj.coords[Dims.IMAGE[0]].values.tolist()

        i = 0
        pbar = tqdm(all_channels)
        for channel in pbar:
            pbar.set_description(f"Processing {channel}")
            props = regionprops_table(
                self._obj[Layers.SEGMENTATION].values,
                intensity_image=self._obj[Layers.IMAGE].loc[channel].values,
                extra_properties=(func,),
            )

            if i == 0:
                cell_idx = props["label"]

            measurements.append(props[func.__name__])
            i += 1

        ds = xr.DataArray(
            np.stack(measurements, -1),
            coords=[cell_idx, all_channels],
            dims=Dims.DATA,
        )

        return ds

    def add_labels(self, labels, cell_col: str = "cell", label_col: str = "label"):
        num_cells = self._obj.se.get_cell_num()

        df = (
            pd.DataFrame(index=self._obj.coords[Dims.COORDINATES[0]].values)
            .reset_index()
            .rename(columns={"index": cell_col})
        )

        df = df.merge(labels, on=cell_col, how="inner")

        labels_arr = df.loc[:, label_col].values

        if 0 in labels_arr:
            logger.warning(
                "Found '0' as cell type as label, reindexing. Please ensure that cell type labels are consecutive integers (1, 2, ..., k) starting from 1."
            )
            labels_arr += 1

        da = xr.DataArray(
            labels_arr.reshape(-1, 1),
            coords=[df.loc[:, cell_col].values, ["label"]],
            dims=["cell_idx", "labels"],
            name="_labels",
        )

        ds = self._obj.merge(da, join="inner")

        num_cells_merged = ds.se.get_cell_num()
        lost_cells = num_cells - num_cells_merged

        if lost_cells > 0:
            logger.warning(f"No cell label for {lost_cells} cells. Dropping cells.")

        ds[Layers.SEGMENTATION].values[
            ~np.isin(ds[Layers.SEGMENTATION], ds.coords[Dims.COORDINATES[0]])
        ] = 0

        mask = label_segmentation_mask(
            ds[Layers.SEGMENTATION].values,
            labels,
            cell_col=cell_col,
            label_col=label_col,
        )

        da_mask = xr.DataArray(
            mask, coords=[ds.coords["x"], ds.coords["y"]], dims=["x", "y"]
        )
        ds["_labeled_segmentation"] = da_mask
        return ds

        # df = pd.DataFrame(
        #     self._obj[Layers.COORDINATES].values,
        #     index=ds[Layers.COORDINATES].coords[Dims.COORDINATES[0]].values
        #     )

        # xr.DataArray(
        #     labels[labels[cell_col].isin(cell_idx)].loc[;, label_col].values
        #     coords=
        # )
