from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from scipy.spatial import Delaunay
from skimage.measure import regionprops_table
from tqdm import tqdm

from ..constants import Attrs, Dims, Features, Layers
from .helper import label_segmentation_mask, render_label, sum_intensity

# from matplotlib.colors import LinearSegmentedColormap,
PROPS_DICT = {"centroid-1": Features.X, "centroid-0": Features.Y}


def _remove_unlabeled_cells(segmentation: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Removes all cells from the segmentation that are not in cells."""
    bool_mask = ~np.isin(segmentation, cells)
    segmentation[bool_mask] = 0

    return segmentation


@xr.register_dataset_accessor("se")
class SegmentationAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_segmentation(
        self, segmentation: np.ndarray, copy: bool = True
    ) -> np.ndarray:
        """Adds a segmentation mask (_segmentation) and an obs (_obs) field to the xarray dataset.

        Parameters
        ----------
        segmentation : np.ndarray
            A segmentation mask, i.e. a np.ndarray with image.shape = (n, x, y),
            that indicates the location of each cell.
        copy: bool
            If true the segmentation mask is copied.

        Returns
        -------
        xr.Dataset
            The amended xarray.
        """
        assert ~np.all(
            segmentation < 0
        ), "A segmentation mask may not contain negative numbers."

        y_dim, x_dim = segmentation.shape

        assert (x_dim == self._obj.dims[Dims.X]) & (
            y_dim == self._obj.dims[Dims.Y]
        ), "The shape of segmentation mask does not match that of the image."

        if copy:
            segmentation = segmentation.copy()

        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )

        self._obj[Layers.SEGMENTATION] = da
        self._obj = self._obj.se.add_obs("centroid")

        return self._obj

    def add_obs(
        self,
        properties: Union[str, list, tuple] = ("label", "centroid"),
        return_xarray: bool = False,
    ):
        if type(properties) is str:
            properties = [properties]

        if "label" not in properties:
            properties = ["label", *properties]

        table = regionprops_table(
            self._obj[Layers.SEGMENTATION].values, properties=properties
        )

        label = table.pop("label")
        data = []
        cols = []

        for k, v in table.items():
            col = PROPS_DICT.get(k, k)
            cols.append(col)
            data.append(v)

        da = xr.DataArray(
            np.stack(data, -1),
            coords=[label, cols],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        if Layers.OBS in self._obj:
            da = xr.concat(
                [self._obj[Layers.OBS].copy(), da],
                dim=Dims.FEATURES,
            )

        if return_xarray:
            return da

        return xr.merge([da, self._obj])

    def get_coordinates(self):
        """
        Returns a xr.DataArray with the coordinates of each cell.
        """
        table = regionprops_table(
            self._obj[Layers.SEGMENTATION].values, properties=("label", "centroid")
        )
        da = xr.DataArray(
            np.stack([table["centroid-1"], table["centroid-0"]], -1),
            coords=[table["label"], [Dims.X, Dims.Y]],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )
        return da

    def quantify(self, channels: Union[str, list] = "all", func=sum_intensity):
        """
        Adds channel(s) to an existing image container.
        """

        measurements = []
        all_channels = self._obj.coords[Dims.CHANNELS].values.tolist()

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
            dims=[Dims.CELLS, Dims.CHANNELS],
        )

        return ds

    def get_graph(self, graph_type="Delaunuay"):
        if graph_type == "Delaunuay":
            tri = Delaunay(self._obj[Layers.OBS].values)

        return tri

    def render_labels(self, alpha=0.2, alpha_boundary=0, mode="inner"):
        assert (
            Layers.LABELS in self._obj
        ), "Add labels via the add_labels function first."
        ds = self._obj
        # TODO: Attribute class in constants.py
        color_dict = self._obj[Layers.LABELS].attrs[Attrs.LABEL_COLORS]
        sorted_labels = sorted(color_dict.keys())

        # TODO: This needs to be refactored
        cmap = ListedColormap(
            ["black"] + [color_dict[k] for k in sorted_labels], N=len(sorted_labels)
        )

        mask = self.get_labeled_segmentation()

        if Layers.PLOT in self._obj:
            rendered = render_label(
                mask,
                cmap,
                self._obj[Layers.PLOT].values,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )
            attrs = self._obj[Layers.PLOT].attrs
            ds = ds.drop_vars(Layers.PLOT)
        else:
            rendered = render_label(
                mask, cmap, alpha=alpha, alpha_boundary=alpha_boundary, mode=mode
            )
            attrs = {}

        da = xr.DataArray(
            rendered,
            coords=[
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.Y, Dims.X, Dims.RGBA],
            attrs=attrs,
            name=Layers.PLOT,
        )

        return xr.merge([ds, da])

    def get_labeled_segmentation(
        self,
        labels: Union[None, pd.DataFrame] = None,
        cell_col: str = "cell",
        label_col: str = "label",
        src_layer: str = "_labels",
        dst_layer: str = "_labeled_segmentation",
    ) -> np.ndarray:
        if labels is None:
            labels = pd.DataFrame(
                {
                    cell_col: self._obj.coords[Dims.CELLS].values,
                    label_col: self._obj[Layers.LABELS].values.squeeze(),
                }
            )

        mask = label_segmentation_mask(
            self._obj[Layers.SEGMENTATION].values,
            labels,
            cell_col=cell_col,
            label_col=label_col,
        )

        return mask
