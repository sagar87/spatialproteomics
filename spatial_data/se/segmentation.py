from typing import Union

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay
from skimage.measure import regionprops_table
from tqdm import tqdm

from ..constants import Dims, Features, Layers
from .helper import sum_intensity

# from matplotlib.colors import LinearSegmentedColormap,
PROPS_DICT = {"centroid-1": Features.X, "centroid-0": Features.Y}


def _remove_unlabeled_cells(
    segmentation: np.ndarray, cells: np.ndarray, copy: bool = True
) -> np.ndarray:
    """Removes all cells from the segmentation that are not in cells."""
    if copy:
        segmentation = segmentation.copy()
    bool_mask = ~np.isin(segmentation, cells)
    segmentation[bool_mask] = 0

    return segmentation


@xr.register_dataset_accessor("se")
class SegmentationAccessor:
    """Handles everything that relates to the provided segmentation."""

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

    def quantify(
        self,
        channels: Union[str, list] = "all",
        segmentation: Union[np.ndarray, None] = None,
        func=sum_intensity,
        batch=True,
    ):
        """
        Adds channel(s) to an existing image container.
        """

        measurements = []
        all_channels = self._obj.coords[Dims.CHANNELS].values.tolist()

        if segmentation is None:
            segmentation = self._obj[Layers.SEGMENTATION].values

        if batch:
            image = np.rollaxis(self._obj[Layers.IMAGE].values, 0, 3)
            props = regionprops_table(
                segmentation, intensity_image=image, extra_properties=(func,)
            )
            cell_idx = props.pop("label")
            for k in sorted(props.keys(), key=lambda x: int(x.split("-")[-1])):
                if k.startswith(func.__name__):
                    # print(k)
                    measurements.append(props[k])
            # return props
        else:
            i = 0
            pbar = tqdm(all_channels)
            for channel in pbar:
                pbar.set_description(f"Processing {channel}")
                props = regionprops_table(
                    segmentation,
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

    def quantify_cells(self, cells: list):
        segmentation_layer = self._obj[Layers.SEGMENTATION]
        segmentation_mask = _remove_unlabeled_cells(segmentation_layer.values, cells)
        return self._obj.se.quantify(segmentation=segmentation_mask)

    def get_graph(self, graph_type="Delaunuay"):
        if graph_type == "Delaunuay":
            tri = Delaunay(self._obj[Layers.OBS].values)

        return tri
