from typing import List, Union

import numpy as np
import xarray as xr

from ..base_logger import logger
from ..constants import Attrs, Dims, Features, Layers
from .transforms import _colorize, _normalize


@xr.register_dataset_accessor("im")
class ImageAccessor:
    """The image accessor enables fast indexing and preprocesses image.data
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, indices):
        """Subsets the image container.
        """
        num_cells = self._obj.dims[Dims.CELLS]

        # argument handling
        if type(indices) is str:
            c_slice = [indices]
            x_slice = slice(None)
            y_slice = slice(None)
        elif type(indices) is slice:
            c_slice = slice(None)
            x_slice = indices
            y_slice = slice(None)
        elif type(indices) is list:
            all_str = all([type(s) is str for s in indices])

            if all_str:
                c_slice = indices
                x_slice = slice(None)
                y_slice = slice(None)
        elif type(indices) is tuple:
            all_str = all([type(s) is str for s in indices])

            if all_str:
                c_slice = [*indices]
                x_slice = slice(None)
                y_slice = slice(None)

            if len(indices) == 2:
                if (type(indices[0]) is slice) & (type(indices[1]) is slice):
                    c_slice = slice(None)
                    x_slice = indices[0]
                    y_slice = indices[1]

            elif len(indices) == 3:
                if type(indices[0]) is str:
                    c_slice = [indices[0]]
                elif type(indices[0]) is list:
                    c_slice = indices[0]
                else:
                    raise AssertionError("First index must index channel coordinates.")

                if (type(indices[1]) is slice) & (type(indices[2]) is slice):
                    x_slice = indices[1]
                    y_slice = indices[2]

        x_start = 0 if x_slice.start is None else x_slice.start
        y_start = 0 if y_slice.start is None else y_slice.start
        x_stop = -1 if x_slice.stop is None else x_slice.stop
        y_stop = -1 if y_slice.stop is None else y_slice.stop

        coords = self._obj[Layers.OBS]

        cells = (
            (coords.loc[:, Features.X] >= self._obj.coords[Dims.X][x_start])
            & (coords.loc[:, Features.X] <= self._obj.coords[Dims.X][x_stop])
            & (coords.loc[:, Features.Y] >= self._obj.coords[Dims.Y][y_start])
            & (coords.loc[:, Features.Y] <= self._obj.coords[Dims.Y][y_stop])
        ).values

        ds = self._obj.sel(
            {
                Dims.CHANNELS: c_slice,
                Dims.X: x_slice,
                Dims.Y: y_slice,
                Dims.CELLS: cells,
            }
        )

        lost_cells = num_cells - ds.dims[Dims.CELLS]

        if lost_cells > 0:
            logger.warning(f"Dropped {lost_cells} cells.")

        return ds

    def add_channel(self, channels: Union[str, list], array: np.ndarray):
        """
        Adds channel(s) to an existing image container.


        """
        assert type(array) is np.ndarray, "Added channel(s) must be numpy arrays"

        if array.ndim == 2:
            array = np.expand_dims(array, 0)

        if type(channels) is str:
            channels = [channels]

        self_channels, self_x_dim, self_y_dim = self._obj[Layers.IMAGE].shape
        other_channels, other_x_dim, other_y_dim = array.shape

        assert (
            len(channels) == other_channels
        ), "The length of channels must match the number of channels in array (DxMxN)."
        assert (self_x_dim == other_x_dim) & (
            self_y_dim == other_y_dim
        ), "Dims do not match."

        da = xr.DataArray(
            array,
            coords=[channels, range(other_x_dim), range(other_y_dim)],
            dims=Dims.IMAGE,
            name=Layers.IMAGE,
        )
        # im = xr.concat([self._obj[Layers.IMAGE], da], dim=Dims.IMAGE[0])

        return xr.merge([self._obj, da])

    def normalize(self):
        """Performs a percentile normalisation on each channel.

        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in Layers.PLOT.
        """
        image_layer = self._obj[Layers.IMAGE]
        normed = xr.DataArray(
            _normalize(image_layer.values),
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.PLOT,
        )

        return xr.merge([self._obj, normed])

    def colorize(
        self,
        colors: List[str] = ["C0", "C1", "C2", "C3"],
        background: str = "black",
        normalize: bool = True,
        merge=True,
    ) -> xr.Dataset:
        """Colorizes a stack of images.

        Parameters
        ----------
        colors: List[str]
            A list of strings that denote the color of each channel.
        background: float
            Background color of the colorized image.
        normalize: bool
            Normalizes the image prior to colorizing it.
        merge: True
            Merge the channel dimension.


        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in Layers.PLOT.
        """

        image_layer = self._obj[Layers.IMAGE]
        colored = _colorize(
            image_layer.values,
            colors=colors,
            background=background,
            normalize=normalize,
        )
        da = xr.DataArray(
            colored,
            coords=[
                image_layer.coords[Dims.CHANNELS],
                image_layer.coords[Dims.Y],
                image_layer.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs={
                Attrs.IMAGE_COLORS: {
                    k.item(): v
                    for k, v in zip(image_layer.coords[Dims.CHANNELS], colors)
                }
            },
        )

        if merge:
            da = da.sum(Dims.CHANNELS, keep_attrs=True)
            da.values[da.values > 1] = 1.0

        return xr.merge([self._obj, da])
