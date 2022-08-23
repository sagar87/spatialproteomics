from typing import List, Union

import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from .base_logger import logger
from .constants import Dims, Layers


def normalize(
    dataarray: xr.DataArray,
    pmin: float = 3.0,
    pmax: float = 99.8,
    eps: float = 1e-20,
    clip: bool = False,
) -> xr.DataArray:
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
    img = dataarray.values
    perc = np.percentile(img, [pmin, pmax], axis=(1, 2)).T

    norm = (img - np.expand_dims(perc[:, 0], (1, 2))) / (
        np.expand_dims(perc[:, 1] - perc[:, 0], (1, 2)) + eps
    )

    if clip:
        norm = np.clip(norm, 0, 1)

    return xr.DataArray(norm, coords=dataarray.coords, dims=Dims.IMAGE)


def colorize(
    dataarray: xr.DataArray,
    colors: List[str] = ["C1", "C2", "C3", "C4", "C5"],
    background: str = "black",
    normalize_img: bool = True,
) -> xr.DataArray:
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

    num_channels = len(dataarray.coords[Dims.IMAGE[0]])

    assert (
        len(colors) >= num_channels
    ), "Length of colors must at least be greater or equal the number of channels of the image."

    cmaps = [
        LinearSegmentedColormap.from_list(c, [background, c], N=256)
        for c in colors[:num_channels]
    ]

    if normalize_img:
        image = normalize(dataarray)
    else:
        image = dataarray

    da = xr.DataArray(
        np.stack([cmaps[i](image.values[i]) for i in range(num_channels)], 0),
        coords=[
            image.coords[Dims.IMAGE[0]],
            image.coords[Dims.IMAGE[1]],
            image.coords[Dims.IMAGE[2]],
            ["r", "g", "b", "a"],
        ],
        dims=Dims.COLORED_IMAGE,
        attrs=dict(colors=colors[:num_channels]),
    )
    return da


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
                images[i][:, :, 0:3] * alpha_a
                + im_base[:, :, 0:3] * alpha_b * (1 - alpha_a)
            ) / alpha_0
            im_base = im_combined

    return im_combined


@xr.register_dataset_accessor("im")
class ImageAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

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

    def __getitem__(self, indices):

        num_cells = self._obj.se.get_cell_num()

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

        # im = self._obj[Layers.IMAGE].loc[c_slice, x_slice, y_slice]
        # sg = self._obj[Layers.SEGMENTATION].loc[x_slice, y_slice]

        x_start = 0 if x_slice.start is None else x_slice.start
        y_start = 0 if y_slice.start is None else y_slice.start
        x_stop = -1 if x_slice.stop is None else x_slice.stop
        y_stop = -1 if y_slice.stop is None else y_slice.stop

        cell_idx = (
            (
                self._obj[Layers.COORDINATES].loc[:, "x"]
                >= self._obj.coords["x"][x_start]
            )
            & (
                self._obj[Layers.COORDINATES].loc[:, "x"]
                <= self._obj.coords["x"][x_stop]
            )
            & (
                self._obj[Layers.COORDINATES].loc[:, "y"]
                >= self._obj.coords["y"][y_start]
            )
            & (
                self._obj[Layers.COORDINATES].loc[:, "y"]
                <= self._obj.coords["y"][y_stop]
            )
        ).values

        # co = self._obj[Layers.COORDINATES][cell_idx]
        # da = self._obj[Layers.DATA][cell_idx].loc[:, c_slice]

        # sub_ds = xr.Dataset(
        #     {
        #         Layers.IMAGE: im,
        #         Layers.SEGMENTATION: sg,
        #         Layers.COORDINATES: co,
        #         Layers.DATA: da,
        #     }
        # )

        # if '_labels' in self._obj:
        #     sub_ds['_labels'] = self._obj['_labels']
        # if '_labeled_segmentation' in self._obj:
        #     sub_ds['_labeled_segmentation'] = self._obj['_labeled_segmentation']

        sliced = self._obj.sel(
            dict(channels=c_slice, x=x_slice, y=y_slice, cell_idx=cell_idx)
        )

        sliced_cell_num = sliced.se.get_cell_num()
        lost_cells = num_cells - sliced_cell_num

        if lost_cells > 0:
            logger.warning(f"Dropped {lost_cells} cells.")

        return sliced
