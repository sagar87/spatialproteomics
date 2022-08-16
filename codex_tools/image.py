from typing import Union

import numpy as np
import xarray as xr
from skimage.measure import regionprops_table
from matplotlib.colors import LinearSegmentedColormap

from .constants import Dims, Layers


def sum_intensity(regionmask, intensity_image):
    return np.sum(intensity_image[regionmask])


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

        self_channels, self_x_dim, self_y_dim = self._obj._image.shape
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
        )
        im = xr.concat([self._obj._image, da], dim=Dims.IMAGE[0])
        return im

    def normalize(self, pmin=3, pmax=99.8, eps=1e-20, clip=False):
        """
        Performs a min max normalisation.
        """
        img = self._obj[Layers.IMAGE].values
        perc = np.percentile(img, [pmin, pmax], axis=(1, 2)).T

        norm = (img - np.expand_dims(perc[:, 0], (1, 2))) / (
            np.expand_dims(perc[:, 1] - perc[:, 0], (1, 2)) + eps
        )

        if clip:
            x = np.clip(x, 0, 1)

        return norm

    def colorize(
        self, colors=["C1", "C2", "C3", "C4", "C5"], background="black", normalize=True
    ):
        channel_dim = len(self._obj[Layers.IMAGE].coords[Dims.IMAGE[0]])
        cmaps = [
            LinearSegmentedColormap.from_list(c, [background, c], N=256)
            for c in colors[:channel_dim]
        ]
        normed = self.normalize()
        return [cmaps[i](normed[i]) for i in range(channel_dim)]

    def merge(self, colors=["C1", "C2", "C3", "C4", "C5"], proj="sum", alpha=0.5):
        images = self.colorize(colors=colors)

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

    def __getitem__(self, indices):

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

        im = self._obj[Layers.IMAGE].loc[c_slice, x_slice, y_slice]
        sg = self._obj[Layers.SEGMENTATION].loc[x_slice, y_slice]

        x_start = 0 if x_slice.start is None else x_slice.start
        y_start = 0 if y_slice.start is None else y_slice.start
        x_stop = -1 if x_slice.stop is None else x_slice.stop
        y_stop = -1 if y_slice.stop is None else y_slice.stop

        cell_idx = (
            (self._obj[Layers.COORDINATES].loc[:, "x"] > x_start)
            & (self._obj[Layers.COORDINATES].loc[:, "x"] < x_stop)
            & (self._obj[Layers.COORDINATES].loc[:, "y"] > y_start)
            & (self._obj[Layers.COORDINATES].loc[:, "y"] < y_stop)
        ).values

        co = self._obj[Layers.COORDINATES][cell_idx]
        da = self._obj[Layers.DATA][cell_idx].loc[:, c_slice]

        sub_ds = xr.Dataset(
            {
                Layers.IMAGE: im,
                Layers.SEGMENTATION: sg,
                Layers.COORDINATES: co,
                Layers.DATA: da,
            }
        )

        return sub_ds


@xr.register_dataset_accessor("se")
class SegmentationAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def get_coordinates(self):
        """
        Returns a xr.DataArray with the coordinates of each cell.
        """
        table = regionprops_table(
            self._obj[Layers.SEGMENTATION].values, properties=("label", "centroid")
        )
        dataset = xr.DataArray(
            np.stack([table["centroid-0"], table["centroid-1"]], -1),
            coords=[table["label"], ["x", "y"]],
            dims=Dims.COORDINATES,
        )
        return dataset

    def quantify(self, channels: Union[str, list] = "all"):
        """
        Adds channel(s) to an existing image container.
        """

        measurements = []
        all_channels = self._obj[Layers.IMAGE].coords[Dims.IMAGE[0]].values.tolist()

        for i, channel in enumerate(all_channels):

            props = regionprops_table(
                self._obj[Layers.SEGMENTATION].values,
                intensity_image=self._obj[Layers.IMAGE].loc[channel].values,
                extra_properties=(sum_intensity,),
            )

            if i == 0:
                cell_idx = props["label"]

            measurements.append(props[sum_intensity.__name__])

        ds = xr.DataArray(
            np.stack(measurements, -1),
            coords=[cell_idx, all_channels],
            dims=Dims.DATA,
        )

        return ds
