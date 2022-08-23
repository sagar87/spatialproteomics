from typing import List, Union

import numpy as np
import xarray as xr

from .constants import Dims, Layers


def hello_world(test):
    return f"Hello World {test}"


def load_image_data(
    image: np.ndarray,
    channel_coords: Union[str, List[str]],
    segmentaton_mask: Union[None, np.ndarray] = None,
):
    """Creates a image container.

    Creates an Xarray dataset with images, segmentation, and
    coordinate fields.

    Parameters
    ----------
    image : np.ndarray
        np.ndarray with image.shape = (n, x, y)
    channel_coords: str | List[str]
        list with the names for each channel

    Returns
    -------
    xr.Dataset
        An X-array dataset with all fields.
    """
    if type(channel_coords) is str:
        channel_coords = [channel_coords]

    if image.ndim == 2:
        image = np.expand_dims(image, 0)

    channel_dim, x_dim, y_dim = image.shape

    assert (
        len(channel_coords) == channel_dim
    ), "Length of channel_coords must match image.shape[0]."

    im = xr.DataArray(
        image,
        coords=[channel_coords, range(x_dim), range(y_dim)],
        dims=Dims.IMAGE,
    )

    if segmentaton_mask is None:
        segmentaton_mask = np.zeros((x_dim, y_dim)).astype(int)
    else:
        seg_x_dim, seg_y_dim = segmentaton_mask.shape
        assert (x_dim == seg_x_dim) & (
            y_dim == seg_y_dim
        ), f"The shape of segmentation mask ({seg_x_dim}, {seg_y_dim}) must match the x, y dims of the image ({x_dim}, {y_dim})."

    sg = xr.DataArray(
        segmentaton_mask.copy(),
        coords=[range(x_dim), range(y_dim)],
        dims=Dims.SEGMENTATION,
    )

    dataset = xr.Dataset(
        {
            Layers.IMAGE: im,
            Layers.SEGMENTATION: sg,
        }
    )

    coordinates = dataset.se.get_coordinates()
    dataset[Layers.COORDINATES] = coordinates

    # data = dataset.se.quantify()
    # dataset[Layers.DATA] = data

    return dataset
