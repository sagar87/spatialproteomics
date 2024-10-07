from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .constants import Dims, Layers


def load_image_data(
    image: np.ndarray,
    channel_coords: Union[str, List[str]],
    segmentation: Union[None, np.ndarray] = None,
    labels: Union[None, pd.DataFrame] = None,
    neighborhood: Union[None, pd.DataFrame] = None,
    cell_col: str = "cell",
    label_col: str = "label",
    neighborhood_col: str = "neighborhood",
    copy_image: bool = False,
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
    if copy_image:
        image = image.copy()

    if type(channel_coords) is str:
        channel_coords = [channel_coords]

    if image.ndim == 2:
        image = np.expand_dims(image, 0)

    channel_dim, y_dim, x_dim = image.shape

    assert len(channel_coords) == channel_dim, "Length of channel_coords must match image.shape[0]."

    if labels is not None:
        assert segmentation is not None, "Labels may only be provided in conjunction with a segmentation."
        assert (
            labels.shape[0] == np.unique(segmentation).shape[0] - 1
        ), f"Number of labels must match number of segments. Got {labels.shape[0]} labels, but segmentation contained {np.unique(segmentation).shape[0] - 1} cells."

    if neighborhood is not None:
        assert labels is not None, "Neighborhoods may only be provided in conjunction with labels."
        assert (
            neighborhood.shape[0] == labels.shape[0]
        ), f"Number of neighborhoods must match number of labels. Got {neighborhood.shape[0]} neighborhoods, but {labels.shape[0]} labels."

    im = xr.DataArray(
        image,
        coords=[channel_coords, range(y_dim), range(x_dim)],
        dims=[Dims.CHANNELS, Dims.Y, Dims.X],
        name=Layers.IMAGE,
    )

    dataset = xr.Dataset(data_vars={Layers.IMAGE: im})

    if segmentation is not None:
        dataset = dataset.pp.add_segmentation(segmentation)

        if labels is not None:
            dataset = dataset.la.add_labels_from_dataframe(labels, cell_col=cell_col, label_col=label_col)

            if neighborhood is not None:
                dataset = dataset.nh.add_neighborhoods_from_dataframe(neighborhood, neighborhood_col=neighborhood_col)

    else:
        dataset = xr.Dataset(data_vars={Layers.IMAGE: im})

    return dataset


def read_from_spatialdata(spatial_data_object, image_key: str = "image", segmentation_key: str = "segmentation"):
    """
    Read data from a spatialdata object into the spatialproteomics object.

    Parameters:
        spatial_data_object (spatialdata.SpatialData, str): The spatialdata object to read data from. If str, spatialdata is used to read the file from that path.
        image_key (str): The key of the image in the spatialdata object.
        segmentation_key (str): The key of the segmentation in the spatialdata object.

    Returns:
        spatialproteomics_object (xr.Dataset): The spatialproteomics object.
    """
    import spatialdata

    if isinstance(spatial_data_object, str):
        spatial_data_object = spatialdata.read_zarr(spatial_data_object)

    # image
    assert (
        image_key in spatial_data_object.images
    ), f"Image key {image_key} not found in spatial data object. Available keys: {list(spatial_data_object.images.keys())}"
    image = spatial_data_object.images[image_key]
    # coordinates
    markers = image.coords["c"].values

    # create the spatialproteomics object
    obj = load_image_data(image, channel_coords=markers)

    # segmentation
    if segmentation_key in spatial_data_object.labels:
        segmentation = spatial_data_object.labels[segmentation_key]
        obj = obj.pp.add_segmentation(segmentation)

        # obs (from anndata)
        if spatial_data_object.table is not None:
            obs = spatial_data_object.table.obs
            obj = obj.pp.add_obs_from_dataframe(obs)

    return obj
