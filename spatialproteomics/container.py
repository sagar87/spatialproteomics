from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base_logger import logger
from .constants import Dims, Layers


def load_image_data(
    image: np.ndarray,
    channel_coords: Union[str, List[str]],
    x_coords: Union[None, np.ndarray] = None,
    y_coords: Union[None, np.ndarray] = None,
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

    if x_coords is None:
        x_coords = np.arange(x_dim)
    if y_coords is None:
        y_coords = np.arange(y_dim)

    im = xr.DataArray(
        image,
        coords=[channel_coords, y_coords, x_coords],
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


def read_from_spatialdata(
    spatial_data_object, image_key: str = "image", segmentation_key: str = "segmentation", table_key: str = "table"
):
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
    from spatialdata.transformations import Affine, Translation, get_transformation

    if isinstance(spatial_data_object, str):
        spatial_data_object = spatialdata.read_zarr(spatial_data_object)

    # image
    assert (
        image_key in spatial_data_object.images
    ), f"Image key {image_key} not found in spatial data object. Available keys: {list(spatial_data_object.images.keys())}"
    image = spatial_data_object.images[image_key]
    # coordinates
    markers = image.coords["c"].values

    # we have to ensure that we get the right coordinates for x and y as well
    # get transform
    transform = get_transformation(spatial_data_object[image_key])

    # default coords
    y_coords = np.arange(image.shape[1])
    x_coords = np.arange(image.shape[2])

    # apply transform if exists
    if isinstance(transform, Translation):
        shift_y, shift_x = transform.translation
        y_coords = y_coords + shift_y
        x_coords = x_coords + shift_x

    elif isinstance(transform, Affine):
        matrix = transform.matrix

        a, b, c = matrix[0, :]
        d, e, f = matrix[1, :]

        if not np.isclose(b, 0) or not np.isclose(d, 0):
            logger.warning(
                "Affine transformation has shear components, which are not supported in spatialproteomics. Resetting coordinates to start from 0."
            )
        else:
            x_coords = a * x_coords + c
            y_coords = e * y_coords + f
    else:
        logger.warning(f"Unsupported transform {transform}, resetting coordinates for the spatialproteomics object.")

    # create the spatialproteomics object
    obj = load_image_data(image, channel_coords=markers, x_coords=x_coords, y_coords=y_coords)

    # segmentation
    if segmentation_key in spatial_data_object.labels:
        segmentation = spatial_data_object.labels[segmentation_key]
        # if there are obs in the spatialdata object, we just use those and do not recompute them
        add_obs_from_sdata = len(spatial_data_object.tables.keys()) > 0
        obj = obj.pp.add_segmentation(segmentation, add_obs=not add_obs_from_sdata)

        # obs (from anndata)
        if add_obs_from_sdata:
            obs = spatial_data_object.tables[table_key].obs
            obj = obj.pp.add_obs_from_dataframe(obs)

    return obj
