from typing import List, Optional

import xarray as xr

from ..constants import Layers


# this file contains utility functions for working with spatialdata objects
def _get_channels_spatialdata(channel):
    if channel is not None:
        if isinstance(channel, list):
            channels = channel
        else:
            channels = [channel]
    else:
        return channel
    return channels


def _process_image(
    sdata,
    channels: Optional[List] = None,
    image_key: str = Layers.IMAGE,
    key_added: str = Layers.SEGMENTATION,
    return_values: bool = True,
    data_key: Optional[str] = None,
):
    assert (
        image_key in sdata.images
    ), f"Image key {image_key} not found in spatial data object. Available keys: {list(sdata.images.keys())}"
    if key_added is not None:
        assert (
            key_added not in sdata.labels.keys()
        ), f"Key {key_added} already exists in spatial data object. Please choose another key."

    # access the image data
    image = sdata.images[image_key]

    # handling weird spatialdata structures
    if isinstance(image, xr.DataTree):
        assert (
            data_key is not None
        ), f"It looks like your image is stored as a DataTree. Please provide a data_key to access the image data. Available keys are: {list(image.keys())}."
        assert (
            data_key.split("/")[0] in image.keys()
        ), f"Data key {data_key} not found in the image data. Available keys: {list(image.keys())}"

        image = image[data_key]  # Get the dataset node

        assert isinstance(
            image, xr.DataArray
        ), f"The image data should be a DataArray. Please provide a valid data key. Available keys are: {[data_key + '/' + x for x in list(image.keys())]}."

    # extract only the relevant channels
    if channels is not None:
        try:
            image = image.sel(c=channels)
        except KeyError:
            raise KeyError(
                f"Channels {channels} not found in the image data. Available channels: {list(image.c.values)}"
            )

    if return_values:
        # returning a numpy array
        return image.values
    # returning an xarray object
    return image


def _process_segmentation(sdata, segmentation_key: str = Layers.SEGMENTATION):
    assert (
        segmentation_key in sdata.labels
    ), f"Segmentation key {segmentation_key} not found in spatial data object. Available keys: {list(sdata.labels.keys())}"

    # access the segmentation mask
    segmentation = sdata.labels[segmentation_key]

    # returning a numpy array
    return segmentation.values


def _process_adata(sdata, table_key: str = "table"):
    assert (
        table_key in sdata.tables
    ), f"Tables key {table_key} not found in spatial data object. Available keys: {list(sdata.tables.keys())}. To add observations, please aggregate the intensities first."

    # access the segmentation mask
    tables = sdata.tables[table_key]

    # returning the adata object
    return tables
