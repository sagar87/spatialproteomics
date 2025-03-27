from typing import List, Optional

from spatialdata import SpatialData

from ..constants import Layers


def _get_channels(channel):
    if channel is not None:
        if isinstance(channel, list):
            channels = channel
        else:
            channels = [channel]
    else:
        return channel
    return channels


def _process_image(
    sdata: SpatialData,
    channels: Optional[List] = None,
    image_key: str = Layers.IMAGE,
    key_added: str = Layers.SEGMENTATION,
    return_values: bool = True,
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

    # extract only the relevant channels
    if channels is not None:
        image = image.sel(c=channels)
    # TODO: this is too custom at the moment, needs to also be able to handle the cases from the spatialdata docs
    # image = image['scale0']['image'].values

    if return_values:
        # returning a numpy array
        return image.values
    # returning an xarray object
    return image


def _process_segmentation(sdata: SpatialData, segmentation_key: str = Layers.SEGMENTATION):
    assert (
        segmentation_key in sdata.labels
    ), f"Segmentation key {segmentation_key} not found in spatial data object. Available keys: {list(sdata.labels.keys())}"

    # TODO: CHECK THE ANNDATA OBJECT HERE
    # assert (
    #    key_added not in sdata.labels.keys()
    # ), f"Key {key_added} already exists in spatial data object. Please choose another key."

    # access the segmentation mask
    segmentation = sdata.labels[segmentation_key]

    # TODO: this is too custom at the moment, needs to also be able to handle the cases from the spatialdata docs
    # image = image['scale0']['image'].values

    # returning a numpy array
    return segmentation.values


def _process_adata(sdata: SpatialData, table_key: str = "table"):
    assert (
        table_key in sdata.tables
    ), f"Tables key {table_key} not found in spatial data object. Available keys: {list(sdata.tables.keys())}. To add observations, please aggregate the intensities first."

    # access the segmentation mask
    tables = sdata.tables[table_key]

    # returning the adata object
    return tables
