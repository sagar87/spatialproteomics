from typing import Optional

import spatialdata

from ..constants import Layers
from ..tl.utils import _cellpose, _mesmer, _stardist
from .utils import _get_channels, _process_image


def cellpose(
    sdata: spatialdata.SpatialData,
    channel: Optional[str],
    image_key: str = "image",
    key_added: str = Layers.SEGMENTATION,
    **kwargs,
):
    channels = _get_channels(channel)

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels, image_key, key_added)

    # run cellpose
    segmentation_masks, _ = _cellpose(image, **kwargs)

    # add the segmentation masks to the spatial data object
    if segmentation_masks.shape[0] > 1:
        for i, channel in enumerate(channels):
            sdata.labels[f"{key_added}_{channel}"] = spatialdata.models.Labels2DModel.parse(
                segmentation_masks[i], transformations=None, dims=("y", "x")
            )
    else:
        sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
            segmentation_masks[0], transformations=None, dims=("y", "x")
        )


def stardist(
    sdata: spatialdata.SpatialData,
    channel: Optional[str],
    image_key: str = "image",
    key_added: str = Layers.SEGMENTATION,
    **kwargs,
):
    channels = _get_channels(channel)

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels, image_key, key_added)

    # run stardist
    segmentation_masks = _stardist(image, **kwargs)

    # add the segmentation masks to the spatial data object
    if segmentation_masks.shape[0] > 1:
        for i, channel in enumerate(channels):
            sdata.labels[f"{key_added}_{channel}"] = spatialdata.models.Labels2DModel.parse(
                segmentation_masks[i], transformations=None, dims=("y", "x")
            )
    else:
        sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
            segmentation_masks[0], transformations=None, dims=("y", "x")
        )


def mesmer(
    sdata: spatialdata.SpatialData,
    channel: Optional[str],
    image_key: str = "image",
    key_added: str = Layers.SEGMENTATION,
    **kwargs,
):
    channels = _get_channels(channel)

    assert (
        len(channels) == 2
    ), "Mesmer only supports two channel segmentation. Please ensure that the first channel is nuclear and the second one is membraneous."

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels, image_key, key_added)

    # run mesmer
    segmentation_masks = _mesmer(image, **kwargs)

    # add the segmentation masks to the spatial data object
    sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
        segmentation_masks[0].squeeze(), transformations=None, dims=("y", "x")
    )
