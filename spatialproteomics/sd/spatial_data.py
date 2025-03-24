from typing import Optional

from spatialdata import SpatialData

from ..constants import Layers
from ..tl.utils import _cellpose


def cellpose(
    sdata: SpatialData,
    channel: Optional[str],
    image_key: str = "raw_image",
    key_added: str = Layers.SEGMENTATION,
    **kwargs,
):
    assert (
        image_key in sdata.images
    ), f"Image key {image_key} not found in spatial data object. Available keys: {list(sdata.images.keys())}"
    assert (
        key_added not in sdata.labels.keys()
    ), f"Key {key_added} already exists in spatial data object. Please choose another key."

    # access the image data
    image = sdata.images[image_key]

    # extract only the relevant channels
    image = image.sel(c=channel)
    # TODO: this is too custom at the moment, needs to also be able to handle the cases from the spatialdata docs
    # image = image['scale0']['image'].values

    # run cellpose
    segmentation_masks, _ = _cellpose(image, **kwargs)

    # add the segmentation masks to the spatial data object
    sdata.labels[key_added] = segmentation_masks
