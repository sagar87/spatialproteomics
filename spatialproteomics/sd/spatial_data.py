from typing import Callable, Optional, Union

import pandas as pd
import spatialdata
from anndata import AnnData
from skimage.measure import regionprops_table

from ..constants import Layers
from ..pp.utils import _apply, _compute_quantification, _threshold
from ..tl.utils import _cellpose, _mesmer, _stardist
from .utils import _get_channels, _process_adata, _process_image, _process_segmentation

# === SEGMENTATION ===


def cellpose(
    sdata: spatialdata.SpatialData,
    channel: Optional[str],
    image_key: str = Layers.IMAGE,
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
    image_key: str = Layers.IMAGE,
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
    image_key: str = Layers.IMAGE,
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


# === AGGREGATION AND PREPROCESSING ===
def add_quantification(
    sdata: spatialdata.SpatialData,
    func: Union[str, Callable] = "intensity_mean",
    key_added: str = Layers.INTENSITY,
    image_key: str = Layers.IMAGE,
    segmentation_key=Layers.SEGMENTATION,
):
    # sanity checks for image and segmentation
    image = _process_image(sdata, image_key=image_key, channels=None, key_added=None)
    segmentation = _process_segmentation(sdata, segmentation_key)

    # computing the quantification
    measurements, cell_idx = _compute_quantification(image, segmentation, func)

    # putting the result into the anndata object
    adata = AnnData(measurements.T)
    adata.obs["id"] = cell_idx
    adata.obs["region"] = segmentation_key

    # putting the anndata object into the spatialdata object
    sdata.tables[key_added] = adata


def add_observations(
    sdata: spatialdata.SpatialData,
    properties: Union[str, list, tuple] = ("label", "centroid"),
    segmentation_key=Layers.SEGMENTATION,
    table_key="table",
    **kwargs,
):
    segmentation = _process_segmentation(sdata, segmentation_key)
    adata = _process_adata(sdata, table_key)
    existing_features = adata.obs.columns

    if type(properties) is str:
        properties = [properties]

    if "label" not in properties:
        properties = ["label", *properties]

    table = regionprops_table(segmentation, properties=properties)

    # remove existing features
    table = pd.DataFrame({k: v for k, v in table.items() if k not in existing_features})

    # setting the label to be the index and removing it from the table
    table.index = table["label"]
    table = table.drop(columns="label")

    # add data into adata.obs
    # TODO: this needs to be more flexible
    adata.obs = adata.obs.merge(table, left_on="id", right_index=True, how="left")


def apply(
    sdata: spatialdata.SpatialData,
    func: Callable,
    image_key=Layers.IMAGE,
    **kwargs,
):
    image = _process_image(sdata, image_key=image_key, channels=None, key_added=None)
    processed_image = _apply(image, func, **kwargs)
    channels = sdata.images[image_key].coords["c"].values
    sdata.images[image_key] = spatialdata.models.Image2DModel.parse(
        processed_image, c_coords=channels, transformations=None, dims=("c", "y", "x")
    )


def threshold(
    sdata: spatialdata.SpatialData,
    image_key: str = Layers.IMAGE,
    quantile: Union[float, list] = None,
    intensity: Union[int, list] = None,
    channels: Optional[Union[str, list]] = None,
    shift: bool = True,
):
    # this gets the image as an xarray object
    image = _process_image(sdata, image_key=image_key, channels=None, key_added=None, return_values=False)
    processed_image = _threshold(
        image, quantile=quantile, intensity=intensity, channels=channels, shift=shift, channel_coord="c"
    )
    channels = sdata.images[image_key].coords["c"].values
    sdata.images[image_key] = spatialdata.models.Image2DModel.parse(
        processed_image, c_coords=channels, transformations=None, dims=("c", "y", "x")
    )
