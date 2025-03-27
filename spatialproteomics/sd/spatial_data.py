from typing import Callable, Optional, Union

import pandas as pd
import spatialdata
from anndata import AnnData
from skimage.measure import regionprops_table
from spatialdata.transformations import get_transformation, set_transformation

from ..base_logger import logger
from ..constants import Layers
from ..la.utils import (
    _get_markers_from_subtype_dict,
    _predict_cell_subtypes,
    _predict_cell_types_argmax,
)
from ..pp.utils import (
    _apply,
    _compute_quantification,
    _threshold,
    _transform_expression_matrix,
)
from ..tl.utils import _astir, _cellpose, _mesmer, _stardist
from .utils import _get_channels, _process_adata, _process_image, _process_segmentation

# === SEGMENTATION ===


def cellpose(
    sdata: spatialdata.SpatialData,
    channel: Optional[str],
    image_key: str = Layers.IMAGE,
    key_added: str = Layers.SEGMENTATION,
    data_key: Optional[str] = None,
    **kwargs,
):
    channels = _get_channels(channel)

    # assert that the format is correct and extract the image
    image = _process_image(sdata, channels=channels, image_key=image_key, key_added=key_added, data_key=data_key)

    # run cellpose
    segmentation_masks, _ = _cellpose(image, **kwargs)

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    if segmentation_masks.shape[0] > 1:
        for i, channel in enumerate(channels):
            sdata.labels[f"{key_added}_{channel}"] = spatialdata.models.Labels2DModel.parse(
                segmentation_masks[i], transformations=None, dims=("y", "x")
            )
            set_transformation(sdata.labels[f"{key_added}_{channel}"], transformation)
    else:
        sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
            segmentation_masks[0], transformations=None, dims=("y", "x")
        )
        set_transformation(sdata.labels[key_added], transformation)

    sdata.labels


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

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    if segmentation_masks.shape[0] > 1:
        for i, channel in enumerate(channels):
            sdata.labels[f"{key_added}_{channel}"] = spatialdata.models.Labels2DModel.parse(
                segmentation_masks[i], transformations=None, dims=("y", "x")
            )
            set_transformation(sdata.labels[f"{key_added}_{channel}"], transformation)
    else:
        sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
            segmentation_masks[0], transformations=None, dims=("y", "x")
        )
        set_transformation(sdata.labels[key_added], transformation)


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

    # get transformations
    transformation = get_transformation(sdata[image_key])

    # add the segmentation masks to the spatial data object
    sdata.labels[key_added] = spatialdata.models.Labels2DModel.parse(
        segmentation_masks[0].squeeze(), transformations=None, dims=("y", "x")
    )
    set_transformation(sdata.labels[key_added], transformation)


# === AGGREGATION AND PREPROCESSING ===
def add_quantification(
    sdata: spatialdata.SpatialData,
    func: Union[str, Callable] = "intensity_mean",
    key_added: str = "table",
    image_key: str = Layers.IMAGE,
    segmentation_key=Layers.SEGMENTATION,
    layer_key: Optional[str] = None,
):
    # sanity checks for image and segmentation
    image = _process_image(sdata, image_key=image_key, channels=None, key_added=None, return_values=False)
    segmentation = _process_segmentation(sdata, segmentation_key)

    # computing the quantification
    measurements, cell_idx = _compute_quantification(image.values, segmentation, func)

    # checking if there already is an anndata object in the spatialdata object
    if key_added in sdata.tables:
        # if an anndata object (and hence a quantification) already exists, we add the new quantification to a new layer
        assert (
            layer_key is not None
        ), "An expression matrix already exists in your spatialdata object. Please provide a layer_key to add the new quantification to."
        assert (
            layer_key not in sdata.tables[key_added].layers
        ), f"Layer {layer_key} already exists in spatial data object. Please choose another key."
        sdata.tables[key_added].layers[layer_key] = measurements.T
    else:
        # if there is no anndata object yet, we create one
        adata = AnnData(measurements.T)
        adata.obs["id"] = cell_idx
        adata.obs["region"] = segmentation_key
        adata.var_names = image.coords["c"].values
        adata.obs_names = cell_idx
        adata.obs.index = adata.obs.index.astype(str)

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
    adata = _process_adata(sdata, table_key=table_key)
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
    **kwargs,
):
    # this gets the image as an xarray object
    image = _process_image(sdata, image_key=image_key, channels=None, key_added=None, return_values=False)
    processed_image = _threshold(
        image, quantile=quantile, intensity=intensity, channels=channels, shift=shift, channel_coord="c", **kwargs
    )
    channels = sdata.images[image_key].coords["c"].values
    sdata.images[image_key] = spatialdata.models.Image2DModel.parse(
        processed_image, c_coords=channels, transformations=None, dims=("c", "y", "x")
    )


def transform_expression_matrix(
    sdata: spatialdata.SpatialData,
    method: str = "arcsinh",
    table_key="table",
    cofactor: float = 5.0,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
    **kwargs,
):
    adata = _process_adata(sdata, table_key=table_key)
    transformed_matrix = _transform_expression_matrix(
        adata.X,
        method=method,
        cofactor=cofactor,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        **kwargs,
    )
    adata.X = transformed_matrix


def threshold_labels(
    sdata: spatialdata.SpatialData,
    threshold_dict: dict,
    key_added: str = "binarization",
    table_key: str = "table",
    layer_key: str = "perc_pos",
):
    adata = _process_adata(sdata, table_key=table_key)
    expression_df = adata.to_df(layer=layer_key)

    binarized_df = pd.DataFrame(
        {channel: (expression_df[channel] >= threshold).astype(int) for channel, threshold in threshold_dict.items()},
        index=expression_df.index,
    )

    adata.obsm[key_added] = binarized_df


# === CELL TYPE PREDICTION ===
def astir(
    sdata: spatialdata.SpatialData,
    marker_dict: dict,
    table_key="table",
    threshold: float = 0,
    seed: int = 42,
    learning_rate: float = 0.001,
    batch_size: float = 64,
    n_init: int = 5,
    n_init_epochs: int = 5,
    max_epochs: int = 500,
    cell_id_col: str = "cell_id",
    cell_type_col: str = "cell_type",
    **kwargs,
):
    adata = _process_adata(sdata, table_key=table_key)
    expression_df = adata.to_df()

    assigned_cell_types = _astir(
        expression_df=expression_df,
        marker_dict=marker_dict,
        threshold=threshold,
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_init=n_init,
        n_init_epochs=n_init_epochs,
        max_epochs=max_epochs,
        cell_id_col=cell_id_col,
        cell_type_col=cell_type_col,
    )

    # merging the resulting dataframe to the adata object
    df = pd.DataFrame(adata.obs)
    df = df.merge(assigned_cell_types, left_on="id", right_on=cell_id_col, how="left")
    adata.obs = df.drop(columns=cell_id_col)


def predict_cell_types_argmax(sdata: spatialdata.SpatialData, marker_dict: dict, table_key: str = "table"):
    adata = _process_adata(sdata, table_key=table_key)
    expression_df = adata.to_df()
    celltypes = _predict_cell_types_argmax(expression_df, list(marker_dict.keys()), list(marker_dict.values()))
    adata.obs["celltype"] = celltypes


def predict_cell_subtypes(
    sdata: spatialdata.SpatialData,
    subtype_dict: Union[dict, str],
    table_key: str = "table",
    layer_key: str = "binarization",
):
    adata = _process_adata(sdata, table_key=table_key)
    celltypes = adata.obs["celltype"].copy()
    assert (
        layer_key in adata.obsm
    ), f"Layer {layer_key} not found in adata object. Available layers: {list(adata.obsm.keys())}. Please run threshold_labels first."
    binarization_df = adata.obsm[layer_key]

    # these markers have a sign at the end, which indicates positivity or negativity
    markers_with_sign = _get_markers_from_subtype_dict(subtype_dict)
    # here, we only store the markers without the sign
    markers_for_subtype_prediction = [x[:-1] for x in markers_with_sign]

    # checking if all markers are binarized (this check needs to be removed if we want to still perform classification as far as we can)
    binarized_markers = binarization_df.columns
    if not all([marker in binarized_markers for marker in markers_for_subtype_prediction]):
        logger.warning(
            f"Did not find binarizations for the following markers: {[marker for marker in markers_for_subtype_prediction if marker not in binarized_markers]}."
        )

    # this method relies on the columns to have the suffix _binarized
    binarization_df.columns = [f"{marker}_binarized" for marker in binarization_df.columns]
    # adding the celltypes into the binarization df
    binarization_df["_labels"] = celltypes
    subtype_df = _predict_cell_subtypes(binarization_df, subtype_dict)
    subtype_df.index = subtype_df.index.astype(int)

    # adding the subtypes to the adata object
    sdata.tables[table_key].obs = sdata.tables[table_key].obs.merge(
        subtype_df, left_on="id", right_index=True, how="left"
    )
    # overwriting the celltype column
    adata.obs["celltype"] = subtype_df.iloc[:, -1].values
