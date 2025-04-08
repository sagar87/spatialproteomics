import copy as cp
from typing import Callable, Optional, Union

import pandas as pd
import spatialdata
from anndata import AnnData
from skimage.measure import regionprops_table

from ..base_logger import logger
from ..constants import SDLayers
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
from .utils import _process_adata, _process_image, _process_segmentation

# === SEGMENTATION ===


# === AGGREGATION AND PREPROCESSING ===
def add_quantification(
    sdata: spatialdata.SpatialData,
    func: Union[str, Callable] = "intensity_mean",
    key_added: str = SDLayers.TABLE,
    image_key: str = SDLayers.IMAGE,
    segmentation_key: str = SDLayers.SEGMENTATION,
    layer_key: Optional[str] = None,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
    This function computes the quantification of the image data based on the provided segmentation masks.
    It extracts the image data and segmentation masks from the spatialdata object, applies the quantification function,
    and adds the quantification results to the spatialdata object.
    The quantification results are stored in an AnnData object, which is added to the tables attribute of the spatialdata object.
    The quantification function can be specified as a string or a callable function.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data and segmentation masks.
        func (Union[str, Callable], optional): The quantification function to be applied. Defaults to "intensity_mean". Can be a string or a callable function.
        key_added (str, optional): The key under which the quantification results will be stored in the tables attribute of the spatialdata object. Defaults to table.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        segmentation_key (str, optional): The key for the segmentation masks in the spatialdata object. Defaults to segmentation.
        layer_key (Optional[str], optional): The key for the quantification results in the AnnData object. If None, a new layer will be created. Defaults to None.
        data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    # sanity checks for image and segmentation
    image = _process_image(
        sdata, image_key=image_key, channels=None, key_added=None, data_key=data_key, return_values=False
    )
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

    if copy:
        return sdata


def add_observations(
    sdata: spatialdata.SpatialData,
    properties: Union[str, list, tuple] = ("label", "centroid"),
    segmentation_key: str = SDLayers.SEGMENTATION,
    table_key: str = SDLayers.TABLE,
    copy: bool = False,
    **kwargs,
):
    """
    This function computes the observations for each region in the segmentation masks.
    It extracts the segmentation masks from the spatialdata object, computes the region properties,
    and adds the observations to the AnnData object stored in the tables attribute of the spatialdata object.
    The observations are computed using the regionprops_table function from skimage.measure.
    The properties to be computed can be specified as a string or a list/tuple of strings.
    The default properties are "label" and "centroid", but other properties can be added as well.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the segmentation masks.
        properties (Union[str, list, tuple], optional): The properties to be computed for each region. Defaults to ("label", "centroid").
        segmentation_key (str, optional): The key for the segmentation masks in the spatialdata object. Defaults to segmentation.
        table_key (str, optional): The key under which the AnnData object is stored in the tables attribute of the spatialdata object. Defaults to table.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

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
    adata.obs = adata.obs.merge(table, left_on="id", right_index=True, how="left")

    if copy:
        return sdata


def apply(
    sdata: spatialdata.SpatialData,
    func: Callable,
    image_key: str = SDLayers.IMAGE,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
    This function applies a given function to the image data in the spatialdata object.
    It extracts the image data from the spatialdata object, applies the function,
    and adds the processed image back to the spatialdata object.
    The processed image is stored in the images attribute of the spatialdata object.
    The function can be any callable function that takes an image as input and returns a processed image.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        func (Callable): The function to be applied to the image data. It should take an image as input and return a processed image.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the function.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    image = _process_image(
        sdata, image_key=image_key, channels=None, key_added=None, data_key=data_key, return_values=False
    )
    processed_image = _apply(image.values, func, **kwargs)
    channels = image.coords["c"].values
    sdata.images[image_key] = spatialdata.models.Image2DModel.parse(
        processed_image, c_coords=channels, transformations=None, dims=("c", "y", "x")
    )

    if copy:
        return sdata


def threshold(
    sdata: spatialdata.SpatialData,
    image_key: str = SDLayers.IMAGE,
    quantile: Union[float, list] = None,
    intensity: Union[int, list] = None,
    channels: Optional[Union[str, list]] = None,
    shift: bool = True,
    copy: bool = False,
    **kwargs,
):
    """
    This function applies a threshold to the image data in the spatialdata object.
    It extracts the image data from the spatialdata object, applies the thresholding function,
    and adds the processed image back to the spatialdata object.
    The processed image is stored in the images attribute of the spatialdata object.
    The thresholding function can be specified using the quantile or intensity parameters.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
        image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
        quantile (Union[float, list], optional): The quantile value(s) to be used for thresholding. If None, the intensity parameter will be used. Defaults to None.
        intensity (Union[int, list], optional): The intensity value(s) to be used for thresholding. If None, the quantile parameter will be used. Defaults to None.
        channels (Optional[Union[str, list]], optional): The channel(s) to be used for thresholding. If None, all channels will be used. Defaults to None.
        shift (bool, optional): Whether to shift the intensities towards 0 after thresholding. Defaults to True.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the thresholding function.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    # this gets the image as an xarray object
    image = _process_image(sdata, image_key=image_key, channels=None, key_added=None, return_values=False)
    processed_image = _threshold(
        image, quantile=quantile, intensity=intensity, channels=channels, shift=shift, channel_coord="c", **kwargs
    )
    channels = sdata.images[image_key].coords["c"].values
    sdata.images[image_key] = spatialdata.models.Image2DModel.parse(
        processed_image, c_coords=channels, transformations=None, dims=("c", "y", "x")
    )

    if copy:
        return sdata


def transform_expression_matrix(
    sdata: spatialdata.SpatialData,
    method: str = "arcsinh",
    table_key=SDLayers.TABLE,
    cofactor: float = 5.0,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
    copy: bool = False,
    **kwargs,
):
    """
    This function applies a transformation to the expression matrix in the spatialdata object.
    It extracts the expression matrix from the spatialdata object, applies the transformation function,
    and adds the transformed expression matrix back to the spatialdata object.
    The transformed expression matrix is stored in the tables attribute of the spatialdata object.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
        method (str, optional): The transformation method to be applied. Defaults to "arcsinh".
        table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
        cofactor (float, optional): The cofactor to be used for the transformation. Defaults to 5.0.
        min_percentile (float, optional): The minimum percentile to be used for the transformation. Defaults to 1.0.
        max_percentile (float, optional): The maximum percentile to be used for the transformation. Defaults to 99.0.
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the transformation function.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

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

    if copy:
        return sdata


def threshold_labels(
    sdata: spatialdata.SpatialData,
    threshold_dict: dict,
    key_added: str = SDLayers.BINARIZATION,
    table_key: str = SDLayers.TABLE,
    layer_key: str = "perc_pos",
    copy: bool = False,
):
    """
    Binarise based on a threshold. If a label is specified, the binarization is only applied to this cell type.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
        threshold_dict (dict): A dictionary containing the threshold values for each channel.
        key_added (str, optional): The key under which the processed expression matrix will be stored in the obsm attribute of the spatialdata object. Defaults to "binarization".
        table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
        layer_key (str, optional): The key under which the expression matrix is stored in the layers attribute of the spatialdata object. Defaults to "perc_pos".
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    adata = _process_adata(sdata, table_key=table_key)
    expression_df = adata.to_df(layer=layer_key)

    binarized_df = pd.DataFrame(
        {channel: (expression_df[channel] >= threshold).astype(int) for channel, threshold in threshold_dict.items()},
        index=expression_df.index,
    )

    adata.obsm[key_added] = binarized_df

    if copy:
        return sdata


# === CELL TYPE PREDICTION ===


def predict_cell_types_argmax(
    sdata: spatialdata.SpatialData,
    marker_dict: dict,
    table_key: str = SDLayers.TABLE,
    copy: bool = False,
):
    """
    This function predicts cell types based on the expression matrix using the argmax method.
    It extracts the expression matrix from the spatialdata object, applies the argmax method,
    and adds the predicted cell types to the spatialdata object.
    The predicted cell types are stored in the obs attribute of the AnnData object in the tables attribute of the spatialdata object.
    The argmax method assigns the cell type with the highest expression value to each cell.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
        marker_dict (dict): A dictionary containing the marker genes for each cell type.
        table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    adata = _process_adata(sdata, table_key=table_key)
    expression_df = adata.to_df()
    celltypes = _predict_cell_types_argmax(expression_df, list(marker_dict.keys()), list(marker_dict.values()))
    adata.obs["celltype"] = celltypes

    if copy:
        return sdata


def predict_cell_subtypes(
    sdata: spatialdata.SpatialData,
    subtype_dict: Union[dict, str],
    table_key: str = SDLayers.TABLE,
    layer_key: str = SDLayers.BINARIZATION,
    copy: bool = False,
):
    """
    This function predicts cell subtypes based on the expression matrix using a subtype dictionary.
    It extracts the expression matrix from the spatialdata object, applies the subtype prediction method,
    and adds the predicted cell subtypes to the spatialdata object.
    The predicted cell subtypes are stored in the obs attribute of the AnnData object in the tables attribute of the spatialdata object.

    Args:
        sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
        subtype_dict (Union[dict, str]): A dictionary mapping cell subtypes to the binarized markers used for prediction. Instead of a dictionary, a path to a yaml file containing the subtype dictionary can be provided.
        table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
        layer_key (str, optional): The key under which the binarized expression matrix is stored in the obsm attribute of the spatialdata object. Defaults to "binarization".
        copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

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

    if copy:
        return sdata
