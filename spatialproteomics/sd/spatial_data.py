import copy as cp
from typing import Union

import pandas as pd
import spatialdata

from ..base_logger import logger
from ..constants import SDLayers
from ..la.utils import (
    _get_markers_from_subtype_dict,
    _predict_cell_subtypes,
    _predict_cell_types_argmax,
)
from .utils import _process_adata

# === AGGREGATION AND PREPROCESSING ===


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
