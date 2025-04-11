import numpy as np
import pandas as pd
import pytest

from spatialproteomics.constants import Dims, Labels, Layers, Props

ct_dict = {"CD4": "T_CD4", "CD8": "T_CD8"}


def test_predict_cell_types_argmax(ds_labels):
    # drop cell types (_la_properties) for ct prediction
    ds = ds_labels.pp.drop_layers("_la_properties")

    ds.la.predict_cell_types_argmax(ct_dict)


def test_predict_cell_types_argmax_no_quantification(ds_segmentation):
    # no quantification layer found
    with pytest.raises(
        AssertionError,
        match=f"Quantification layer with key {Layers.INTENSITY} not found. Please run pp.add_quantification",
    ):
        ds_segmentation.la.predict_cell_types_argmax(ct_dict)


def test_predict_cell_types_argmax_invalid_markers(ds_labels):
    # drop cell types (_la_properties) for ct prediction
    ds = ds_labels.pp.drop_layers("_la_properties")

    # not all markers found
    with pytest.raises(AssertionError, match="The following markers were not found in quantification layer"):
        ct_dict = {"CD4": "T_CD4", "dummy": "dummy"}
        ds.la.predict_cell_types_argmax(ct_dict)


def test_predict_cell_types_argmax_without_overwriting_existing_annotations(ds_segmentation):
    # adding dummy labels, some of which are unassigned
    cells = ds_segmentation.coords[Dims.CELLS].values
    num_cells = len(cells)
    df = pd.DataFrame(
        {
            "cell": cells,
            "label": [Labels.UNLABELED] * 10 + ["CT1"] * (num_cells - 10),
        }
    )

    # adding the labels
    ds = ds_segmentation.la.add_labels_from_dataframe(df)

    # adding a quantification layer
    ds = ds.pp.add_quantification()

    # at this point, there should be some assigned and some unassigned cells
    assert np.all(np.unique(ds.coords["labels"].values) == np.array([0, 1]))
    assert "CT1" in ds[Layers.LA_PROPERTIES].sel(la_props=Props.NAME).values
    assert Labels.UNLABELED in ds[Layers.LA_PROPERTIES].sel(la_props=Props.NAME).values

    ct_dict = {"CD4": "T_CD4", "CD8": "T_CD8"}
    ds = ds.la.predict_cell_types_argmax(ct_dict)

    # checking that we have all cell types and no unassigned cells
    assert "CT1" in ds[Layers.LA_PROPERTIES].sel(la_props=Props.NAME).values
    assert "T_CD4" in ds[Layers.LA_PROPERTIES].sel(la_props=Props.NAME).values
    assert Labels.UNLABELED not in ds[Layers.LA_PROPERTIES].sel(la_props=Props.NAME).values
    # this check implicitly checks if there are unassigned (0) cells left
    assert np.all(np.unique(ds.coords["labels"].values) == np.array([1, 2]))
