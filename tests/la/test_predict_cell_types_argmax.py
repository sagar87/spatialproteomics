import numpy as np
import pandas as pd
import pytest

from spatialproteomics.constants import Dims, Labels, Layers, Props


def test_predict_cell_types_argmax(dataset_full):
    quantified = dataset_full.pp.add_quantification()

    ct_dict = {"T_CD4": "CD4", "T_CD8": "CD8"}
    quantified.la.predict_cell_types_argmax(ct_dict)

    # no quantification layer found
    with pytest.raises(
        AssertionError,
        match=f"Quantification layer with key {Layers.INTENSITY} not found. Please run pp.add_quantification",
    ):
        dataset_full.la.predict_cell_types_argmax(ct_dict)

    # not all markers found
    with pytest.raises(AssertionError, match="The following markers were not found in quantification layer"):
        ct_dict = {"T_CD4": "CD4", "dummy": "dummy"}
        quantified.la.predict_cell_types_argmax(ct_dict)


def test_predict_cell_types_argmax_without_overwriting_existing_annotations(dataset_full):
    # adding dummy labels, some of which are unassigned
    cells = dataset_full.coords[Dims.CELLS].values
    num_cells = len(cells)
    df = pd.DataFrame(
        {
            "cell": cells,
            "label": [Labels.UNLABELED] * 10 + ["CT1"] * (num_cells - 10),
        }
    )

    # adding the labels
    ds = dataset_full.pp.add_labels(df)

    # adding a quantification layer
    ds = ds.pp.add_quantification()

    # at this point, there should be some assigned and some unassigned cells
    assert np.all(np.unique(ds.coords["labels"].values) == np.array([0, 1]))
    assert "CT1" in ds[Layers.PROPERTIES].sel(props=Props.NAME).values
    assert Labels.UNLABELED in ds[Layers.PROPERTIES].sel(props=Props.NAME).values

    ct_dict = {"T_CD4": "CD4", "T_CD8": "CD8"}
    ds = ds.la.predict_cell_types_argmax(ct_dict)

    # checking that we have all cell types and no unassigned cells
    assert "CT1" in ds[Layers.PROPERTIES].sel(props=Props.NAME).values
    assert "T_CD4" in ds[Layers.PROPERTIES].sel(props=Props.NAME).values
    assert "T_CD8" in ds[Layers.PROPERTIES].sel(props=Props.NAME).values
    assert Labels.UNLABELED not in ds[Layers.PROPERTIES].sel(props=Props.NAME).values
    # this check implicitly checks if there are unassigned (0) cells left
    assert np.all(np.unique(ds.coords["labels"].values) == np.array([1, 2, 3]))
