import numpy as np
import pandas as pd
import pytest

from spatialproteomics.constants import Dims, Features, Labels, Layers, Props

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


def test_predict_cell_types_argmax_min_intensity_all_unassigned(ds_labels):
    """When min_intensity is very high, all cells should be unassigned."""
    ds = ds_labels.pp.drop_layers("_la_properties")

    ds = ds.la.predict_cell_types_argmax(ct_dict, min_intensity=1e9)

    # all cells should be unassigned
    assert np.all(ds.pp.get_layer_as_df()[Features.LABELS].values == Labels.UNLABELED)


def test_predict_cell_types_argmax_min_intensity_none_unassigned(ds_labels):
    """When min_intensity is 0 (default), behavior should be unchanged — no unassigned cells."""
    ds = ds_labels.pp.drop_layers("_la_properties")
    ds = ds.la.predict_cell_types_argmax(ct_dict, min_intensity=0.0)
    label_values = ds.pp.get_layer_as_df()[Features.LABELS].values
    # no cell should be unassigned
    assert np.all(label_values != Labels.UNLABELED)


def test_predict_cell_types_argmax_min_intensity_partial(ds_segmentation):
    """When min_intensity is set to a moderate value, only cells with all markers
    below the threshold should be unassigned."""
    ds = ds_segmentation.pp.add_quantification()
    # compute a threshold that only some cells fall below
    intensity_df = pd.DataFrame(ds[Layers.INTENSITY].values, columns=ds.coords[Dims.CHANNELS].values)
    markers = list(ct_dict.keys())
    max_per_cell = intensity_df[markers].max(axis=1)
    threshold = float(max_per_cell.quantile(0.25))

    ds = ds.la.predict_cell_types_argmax(ct_dict, min_intensity=threshold)
    label_values = ds.pp.get_layer_as_df()[Features.LABELS].values
    expected_unassigned = (max_per_cell < threshold).sum()

    # the number of unassigned cells should match the number below the threshold
    assert np.sum(label_values == Labels.UNLABELED) == expected_unassigned


def test_predict_cell_types_argmax_min_intensity_preserves_existing_annotations(ds_segmentation):
    """With overwrite_existing_labels=False, min_intensity should not overwrite existing annotations."""
    cells = ds_segmentation.coords[Dims.CELLS].values
    num_cells = len(cells)
    df = pd.DataFrame(
        {
            "cell": cells,
            "label": [Labels.UNLABELED] * 10 + ["CT1"] * (num_cells - 10),
        }
    )
    ds = ds_segmentation.la.add_labels_from_dataframe(df)
    ds = ds.pp.add_quantification()

    # use a very high threshold so argmax would assign everything as unassigned
    ds = ds.la.predict_cell_types_argmax(ct_dict, min_intensity=1e9, overwrite_existing_labels=False)

    label_names = ds[Layers.LA_PROPERTIES].sel(la_props=Props.NAME).values
    # CT1 cells should still be labeled, since they already had annotations
    assert "CT1" in label_names


def test_predict_cell_type_argmax_with_neighborhoods(ds_neighborhoods):
    with pytest.raises(AssertionError, match="Already found neighborhoods in the object"):
        _ = ds_neighborhoods.la.predict_cell_types_argmax(ct_dict)

    # this should work
    _ = ds_neighborhoods.la.predict_cell_types_argmax(ct_dict, ignore_neighborhoods=True)
