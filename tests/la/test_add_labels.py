import numpy as np
import pandas as pd
import pytest

from spatialproteomics.constants import Dims, Features, Labels, Layers


def test_add_labels_from_dataframe_correct_annotation(ds_segmentation):
    # creating a dummy data frame
    cells = ds_segmentation.coords[Dims.CELLS].values
    num_cells = len(cells)
    df = pd.DataFrame(
        {
            "cell": cells,
            "label": ["CT1"] * num_cells,
        }
    )

    # adding the labels
    labeled = ds_segmentation.la.add_labels_from_dataframe(df)

    # checking that the labels were added
    assert Dims.LABELS in labeled.coords
    assert Features.LABELS in labeled[Layers.OBS].coords[Dims.FEATURES].values
    assert 1 in labeled[Layers.OBS].sel(features=Features.LABELS).values


def test_add_labels_from_dataframe_unassigned_cells(ds_segmentation):
    # creating a dummy data frame
    cells = ds_segmentation.coords[Dims.CELLS].values
    num_cells = len(cells)
    df = pd.DataFrame(
        {
            "cell": cells,
            "label": [Labels.UNLABELED] * 10 + ["CT1"] * (num_cells - 10),
        }
    )

    # adding the labels
    labeled = ds_segmentation.la.add_labels_from_dataframe(df)

    # checking that the labels were added
    assert Dims.LABELS in labeled.coords
    assert Features.LABELS in labeled[Layers.OBS].coords[Dims.FEATURES].values
    assert 0 in labeled[Layers.OBS].sel(features=Features.LABELS).values
    assert 1 in labeled[Layers.OBS].sel(features=Features.LABELS).values


def test_add_labels_from_dataframe_invalid_cells(ds_segmentation):
    # creating a dummy data frame
    cells = ds_segmentation.coords[Dims.CELLS].values
    num_cells = len(cells)
    df = pd.DataFrame(
        {
            "cell": [num_cells + 5],
            "label": ["CT1"],
        }
    )

    with pytest.raises(
        AssertionError,
        match="Could not find any overlap between the cells in the data frame",
    ):
        ds_segmentation.la.add_labels_from_dataframe(df)


def test_add_labels(ds_segmentation):
    # creating a dummy dict
    cells = ds_segmentation.coords[Dims.CELLS].values
    num_cells = len(cells)
    label_dict = dict(zip(cells, ["CT1"] * num_cells))

    # adding the labels
    labeled = ds_segmentation.la.add_labels(label_dict)

    # checking that the labels were added
    assert Dims.LABELS in labeled.coords
    assert Features.LABELS in labeled[Layers.OBS].coords[Dims.FEATURES].values
    assert 1 in labeled[Layers.OBS].sel(features=Features.LABELS).values


def test_add_labels_existing_labels(ds_labels):
    # creating a dummy dict
    cells = ds_labels.coords[Dims.CELLS].values
    num_cells = len(cells)
    label_dict = dict(zip(cells, ["CT1"] * num_cells))

    with pytest.raises(
        AssertionError,
        match="Already found label properties in the object.",
    ):
        ds_labels.la.add_labels(label_dict)


@pytest.mark.parametrize("labels", [["A", "B"], [1, 2]], ids=["str", "int"])
def test_add_labels_from_dataframe_partial_dataframe(ds_segmentation, labels):
    # labels for every second cell only, the remaining cells should be kept as unlabeled cells
    cells = ds_segmentation.coords[Dims.CELLS].values
    labeled_cells = cells[::2]
    df = pd.DataFrame({"cell": labeled_cells, "label": np.resize(labels, len(labeled_cells))})

    ds = ds_segmentation.la.add_labels_from_dataframe(df)

    # no cells are removed from obs or the segmentation mask
    np.testing.assert_array_equal(ds.coords[Dims.CELLS].values, cells)
    seg_cells = np.unique(ds[Layers.SEGMENTATION].values)
    np.testing.assert_array_equal(seg_cells[seg_cells > 0], cells)

    # the cells without a label are unlabeled
    obs = ds.pp.get_layer_as_df()
    unlabeled_cells = np.setdiff1d(cells, labeled_cells)
    assert (obs.loc[unlabeled_cells, Features.LABELS] == Labels.UNLABELED).all()
    assert (obs.loc[labeled_cells, Features.LABELS] != Labels.UNLABELED).all()
