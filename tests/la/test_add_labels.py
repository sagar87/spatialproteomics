import pandas as pd
import pytest

from spatialproteomics.constants import Dims, Features, Labels, Layers


def test_add_labels_from_dataframe_correct_annotation(dataset):
    # creating a dummy data frame
    cells = dataset.coords[Dims.CELLS].values
    num_cells = len(cells)
    df = pd.DataFrame(
        {
            "cell": cells,
            "label": ["CT1"] * num_cells,
        }
    )

    # adding the labels
    labeled = dataset.la.add_labels_from_dataframe(df)

    # checking that the labels were added
    assert Dims.LABELS in labeled.coords
    assert Features.LABELS in labeled[Layers.OBS].coords[Dims.FEATURES].values
    assert 1 in labeled[Layers.OBS].sel(features=Features.LABELS).values


def test_add_labels_from_dataframe_unassigned_cells(dataset):
    # creating a dummy data frame
    cells = dataset.coords[Dims.CELLS].values
    num_cells = len(cells)
    df = pd.DataFrame(
        {
            "cell": cells,
            "label": [Labels.UNLABELED] * 10 + ["CT1"] * (num_cells - 10),
        }
    )

    # adding the labels
    labeled = dataset.la.add_labels_from_dataframe(df)

    # checking that the labels were added
    assert Dims.LABELS in labeled.coords
    assert Features.LABELS in labeled[Layers.OBS].coords[Dims.FEATURES].values
    assert 0 in labeled[Layers.OBS].sel(features=Features.LABELS).values
    assert 1 in labeled[Layers.OBS].sel(features=Features.LABELS).values


def test_add_labels(dataset):
    # creating a dummy dict
    cells = dataset.coords[Dims.CELLS].values
    num_cells = len(cells)
    label_dict = dict(zip(cells, ["CT1"] * num_cells))

    # adding the labels
    labeled = dataset.la.add_labels(label_dict)

    # checking that the labels were added
    assert Dims.LABELS in labeled.coords
    assert Features.LABELS in labeled[Layers.OBS].coords[Dims.FEATURES].values
    assert 1 in labeled[Layers.OBS].sel(features=Features.LABELS).values


def test_add_labels_existing_labels(dataset_labeled):
    # creating a dummy dict
    cells = dataset_labeled.coords[Dims.CELLS].values
    num_cells = len(cells)
    label_dict = dict(zip(cells, ["CT1"] * num_cells))

    with pytest.raises(
        AssertionError,
        match="Already found label properties in the object.",
    ):
        dataset_labeled.la.add_labels(label_dict)
