import pandas as pd

from spatial_data.constants import Dims, Features, Labels, Layers


def test_add_labels_correct_annotation(dataset):
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
    labeled = dataset.pp.add_labels(df)

    # checking that the labels were added
    assert Dims.LABELS in labeled.coords
    assert Features.LABELS in labeled[Layers.OBS].coords[Dims.FEATURES].values
    assert 1 in labeled[Layers.OBS].sel(features=Features.LABELS).values


def test_add_labels_unassigned_cells(dataset):
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
    labeled = dataset.pp.add_labels(df)

    # checking that the labels were added
    assert Dims.LABELS in labeled.coords
    assert Features.LABELS in labeled[Layers.OBS].coords[Dims.FEATURES].values
    assert 0 in labeled[Layers.OBS].sel(features=Features.LABELS).values
    assert 1 in labeled[Layers.OBS].sel(features=Features.LABELS).values
