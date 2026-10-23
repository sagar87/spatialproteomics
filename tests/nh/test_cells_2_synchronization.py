import numpy as np
import pandas as pd
import pytest

from spatialproteomics.constants import Dims, Features, Layers

# The adjacency matrix computed by the nh module is stored with the dimensions (cells, cells_2),
# because xarray does not support duplicate dimension names. Row i and column i of the matrix
# must refer to the same cell, so every operation that removes or relabels cells needs to update
# both the cells and the cells_2 coordinate.


def _mask_left_half(ds):
    mask = np.ones((ds.sizes[Dims.Y], ds.sizes[Dims.X]), dtype=int)
    mask[:, : ds.sizes[Dims.X] // 2] = 0
    return ds.pp.add_layer(mask, key_added=Layers.MASK)


def _median_x(ds):
    return float(np.median(ds.pp.get_layer_as_df()[Features.X]))


def _partial_labels(ds):
    # labels for every second cell only, the remaining cells are dropped by add_labels_from_dataframe
    cells = ds.coords[Dims.CELLS].values[::2]
    df = pd.DataFrame({"cell": cells, "label": ["A", "B"] * (len(cells) // 2) + ["A"] * (len(cells) % 2)})
    ds = ds.pp.drop_layers(Layers.LA_PROPERTIES, suppress_warnings=True)
    return ds.la.add_labels_from_dataframe(df, ignore_neighborhoods=True)


OPERATIONS = {
    "get_bbox": lambda ds: ds.pp[1600:1650, 2100:2150],
    "la_getitem": lambda ds: ds.la[[1, 2]],
    "nh_getitem": lambda ds: ds.nh[[1, 2]],
    "filter_by_obs": lambda ds: ds.pp.filter_by_obs(Features.X, lambda x: x > _median_x(ds)),
    "mask_cells": lambda ds: _mask_left_half(ds).pp.mask_cells(),
    "la_deselect": lambda ds: ds.la.deselect([1]),
    "nh_deselect": lambda ds: ds.nh.deselect([1]),
    "add_labels_from_dataframe": _partial_labels,
    "mask_cells_reindex": lambda ds: _mask_left_half(ds).pp.mask_cells(reindex=True),
    "filter_by_obs_reindex": lambda ds: ds.pp.filter_by_obs(Features.X, lambda x: x > _median_x(ds), reindex=True),
    "remove_outlying_cells_reindex": lambda ds: ds.pp.remove_outlying_cells(dilation_size=1, threshold=4, reindex=True),
}


def _params(known_failures):
    return [
        pytest.param(
            operation,
            id=name,
            marks=(
                [pytest.mark.xfail(strict=True, reason="cells and cells_2 are not kept in sync yet")]
                if name in known_failures
                else []
            ),
        )
        for name, operation in OPERATIONS.items()
    ]


@pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
def test_cells_and_cells_2_stay_in_sync(ds_neighborhoods, operation):
    assert Layers.ADJACENCY_MATRIX in ds_neighborhoods

    result = operation(ds_neighborhoods)

    # the operation should actually have removed cells, otherwise the test does not check anything
    assert result.sizes[Dims.CELLS] < ds_neighborhoods.sizes[Dims.CELLS]
    # the adjacency matrix needs to stay square
    assert result[Layers.ADJACENCY_MATRIX].shape == (result.sizes[Dims.CELLS], result.sizes[Dims.CELLS])
    # and row i and column i need to refer to the same cell
    np.testing.assert_array_equal(result.coords[Dims.CELLS].values, result.coords[Dims.CELLS_2].values)


@pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
def test_network_features_after_subsetting(ds_neighborhoods, operation):
    pytest.importorskip("networkx")

    result = operation(ds_neighborhoods)

    # building the graph from the adjacency matrix requires a square matrix
    result = result.nh.add_neighborhood_obs(features="degree")
    assert "degree" in result.coords[Dims.FEATURES].values
