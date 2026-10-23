import numpy as np
import pandas as pd
import pytest

from spatialproteomics import Layers
from spatialproteomics.nh.utils import _construct_neighborhood_df_knn


def test_compute_neighborhoods_knn(ds_labels):
    dataset_neighborhoods = ds_labels.nh.compute_neighborhoods_knn()
    neighborhod_df = dataset_neighborhoods.pp.get_layer_as_df(Layers.NEIGHBORHOODS)
    # checking that each row sums to 1
    assert neighborhod_df.sum(axis=1).all() == pytest.approx(1.0)
    assert Layers.ADJACENCY_MATRIX in dataset_neighborhoods
    # checking that the adjacency matrix is symmetric
    assert (
        (dataset_neighborhoods[Layers.ADJACENCY_MATRIX] == dataset_neighborhoods[Layers.ADJACENCY_MATRIX].T).all().all()
    )


def test_compute_neighborhoods_knn_no_labels(ds_segmentation):
    with pytest.raises(AssertionError, match="No cell type labels found in the object."):
        ds_segmentation.nh.compute_neighborhoods_knn()


def test_compute_neighborhoods_knn_negative_k(ds_labels):
    with pytest.raises(AssertionError, match="K must be greater than 0."):
        ds_labels.nh.compute_neighborhoods_knn(k=-1)


def test_knn_adjacency_is_local():
    df = pd.DataFrame({"x": [0, 1, 2, 100, 101, 102], "y": [0] * 6, "labels": [1] * 6})
    _, adj = _construct_neighborhood_df_knn(df, [1], x="x", y="y", label_col="labels", k=2)
    expected = np.kron(np.eye(2, dtype=int), np.ones((3, 3), dtype=int))
    np.testing.assert_array_equal(adj, expected)

    _, adj = _construct_neighborhood_df_knn(df, [1], x="x", y="y", label_col="labels", k=2, include_center=False)
    np.testing.assert_array_equal(adj, expected - np.eye(6, dtype=int))
