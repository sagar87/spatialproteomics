import pytest

from spatialproteomics import Layers


def test_compute_neighborhoods_knn(dataset_labeled):
    dataset_neighborhoods = dataset_labeled.nh.compute_neighborhoods_knn()
    neighborhod_df = dataset_neighborhoods.pp.get_layer_as_df(Layers.NEIGHBORHOODS)
    # checking that each row sums to 1
    assert neighborhod_df.sum(axis=1).all() == pytest.approx(1.0)
    assert Layers.ADJACENCY_MATRIX in dataset_neighborhoods


def test_compute_neighborhoods_knn_no_labels(dataset):
    with pytest.raises(AssertionError, match="No cell type labels found in the object."):
        dataset.nh.compute_neighborhoods_knn()


def test_compute_neighborhoods_knn_negative_k(dataset):
    with pytest.raises(AssertionError, match="K must be greater than 0."):
        dataset.nh.compute_neighborhoods_knn(k=-1)
