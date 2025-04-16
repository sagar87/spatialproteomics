import pytest

from spatialproteomics import Layers


def test_compute_neighborhoods_radius(ds_labels):
    dataset_neighborhoods = ds_labels.nh.compute_neighborhoods_radius()
    neighborhod_df = dataset_neighborhoods.pp.get_layer_as_df(Layers.NEIGHBORHOODS)
    # checking that each row sums to 1
    assert neighborhod_df.sum(axis=1).all() == pytest.approx(1.0)
    assert Layers.ADJACENCY_MATRIX in dataset_neighborhoods
    # checking that the adjacency matrix is symmetric
    assert (
        (dataset_neighborhoods[Layers.ADJACENCY_MATRIX] == dataset_neighborhoods[Layers.ADJACENCY_MATRIX].T).all().all()
    )


def test_compute_neighborhoods_radius_no_labels(ds_segmentation):
    with pytest.raises(AssertionError, match="No cell type labels found in the object."):
        ds_segmentation.nh.compute_neighborhoods_radius()


def test_compute_neighborhoods_radius_negative_radius(ds_labels):
    with pytest.raises(AssertionError, match="Radius must be greater than 0."):
        ds_labels.nh.compute_neighborhoods_radius(radius=-1)
