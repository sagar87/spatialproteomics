import numpy as np
import pytest

from spatialproteomics import Layers


def test_compute_neighborhoods_delaunay(ds_labels):
    dataset_neighborhoods = ds_labels.nh.compute_neighborhoods_delaunay()
    neighborhod_df = dataset_neighborhoods.pp.get_layer_as_df(Layers.NEIGHBORHOODS)
    # checking that each row sums to 1
    assert neighborhod_df.sum(axis=1).all() == pytest.approx(1.0)
    assert Layers.ADJACENCY_MATRIX in dataset_neighborhoods
    # checking that the adjacency matrix is symmetric
    assert (
        (dataset_neighborhoods[Layers.ADJACENCY_MATRIX] == dataset_neighborhoods[Layers.ADJACENCY_MATRIX].T).all().all()
    )


def test_compute_neighborhoods_delaunay_no_labels(ds_segmentation):
    with pytest.raises(AssertionError, match="No cell type labels found in the object."):
        ds_segmentation.nh.compute_neighborhoods_delaunay()


@pytest.mark.parametrize("include_center", [True, False])
def test_delaunay_adjacency_has_no_self_loops(ds_labels, include_center):
    ds = ds_labels.nh.compute_neighborhoods_delaunay(include_center=include_center)
    adj = ds[Layers.ADJACENCY_MATRIX].values
    assert np.all(np.diag(adj) == 0)
    assert np.all(adj.sum(axis=1) > 0)


def test_add_neighborhood_obs_ignores_stored_self_loops(ds_labels):
    ds = ds_labels.nh.compute_neighborhoods_delaunay()
    expected = ds.nh.add_neighborhood_obs(features=["degree", "homophily", "diversity_index"]).pp.get_layer_as_df()

    # adjacency matrices computed with older versions contained self-loops
    ds[Layers.ADJACENCY_MATRIX] = ds[Layers.ADJACENCY_MATRIX] + np.eye(ds.sizes["cells"], dtype=int)
    obs = ds.nh.add_neighborhood_obs(features=["degree", "homophily", "diversity_index"]).pp.get_layer_as_df()

    for feature in ["degree", "homophily", "diversity_index"]:
        np.testing.assert_allclose(obs[feature].values, expected[feature].values)
