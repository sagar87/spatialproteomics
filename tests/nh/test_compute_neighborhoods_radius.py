import numpy as np
import pandas as pd
import pytest

from spatialproteomics import Layers
from spatialproteomics.nh.utils import _construct_neighborhood_df_radius


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


@pytest.mark.parametrize("include_center", [True, False])
def test_radius_adjacency_has_no_self_loops(include_center):
    df = pd.DataFrame({"x": [0, 1, 2, 100], "y": [0] * 4, "labels": [1, 1, 2, 2]})
    profile, adj = _construct_neighborhood_df_radius(
        df, [1, 2], x="x", y="y", label_col="labels", radius=1.5, include_center=include_center
    )

    np.testing.assert_array_equal(adj, [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    # the center cell is only part of the neighborhood profile
    expected = [1 / 2, 1 / 2] if include_center else [1.0, 0.0]
    np.testing.assert_allclose(profile.iloc[2].values, expected)


def test_add_neighborhood_obs_without_self_loops(ds_labels):
    ds = ds_labels.nh.compute_neighborhoods_radius(include_center=True)
    adj = ds[Layers.ADJACENCY_MATRIX].values
    assert np.all(np.diag(adj) == 0)

    ds = ds.nh.add_neighborhood_obs(features=["degree", "homophily"])
    obs = ds.pp.get_layer_as_df()
    np.testing.assert_array_equal(obs["degree"].values, adj.sum(axis=1))
    assert obs["homophily"].between(0, 1).all()
