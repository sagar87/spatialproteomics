import spatialproteomics as sp
from spatialproteomics.constants import SDLayers


def test_add_obs_centroids(ds_labels_spatialdata):
    ds = sp.pp.add_observations(ds_labels_spatialdata, copy=True)

    assert SDLayers.TABLE in ds.tables.keys()
    adata = ds.tables[SDLayers.TABLE]
    assert adata.obs is not None
    assert "centroid-0" in adata.obs.columns
    assert "centroid-1" in adata.obs.columns


def test_add_obs_append_table(ds_labels_spatialdata):
    ds = sp.pp.add_observations(ds_labels_spatialdata, "area", copy=True)

    assert SDLayers.TABLE in ds.tables.keys()
    adata = ds.tables[SDLayers.TABLE]
    assert adata.obs is not None
    assert "centroid-0" not in adata.obs.columns
    assert "centroid-1" not in adata.obs.columns
    assert "area" in adata.obs.columns

    sp.pp.add_observations(ds, "eccentricity")

    assert SDLayers.TABLE in ds.tables.keys()
    adata = ds.tables[SDLayers.TABLE]
    assert adata.obs is not None
    assert "centroid-0" not in adata.obs.columns
    assert "centroid-1" not in adata.obs.columns
    assert "area" in adata.obs.columns
    assert "eccentricity" in adata.obs.columns
