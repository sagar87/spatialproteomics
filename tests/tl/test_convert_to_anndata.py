import numpy as np
import pytest
import xarray as xr

from spatialproteomics.constants import Dims


def test_convert_to_anndata_image(ds_image):
    adata = ds_image.tl.convert_to_anndata()
    assert adata.X is None


def test_convert_to_anndata_segmentation(ds_segmentation):
    adata = ds_segmentation.tl.convert_to_anndata()
    assert adata.X is None


def test_convert_to_anndata_labels(ds_labels):
    adata = ds_labels.tl.convert_to_anndata()
    assert adata.X.shape == (56, 5)
    assert "CD4_binarized" in adata.obs.columns
    assert "CD8_binarized" in adata.obs.columns
    assert "_labels" in adata.obs.columns
    assert "centroid-0" in adata.obs.columns
    assert "centroid-1" in adata.obs.columns
    assert "_labels_colors" in adata.uns.keys()
    assert "spatial" in adata.obsm.keys()


def test_convert_to_anndata_neighborhoods(ds_neighborhoods):
    adata = ds_neighborhoods.tl.convert_to_anndata()
    assert adata.X.shape == (56, 5)
    assert "_neighborhoods" in adata.obs.columns
    assert "CD4_binarized" in adata.obs.columns
    assert "CD8_binarized" in adata.obs.columns
    assert "_labels" in adata.obs.columns
    assert "centroid-0" in adata.obs.columns
    assert "centroid-1" in adata.obs.columns
    assert "_labels_colors" in adata.uns.keys()
    assert "spatial" in adata.obsm.keys()


def test_convert_to_anndata_additional_obsm(ds_labels):
    # make a copy to avoid carrying modifications of the original into other tests where they cause issues
    ds = ds_labels.copy()

    # create and add new embedding layer
    ds["_embeddings"] = xr.DataArray(
        np.ones((ds.sizes[Dims.CELLS], 3)),
        dims=(Dims.CELLS, "embedding_dim"),
        coords={
            Dims.CELLS: ds.coords[Dims.CELLS],
            "embedding_dim": ["embedding_1", "embedding_2", "embedding_3"],
        },
        name="_embeddings",
    )

    # check that trying to pass as layer errors
    with pytest.raises(AssertionError, match="AnnData layers must match"):
        ds.tl.convert_to_anndata(additional_layers={"embeddings": "_embeddings"})

    # check that passing as obsm behaves as expected
    adata = ds.tl.convert_to_anndata(additional_obsm={"embeddings": "_embeddings"})
    assert "embeddings" in adata.obsm.keys()
    assert adata.obsm["embeddings"].shape == (ds.sizes[Dims.CELLS], 3)
    np.testing.assert_array_equal(adata.obsm["embeddings"], ds["_embeddings"].values)


def test_convert_to_anndata_additional_obsm_wrong_shape(ds_labels):
    ds = ds_labels.copy()

    # check that trying to pass obsm with wrong number of observations errors
    ds["_marker_embeddings"] = xr.DataArray(
        np.ones((ds.sizes[Dims.CHANNELS], 2)),
        dims=(Dims.CHANNELS, "marker_embedding_dim"),
        coords={
            Dims.CHANNELS: ds.coords[Dims.CHANNELS],
            "marker_embedding_dim": ["marker_embedding_1", "marker_embedding_2"],
        },
        name="_marker_embeddings",
    )

    with pytest.raises(AssertionError, match="AnnData obsm entries must match"):
        ds.tl.convert_to_anndata(additional_obsm={"marker_embeddings": "_marker_embeddings"})
