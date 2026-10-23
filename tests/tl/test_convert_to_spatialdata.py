import numpy as np
import pytest
import spatialdata as sd

import spatialproteomics as sp


def test_convert_to_spatialdata_image(ds_image):
    sdata = ds_image.tl.convert_to_spatialdata()
    assert "image" in sdata.images.keys()
    assert "table" not in sdata.tables.keys()


def test_convert_to_spatialdata_segmentation(ds_segmentation):
    sdata = ds_segmentation.tl.convert_to_spatialdata()
    assert "image" in sdata.images.keys()
    assert "segmentation" in sdata.labels.keys()
    assert "table" not in sdata.tables.keys()


def test_convert_to_spatialdata_labels(ds_labels):
    sdata = ds_labels.tl.convert_to_spatialdata()
    adata = sdata.tables["table"]
    assert "image" in sdata.images.keys()
    assert "segmentation" in sdata.labels.keys()
    assert adata.X.shape == (56, 5)
    assert "CD4_binarized" in adata.obs.columns
    assert "CD8_binarized" in adata.obs.columns
    assert "_labels" in adata.obs.columns
    assert "centroid-0" in adata.obs.columns
    assert "centroid-1" in adata.obs.columns
    assert "_labels_colors" in adata.uns.keys()
    assert "spatial" in adata.obsm.keys()


def test_convert_to_spatialdata_neighborhoods(ds_neighborhoods):
    sdata = ds_neighborhoods.tl.convert_to_spatialdata()
    adata = sdata.tables["table"]
    assert "image" in sdata.images.keys()
    assert "segmentation" in sdata.labels.keys()
    assert adata.X.shape == (56, 5)
    assert "_neighborhoods" in adata.obs.columns
    assert "CD4_binarized" in adata.obs.columns
    assert "CD8_binarized" in adata.obs.columns
    assert "_labels" in adata.obs.columns
    assert "centroid-0" in adata.obs.columns
    assert "centroid-1" in adata.obs.columns
    assert "_labels_colors" in adata.uns.keys()
    assert "spatial" in adata.obsm.keys()


@pytest.fixture
def ds_non_square():
    # non-square and offset, so that swapped axes or a lost transform cannot go unnoticed
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(2, 30, 50), dtype=np.uint8)
    segmentation = np.zeros((30, 50), dtype=np.int64)
    segmentation[2:10, 5:20] = 1
    segmentation[15:28, 30:45] = 2
    return sp.load_image_data(
        image,
        channel_coords=["CD4", "CD8"],
        x_coords=np.arange(50) + 6400,
        y_coords=np.arange(30) + 22400,
        segmentation=segmentation,
    )


def _assert_same_object(ds, ds_back):
    np.testing.assert_array_equal(ds_back["_image"].values, ds["_image"].values)
    np.testing.assert_array_equal(ds_back["_segmentation"].values, ds["_segmentation"].values)
    np.testing.assert_allclose(ds_back.coords["x"].values, ds.coords["x"].values)
    np.testing.assert_allclose(ds_back.coords["y"].values, ds.coords["y"].values)


def test_convert_to_spatialdata_keeps_axis_order(ds_non_square):
    sdata = ds_non_square.tl.convert_to_spatialdata()
    np.testing.assert_array_equal(sdata.images["image"].values, ds_non_square["_image"].values)
    np.testing.assert_array_equal(sdata.labels["segmentation"].values, ds_non_square["_segmentation"].values)


def test_spatialdata_roundtrip_in_memory(ds_non_square):
    sdata = ds_non_square.tl.convert_to_spatialdata()
    _assert_same_object(ds_non_square, sp.read_from_spatialdata(sdata))


def test_spatialdata_roundtrip_zarr(ds_non_square, tmp_path):
    path = tmp_path / "sdata.zarr"
    ds_non_square.tl.convert_to_spatialdata().write(path)
    _assert_same_object(ds_non_square, sp.read_from_spatialdata(sd.read_zarr(path)))
