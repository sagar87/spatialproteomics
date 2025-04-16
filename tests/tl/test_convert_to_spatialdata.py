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
