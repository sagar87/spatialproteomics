import numpy as np

import spatialproteomics as sp


def test_apply(ds_image_spatialdata):
    def set_to_zero(arr):
        return arr * 0

    ds = sp.pp.apply(ds_image_spatialdata, set_to_zero, key_added="processed_image", copy=True)
    assert "image" in ds.images.keys()
    assert "processed_image" in ds.images.keys()
    assert ds.images["processed_image"].shape == ds.images["image"].shape
    assert np.all(ds.images["processed_image"].values == 0)
    assert not np.all(ds.images["image"].values == 0)
