import numpy as np


def test_apply(ds_image):
    def set_to_zero(arr):
        return arr * 0

    processed = ds_image.pp.apply(set_to_zero, key_added="_processed_image")
    assert np.all(processed["_processed_image"].values == 0)
