import numpy as np
import pytest


def test_convert_to_8bit(ds_image):
    # this is already in 8 bit, so should be fine
    converted = ds_image.pp.convert_to_8bit(key_added="_converted_image")
    assert converted["_converted_image"].values.dtype == np.uint8


def test_convert_16_to_8bit(ds_image):
    img_size = (ds_image.sizes["x"], ds_image.sizes["y"])
    ds = ds_image.pp.add_layer(np.random.randint(0, 2**16, img_size).astype(np.uint16), "_16bit_image")
    ds = ds.pp.convert_to_8bit("_16bit_image", key_added="_converted_image")
    assert ds["_converted_image"].values.dtype == np.uint8


def test_convert_32_to_8bit(ds_image):
    img_size = (ds_image.sizes["x"], ds_image.sizes["y"])
    ds = ds_image.pp.add_layer(np.random.randint(0, 2**32, img_size).astype(np.uint32), "_32bit_image")
    ds = ds.pp.convert_to_8bit("_32bit_image", key_added="_converted_image")
    assert ds["_converted_image"].values.dtype == np.uint8


def test_convert_float_to_8bit(ds_image):
    img_size = (ds_image.sizes["x"], ds_image.sizes["y"])
    ds = ds_image.pp.add_layer(np.random.rand(*img_size).astype(np.float64), "_float_image")
    ds = ds.pp.convert_to_8bit("_float_image", key_added="_converted_image")
    assert ds["_converted_image"].values.dtype == np.uint8


def test_convert_float_with_values_larger_than_one(ds_image):
    img_size = (ds_image.sizes["x"], ds_image.sizes["y"])
    ds = ds_image.pp.add_layer(np.random.rand(*img_size) * 2, "_float_image")
    with pytest.raises(ValueError, match="The image is of type float, but the values are not in the range"):
        ds.pp.convert_to_8bit("_float_image", key_added="_converted_image")


def test_convert_float_with_values_smaller_than_zero(ds_image):
    img_size = (ds_image.sizes["x"], ds_image.sizes["y"])
    ds = ds_image.pp.add_layer(np.random.rand(*img_size) - 1, "_float_image")
    with pytest.raises(
        AssertionError, match="The image contains negative values. Please make sure that the image is non-negative."
    ):
        ds.pp.convert_to_8bit("_float_image", key_added="_converted_image")


def test_convert_to_8bit_key_does_not_exist(ds_image):
    with pytest.raises(AssertionError, match="The key non_existing_key does not exist in the object."):
        ds_image.pp.convert_to_8bit("non_existing_key", key_added="_converted_image")
