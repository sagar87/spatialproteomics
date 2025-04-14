import numpy as np
import pytest

import spatialproteomics as sp
from spatialproteomics.constants import SDLayers


def test_add_quantification(ds_segmentation_spatialdata):
    ds = sp.pp.add_quantification(ds_segmentation_spatialdata, copy=True)

    # check that a quantification is added to the copy, but not the object itself
    assert SDLayers.TABLE in ds.tables.keys()
    assert SDLayers.TABLE not in ds_segmentation_spatialdata.tables.keys()


def test_add_quantification_key_exists(ds_segmentation_spatialdata):
    # check that it doesn't work if the key already exists
    with pytest.raises(
        AssertionError,
        match="An expression matrix already exists in your spatialdata object. Please provide a layer_key to add the new quantification to.",
    ):
        ds = sp.pp.add_quantification(ds_segmentation_spatialdata, copy=True)
        sp.pp.add_quantification(ds)


def test_add_quantification_spurious_function(ds_segmentation_spatialdata):
    # check that it doesn't work if the input function does not output the right format
    with pytest.raises(ValueError, match="Length of passed value for var_names is 5"):
        sp.pp.add_quantification(ds_segmentation_spatialdata, func=lambda x: np.zeros([5, 10]), copy=True)


def test_add_quantification_spurious_function_2(ds_segmentation_spatialdata):
    # check that it doesn't work if the function is neither a function nor a string
    with pytest.raises(
        ValueError,
        match="The func parameter should be either a string for default skimage properties or a callable function.",
    ):
        sp.pp.add_quantification(ds_segmentation_spatialdata, func=3, copy=True)


def test_add_quantification_spurious_regionprop(ds_segmentation_spatialdata):
    # check that it doesn't work if the function is a string but not a valid regionprop
    with pytest.raises(
        AttributeError,
        match="Invalid regionprop",
    ):
        sp.pp.add_quantification(ds_segmentation_spatialdata, func="dummy_str")
