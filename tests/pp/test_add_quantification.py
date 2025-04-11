import numpy as np
import pandas as pd
import pytest

from spatialproteomics.constants import Layers


def test_add_quantification(ds_segmentation):
    quantified = ds_segmentation.pp.add_quantification()

    # check that a quantification is added and the existing layers are retained
    assert Layers.IMAGE in quantified
    assert Layers.SEGMENTATION in quantified
    assert Layers.INTENSITY in quantified


def test_add_quantification_key_exists(ds_segmentation):
    # check that it doesn't work if the key already exists
    with pytest.raises(
        AssertionError,
        match="Found _intensity in image container. Please add a different key or remove the previous quantification.",
    ):
        ds_segmentation.pp.add_quantification().pp.add_quantification()


def test_add_quantification_spurious_function(ds_segmentation):
    # check that it doesn't work if the input function does not output the right format
    with pytest.raises(ValueError, match="conflicting sizes for dimension"):
        ds_segmentation.pp.add_quantification(func=lambda x: np.zeros([5, 10]))


def test_add_quantification_spurious_function_2(ds_segmentation):
    # check that it doesn't work if the function is neither a function nor a string
    with pytest.raises(
        ValueError,
        match="The func parameter should be either a string for default skimage properties or a callable function.",
    ):
        ds_segmentation.pp.add_quantification(func=3)


def test_add_quantification_spurious_regionprop(ds_segmentation):
    # check that it doesn't work if the function is a string but not a valid regionprop
    with pytest.raises(
        AttributeError,
        match="Invalid regionprop",
    ):
        ds_segmentation.pp.add_quantification(func="dummy_str")


def test_add_quantification_from_dataframe(ds_segmentation):
    quantification_df = pd.DataFrame(
        np.zeros([ds_segmentation.sizes["cells"], ds_segmentation.sizes["channels"]]),
        columns=ds_segmentation.coords["channels"],
    )
    quantification_df.index = ds_segmentation.coords["cells"]
    quantified = ds_segmentation.pp.add_quantification_from_dataframe(quantification_df)

    # check that a quantification is added and the existing layers are retained
    assert Layers.IMAGE in quantified
    assert Layers.SEGMENTATION in quantified
    assert Layers.INTENSITY in quantified


def test_add_quantification_from_dataframe_mismatched_index(ds_segmentation):
    quantification_df = pd.DataFrame(
        np.zeros([ds_segmentation.sizes["cells"], ds_segmentation.sizes["channels"]]),
        columns=ds_segmentation.coords["channels"],
    )
    with pytest.raises(AssertionError, match="Cells in the image container are not in the dataframe."):
        ds_segmentation.pp.add_quantification_from_dataframe(quantification_df)


def test_add_quantification_from_dataframe_mismatched_columns(ds_segmentation):
    quantification_df = pd.DataFrame(np.zeros([ds_segmentation.sizes["cells"], ds_segmentation.sizes["channels"]]))
    quantification_df.index = ds_segmentation.coords["cells"]
    with pytest.raises(AssertionError, match="Channels in the image container are not in the dataframe."):
        ds_segmentation.pp.add_quantification_from_dataframe(quantification_df)


def test_add_quantification_from_dataframe_wrong_dtype(ds_segmentation):
    quantification_array = np.zeros([ds_segmentation.sizes["cells"], ds_segmentation.sizes["channels"]])
    with pytest.raises(TypeError, match="The input must be a pandas DataFrame."):
        ds_segmentation.pp.add_quantification_from_dataframe(quantification_array)
