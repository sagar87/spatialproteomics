import numpy as np
import pandas as pd
import pytest

from spatial_proteomics.constants import Layers


def test_add_quantification(dataset):
    quantified = dataset.pp.add_quantification()

    # check that a quantification is added and the existing layers are retained
    assert Layers.IMAGE in quantified
    assert Layers.SEGMENTATION in quantified
    assert Layers.INTENSITY in quantified


def test_add_quantification_key_exists(dataset):
    # check that it doesn't work if the key already exists
    with pytest.raises(
        AssertionError,
        match="Found _intensity in image container. Please add a different key or remove the previous quantification.",
    ):
        dataset.pp.add_quantification().pp.add_quantification()


def test_add_quantification_spurious_function(dataset):
    # check that it doesn't work if the input function does not output the right format
    with pytest.raises(ValueError, match="conflicting sizes for dimension"):
        dataset.pp.add_quantification(func=lambda x: np.zeros([5, 10]))


def test_add_quantification_from_dataframe(dataset):
    quantification_df = pd.DataFrame(
        np.zeros([dataset.dims["cells"], dataset.dims["channels"]]), columns=dataset.coords["channels"]
    )
    quantification_df.index = dataset.coords["cells"]
    quantified = dataset.pp.add_quantification_from_dataframe(quantification_df)

    # check that a quantification is added and the existing layers are retained
    assert Layers.IMAGE in quantified
    assert Layers.SEGMENTATION in quantified
    assert Layers.INTENSITY in quantified


def test_add_quantification_from_dataframe_mismatched_index(dataset):
    quantification_df = pd.DataFrame(
        np.zeros([dataset.dims["cells"], dataset.dims["channels"]]), columns=dataset.coords["channels"]
    )
    with pytest.raises(AssertionError, match="Cells in the image container are not in the dataframe."):
        dataset.pp.add_quantification_from_dataframe(quantification_df)


def test_add_quantification_from_dataframe_mismatched_columns(dataset):
    quantification_df = pd.DataFrame(np.zeros([dataset.dims["cells"], dataset.dims["channels"]]))
    quantification_df.index = dataset.coords["cells"]
    with pytest.raises(AssertionError, match="Channels in the image container are not in the dataframe."):
        dataset.pp.add_quantification_from_dataframe(quantification_df)


def test_add_quantification_from_dataframe_wrong_dtype(dataset):
    quantification_array = np.zeros([dataset.dims["cells"], dataset.dims["channels"]])
    with pytest.raises(TypeError, match="The input must be a pandas DataFrame."):
        dataset.pp.add_quantification_from_dataframe(quantification_array)
