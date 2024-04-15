import numpy as np
import pytest

from spatial_data.constants import Layers


def test_add_quantification(dataset):
    quantified = dataset.pp.add_quantification()

    # check that a quantification is added and the existing layers are retained
    assert Layers.IMAGE in quantified
    assert Layers.SEGMENTATION in quantified
    assert Layers.INTENSITY in quantified


def test_add_quantification_key_exists(dataset):
    # check that it doesn't work if the key already exists
    with pytest.raises(AssertionError, match="Found _intensity in image container. Please add a different key."):
        dataset.pp.add_quantification().pp.add_quantification()


def test_add_quantification_spurious_function(dataset):
    # check that it doesn't work if the input function does not output the right format
    with pytest.raises(ValueError, match="conflicting sizes for dimension"):
        dataset.pp.add_quantification(func=lambda x: np.zeros([5, 10]))
