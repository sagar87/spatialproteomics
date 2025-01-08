import pytest


def test_convert_to_anndata(dataset_binarized):
    dataset_binarized.tl.convert_to_anndata()


def test_convert_to_anndata_no_quantification(dataset):
    # this dataset has no intensities, and should hence raise an error
    with pytest.raises(
        AssertionError,
        match="Expression matrix key _intensity not found in the object.",
    ):
        dataset.tl.convert_to_anndata()
