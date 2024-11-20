import pytest


def test_get_layer_as_df(dataset_labeled):
    df = dataset_labeled.pp.get_layer_as_df()

    # checking that the cell types are strings
    assert df["_labels"].dtype == "object"


def test_get_layer_as_df_nonexistent_layer(dataset_labeled):
    with pytest.raises(AssertionError, match="Layer _dummy_layer not found in the object."):
        dataset_labeled.pp.get_layer_as_df(layer="_dummy_layer")


def test_get_layer_as_df_celltypes_as_int(dataset_labeled):
    df = dataset_labeled.pp.get_layer_as_df(celltypes_to_str=False)

    # checking that the cell types are floats
    assert df["_labels"].dtype == "float64"
