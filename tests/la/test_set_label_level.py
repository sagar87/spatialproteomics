import pytest


def test_set_label_level(ds_labels):
    ds_labels.la.set_label_level("labels_0")

    # this is actually the level that is already active, should still work though
    ds_labels.la.set_label_level("labels_1")


def test_set_label_level_no_levels(ds_labels):
    # removing the subtype annotations
    cts = ds_labels.pp.get_layer_as_df("_la_layers").reset_index()
    ds = ds_labels.pp.drop_layers(["_la_properties", "_la_layers"]).la.add_labels_from_dataframe(
        cts, label_col="labels_0", cell_col="index"
    )

    with pytest.raises(
        AssertionError,
        match="No label levels found in the object. Please add label levels first, for example by using la.predict_cell_subtypes().",
    ):
        ds.la.set_label_level("labels_1")


def test_set_label_level_invalid(ds_labels):
    with pytest.raises(AssertionError, match="Level dummy not found in the label levels"):
        ds_labels.la.set_label_level("dummy")
