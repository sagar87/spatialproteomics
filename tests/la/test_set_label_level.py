import pytest


def test_set_label_level(ds_labels):
    ds_labels.la.set_label_level("labels_1")


# TODO: put in the correct dataset here which only has predictions, no subtpyes
def test_set_label_level_no_levels(ds_dummy):
    with pytest.raises(
        AssertionError,
        match="No label levels found in the object. Please add label levels first, for example by using la.predict_cell_subtypes().",
    ):
        ds_dummy.la.set_label_level("labels_1")


def test_set_label_level_invalid(ds_labels):
    with pytest.raises(AssertionError, match="Level dummy not found in the label levels"):
        ds_labels.la.set_label_level("dummy")
