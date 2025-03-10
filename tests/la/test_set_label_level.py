import pytest


def test_set_label_level(dataset_labeled_multilevel):
    dataset_labeled_multilevel.la.set_label_level("labels_1")


def test_set_label_level_no_levels(dataset_labeled):
    with pytest.raises(
        AssertionError,
        match="No label levels found in the object. Please add label levels first, for example by using la.predict_cell_subtypes().",
    ):
        dataset_labeled.la.set_label_level("labels_1")


def test_set_label_level_invalid(dataset_labeled_multilevel):
    with pytest.raises(AssertionError, match="Level dummy not found in the label levels"):
        dataset_labeled_multilevel.la.set_label_level("dummy")
