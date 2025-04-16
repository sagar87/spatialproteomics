import copy as cp

import numpy as np
import pytest

import spatialproteomics as sp

basic_subtype_dict = {
    "T": {
        "subtypes": [
            {"name": "T_h", "markers": ["CD4+"]},
            {"name": "T_tox", "markers": ["CD8+"]},
        ]
    },
}


# helper functions
def get_ds_without_subtype_predictions(ds):
    ds = cp.deepcopy(ds)
    ct_inversion_dict = {"B": "B", "T_tox": "T", "T_h": "T", "T": "T"}
    # removing the subtype annotations
    ds.tables["table"].obs["celltype"] = [ct_inversion_dict[x] for x in ds.tables["table"].obs["_labels"]]
    ds.tables["table"].obsm["binarization"] = ds.tables["table"].obs[["CD4_binarized", "CD8_binarized"]]
    ds.tables["table"].obsm["binarization"].columns = ["CD4", "CD8"]
    return ds


# tests
def test_predict_cell_subtypes(ds_labels_spatialdata):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels_spatialdata)

    # testing the prediction of cell subtypes
    sp.la.predict_cell_subtypes(ds, basic_subtype_dict)
    assert "labels_1" in ds.tables["table"].obs.columns
    assert "T_tox" in ds.tables["table"].obs["labels_1"].values
    assert "T_h" in ds.tables["table"].obs["labels_1"].values
    assert "T" in ds.tables["table"].obs["labels_1"].values
    assert "B" in ds.tables["table"].obs["labels_1"].values
    assert "T_tox" not in ds.tables["table"].obs["labels_0"].values
    assert "T_h" not in ds.tables["table"].obs["labels_0"].values
    assert "T" in ds.tables["table"].obs["labels_0"].values
    assert "B" in ds.tables["table"].obs["labels_0"].values


def test_predict_cell_subtypes_no_labels(ds_segmentation_spatialdata):
    with pytest.raises(
        AssertionError,
        match="Tables key table not found in spatial data object.",
    ):
        sp.la.predict_cell_subtypes(ds_segmentation_spatialdata, basic_subtype_dict)


# === functional tests for different edge cases ===
def test_multilevel(ds_labels_spatialdata):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels_spatialdata)

    subtype_dict = {
        "T": {
            "subtypes": [
                {
                    "name": "T_h",
                    "markers": ["CD4+"],
                },
                {
                    "name": "T_tox",
                    "markers": ["CD4-"],
                    "subtypes": [
                        {"name": "T_tox_CD8_pos", "markers": ["CD8+"]},
                        {"name": "T_tox_CD8_neg", "markers": ["CD8-"]},
                    ],
                },
            ]
        },
    }

    sp.la.predict_cell_subtypes(ds, subtype_dict)
    df = ds.tables["table"].obs[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1", "labels_2"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_2"] == "T_tox_CD8_pos"]["CD8_binarized"] == 1)
    assert np.all(df[df["labels_2"] == "T_tox_CD8_neg"]["CD8_binarized"] == 0)


def test_multiple_markers(ds_labels_spatialdata):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels_spatialdata)

    subtype_dict = {
        "T": {"subtypes": [{"name": "T_h", "markers": ["CD4+"]}, {"name": "T_tox_naive", "markers": ["CD8+", "CD4-"]}]},
    }

    sp.la.predict_cell_subtypes(ds, subtype_dict)
    df = ds.tables["table"].obs[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_1"] == "T_tox_naive"]["CD8_binarized"] == 1)
    assert np.all(df[df["labels_1"] == "T_tox_naive"]["CD4_binarized"] == 0)


def test_alternative_markers(ds_labels_spatialdata):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels_spatialdata)

    subtype_dict = {
        "T": {"subtypes": [{"name": "T_tox", "markers": ["CD4+"]}, {"name": "T_tox", "markers": ["CD8+"]}]},
    }

    sp.la.predict_cell_subtypes(ds, subtype_dict)
    df = ds.tables["table"].obs[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()

    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert df[df["labels_1"] == "T_tox"]["CD8_binarized"].shape[0] == 3


def test_marker_negativity(ds_labels_spatialdata):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels_spatialdata)

    subtype_dict = {
        "T": {"subtypes": [{"name": "T_h", "markers": ["CD4+"]}, {"name": "T_tox", "markers": ["CD4-"]}]},
    }

    sp.la.predict_cell_subtypes(ds, subtype_dict)
    df = ds.tables["table"].obs[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_1"] == "T_tox"]["CD4_binarized"] == 0)


def test_negativity_and_positivity(ds_labels_spatialdata):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels_spatialdata)

    subtype_dict = {
        "T": {
            "subtypes": [
                {"name": "T_h", "markers": ["CD4+", "CD8-"]},
            ]
        },
    }

    sp.la.predict_cell_subtypes(ds, subtype_dict)
    df = ds.tables["table"].obs[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_1"] == "T_h"]["CD4_binarized"] == 1)
    assert np.all(df[df["labels_1"] == "T_h"]["CD8_binarized"] == 0)


def test_invalid_markers(ds_labels_spatialdata):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels_spatialdata)

    subtype_dict = {
        "T": {
            "subtypes": [
                {"name": "T_h", "markers": ["CD4+"]},
                {
                    "name": "T_tox",
                    "markers": ["CD8+"],
                    "subtypes": [
                        {"name": "T_bla", "markers": ["Blub+"]},
                    ],
                },
            ]
        },
    }

    sp.la.predict_cell_subtypes(ds, subtype_dict)
    df = ds.tables["table"].obs[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1", "labels_2"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert "T" in df["labels_2"].values
    assert "T_tox" in df["labels_2"].values
    assert "T_h" in df["labels_2"].values
    # these should not be in there, because those markers were not binarized
    assert "T_reg" not in df["labels_2"].values
    assert "T_bla" not in df["labels_2"].values
