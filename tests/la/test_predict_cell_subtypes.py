import numpy as np
import pandas as pd
import pytest

from spatialproteomics.constants import Layers

basic_subtype_dict = {
    "T": {
        "subtypes": [
            {"name": "T_h", "markers": ["CD4+"]},
            {"name": "T_tox", "markers": ["CD8+"]},
        ]
    },
}


# helper functions
def get_labels_and_binarization(ds):
    df1 = ds.pp.get_layer_as_df()
    df2 = ds.pp.get_layer_as_df("_la_layers")
    df = pd.concat((df1, df2), axis=1)
    return df


def get_ds_without_subtype_predictions(ds):
    # removing the subtype annotations
    cts = ds.pp.get_layer_as_df("_la_layers").reset_index()
    return ds.pp.drop_layers(["_la_properties", "_la_layers"]).la.add_labels_from_dataframe(
        cts, label_col="labels_0", cell_col="index"
    )


# tests
def test_predict_cell_subtypes(ds_labels):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels)

    # testing the prediction of cell subtypes
    ds_subtypes = ds.la.predict_cell_subtypes(basic_subtype_dict)
    assert "labels_1" in ds_subtypes.pp.get_layer_as_df(Layers.LA_LAYERS).columns
    assert (
        ds_subtypes.pp.get_layer_as_df(Layers.LA_PROPERTIES).shape[0]
        > ds.pp.get_layer_as_df(Layers.LA_PROPERTIES).shape[0]
    )


def test_predict_cell_subtypes_no_labels(ds_segmentation):
    with pytest.raises(
        AssertionError,
        match="No cell type labels found in the object",
    ):
        ds_segmentation.la.predict_cell_subtypes(basic_subtype_dict)


# === functional tests for different edge cases ===
def test_multilevel(ds_labels):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels)

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

    ds = ds.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1", "labels_2"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_2"] == "T_tox_CD8_pos"]["CD8_binarized"] == 1)
    assert np.all(df[df["labels_2"] == "T_tox_CD8_neg"]["CD8_binarized"] == 0)


def test_multiple_markers(ds_labels):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels)

    subtype_dict = {
        "T": {"subtypes": [{"name": "T_h", "markers": ["CD4+"]}, {"name": "T_tox_naive", "markers": ["CD8+", "CD4-"]}]},
    }

    ds = ds.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_1"] == "T_tox_naive"]["CD8_binarized"] == 1)
    assert np.all(df[df["labels_1"] == "T_tox_naive"]["CD4_binarized"] == 0)


def test_alternative_markers(ds_labels):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels)

    subtype_dict = {
        "T": {"subtypes": [{"name": "T_tox", "markers": ["CD4+"]}, {"name": "T_tox", "markers": ["CD8+"]}]},
    }

    ds = ds.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()

    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert df[df["labels_1"] == "T_tox"]["CD8_binarized"].shape[0] == 3


def test_marker_negativity(ds_labels):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels)

    subtype_dict = {
        "T": {"subtypes": [{"name": "T_h", "markers": ["CD4+"]}, {"name": "T_tox", "markers": ["CD4-"]}]},
    }

    ds = ds.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_1"] == "T_tox"]["CD4_binarized"] == 0)


def test_negativity_and_positivity(ds_labels):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels)

    subtype_dict = {
        "T": {
            "subtypes": [
                {"name": "T_h", "markers": ["CD4+", "CD8-"]},
            ]
        },
    }

    ds = ds.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_1"] == "T_h"]["CD4_binarized"] == 1)
    assert np.all(df[df["labels_1"] == "T_h"]["CD8_binarized"] == 0)


def test_invalid_markers(ds_labels):
    # removing the subtype annotations
    ds = get_ds_without_subtype_predictions(ds_labels)

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

    ds = ds.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1", "labels_2"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert "T" in df["labels_2"].values
    assert "T_tox" in df["labels_2"].values
    assert "T_h" in df["labels_2"].values
    # these should not be in there, because those markers were not binarized
    assert "T_reg" not in df["labels_2"].values
    assert "T_bla" not in df["labels_2"].values
