import numpy as np
import pandas as pd
import pytest

import spatialproteomics as sp
from spatialproteomics.constants import Layers

basic_subtype_dict = {
    "Cell type 1": {"subtypes": [{"name": "Treg", "markers": ["FOXP3+"]}]},
    "Cell type 2": {"subtypes": [{"name": "T_h", "markers": ["CD4+"]}, {"name": "T_tox", "markers": ["CD8+"]}]},
}


def get_labels_and_binarization(ds):
    df1 = ds.pp.get_layer_as_df()
    df2 = ds.pp.get_layer_as_df("_la_layers")
    df = pd.concat((df1, df2), axis=1)
    return df


def test_predict_cell_subtypes(dataset_labeled):
    # before we can do subsetting, we need to binarize some markers
    binarization_dict = {"CD4": 0.5, "CD8": 0.6, "FOXP3": 0.5, "BCL6": 0.7}
    ds = (
        dataset_labeled.pp[["CD4", "CD8", "FOXP3", "BCL6"]]
        .pp.threshold(quantile=[0.9, 0.9, 0.9, 0.9])
        .pp.add_quantification(func=sp.percentage_positive, key_added="_percentage_positive")
        .la.threshold_labels(binarization_dict, layer_key="_percentage_positive")
    )
    ds = ds.la.predict_cell_subtypes(basic_subtype_dict)
    assert "labels_1" in ds.pp.get_layer_as_df(Layers.LA_LAYERS).columns
    assert (
        ds.pp.get_layer_as_df(Layers.LA_PROPERTIES).shape[0]
        > dataset_labeled.pp.get_layer_as_df(Layers.LA_PROPERTIES).shape[0]
    )


def test_predict_cell_subtypes_no_labels(dataset):
    with pytest.raises(
        AssertionError,
        match="No cell type labels found in the object",
    ):
        dataset.la.predict_cell_subtypes(basic_subtype_dict)


# === functional tests for different edge cases ===
def test_multilevel(dataset_binarized):
    subtype_dict = {
        "B": {"subtypes": [{"name": "B_prol", "markers": ["ki-67+"]}]},
        "T": {
            "subtypes": [
                {
                    "name": "T_h",
                    "markers": ["CD4+"],
                },
                {
                    "name": "T_tox",
                    "markers": ["CD8+"],
                    "subtypes": [
                        {"name": "T_tox_mem", "markers": ["CD45RO+"]},
                        {"name": "T_tox_naive", "markers": ["CD45RA+"]},
                    ],
                },
            ]
        },
    }

    ds = dataset_binarized.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[
        ["CD4_binarized", "CD8_binarized", "CD45RA_binarized", "CD45RO_binarized", "labels_0", "labels_1", "labels_2"]
    ]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 14
    assert np.all(df[df["labels_2"] == "T_tox_naive"]["CD8_binarized"] == 1)
    assert np.all(df[df["labels_2"] == "T_tox_naive"]["CD45RA_binarized"] == 1)


def test_multiple_markers(dataset_binarized):
    subtype_dict = {
        "B": {"subtypes": [{"name": "B_prol", "markers": ["ki-67+"]}]},
        "T": {
            "subtypes": [{"name": "T_h", "markers": ["CD4+"]}, {"name": "T_tox_naive", "markers": ["CD8+", "CD45RA+"]}]
        },
    }

    ds = dataset_binarized.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "CD45RA_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 8
    assert np.all(df[df["labels_1"] == "T_tox_naive"]["CD8_binarized"] == 1)
    assert np.all(df[df["labels_1"] == "T_tox_naive"]["CD45RA_binarized"] == 1)


def test_alternative_markers(dataset_binarized):
    subtype_dict = {
        "B": {"subtypes": [{"name": "B_prol", "markers": ["ki-67+"]}]},
        "T": {"subtypes": [{"name": "T_tox", "markers": ["CD45RO+"]}, {"name": "T_tox", "markers": ["CD45RA+"]}]},
    }

    ds = dataset_binarized.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD45RA_binarized", "CD45RO_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()

    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert df[df["labels_1"] == "T_tox"]["CD45RA_binarized"].shape[0] == 3


def test_marker_negativity(dataset_binarized):
    subtype_dict = {
        "B": {"subtypes": [{"name": "B_prol", "markers": ["ki-67+"]}]},
        "T": {"subtypes": [{"name": "T_h", "markers": ["CD4+"]}, {"name": "T_tox", "markers": ["CD4-"]}]},
    }

    ds = dataset_binarized.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_1"] == "T_tox"]["CD4_binarized"] == 0)


def test_negativity_and_positivity(dataset_binarized):
    subtype_dict = {
        "B": {"subtypes": [{"name": "B_prol", "markers": ["ki-67+"]}]},
        "T": {
            "subtypes": [
                {"name": "T_h", "markers": ["CD4+", "CD8-"]},
            ]
        },
    }

    ds = dataset_binarized.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[["CD4_binarized", "CD8_binarized", "labels_0", "labels_1"]]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 4
    assert np.all(df[df["labels_1"] == "T_h"]["CD4_binarized"] == 1)
    assert np.all(df[df["labels_1"] == "T_h"]["CD8_binarized"] == 0)


def test_invalid_markers(dataset_binarized):
    subtype_dict = {
        "B": {"subtypes": [{"name": "B_prol", "markers": ["ki-67+"]}]},
        "T": {
            "subtypes": [
                {"name": "T_h", "markers": ["CD4+"], "subtypes": [{"name": "T_reg", "markers": ["Foxp3+"]}]},
                {
                    "name": "T_tox",
                    "markers": ["CD8+"],
                    "subtypes": [
                        {"name": "T_tox_mem", "markers": ["CD45RO+"]},
                        {"name": "T_tox_naive", "markers": ["CD45RA+"]},
                        {"name": "T_bla", "markers": ["Blub+"]},
                    ],
                },
            ]
        },
    }

    ds = dataset_binarized.la.predict_cell_subtypes(subtype_dict)
    df = get_labels_and_binarization(ds)[
        ["CD4_binarized", "CD8_binarized", "CD45RA_binarized", "CD45RO_binarized", "labels_0", "labels_1", "labels_2"]
    ]
    df = df[df["labels_0"] == "T"].drop_duplicates()
    # check if the subtypes are correctly assigned
    assert df.shape[0] == 14
    assert "T" in df["labels_2"].values
    assert "T_tox" in df["labels_2"].values
    assert "T_tox_mem" in df["labels_2"].values
    assert "T_tox_naive" in df["labels_2"].values
    assert "T_h" in df["labels_2"].values
    # these should not be in there, because those markers were not binarized
    assert "T_reg" not in df["labels_2"].values
    assert "T_bla" not in df["labels_2"].values
