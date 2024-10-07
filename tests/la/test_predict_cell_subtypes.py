import pytest

import spatialproteomics as sp

subtype_dict = {
    "Cell type 1": {"subtypes": [{"name": "nTreg", "markers": ["FOXP3"]}]},
    "Cell type 2": {"subtypes": [{"name": "T_h", "markers": ["CD4"]}, {"name": "T_tox", "markers": ["CD8"]}]},
}


def test_predict_cell_subtypes(dataset_labeled):
    # before we can do subsetting, we need to binarize some markers
    binarization_dict = {"CD4": 0.5, "CD8": 0.6, "FOXP3": 0.5, "BCL6": 0.7}
    ds = (
        dataset_labeled.pp[["CD4", "CD8", "FOXP3", "BCL6"]]
        .pp.threshold(quantile=[0.9, 0.9, 0.9, 0.9])
        .pp.add_quantification(func=sp.percentage_positive, key_added="_percentage_positive")
        .la.threshold_labels(binarization_dict, layer_key="_percentage_positive")
    )
    ds = ds.la.predict_cell_subtypes(subtype_dict)
    assert "labels_1" in ds.pp.get_layer_as_df().columns
    assert (
        ds.pp.get_layer_as_df("_la_properties").shape[0] > dataset_labeled.pp.get_layer_as_df("_la_properties").shape[0]
    )


# test wrong marker names
def test_predict_cell_subtypes_wrong_marker_names(dataset_labeled):
    # before we can do subsetting, we need to binarize some markers
    binarization_dict = {"CD4": 0.5, "CD8": 0.6, "FOXP3": 0.5, "BCL6": 0.7}
    ds = (
        dataset_labeled.pp[["CD4", "CD8", "FOXP3", "BCL6"]]
        .pp.threshold(quantile=[0.9, 0.9, 0.9, 0.9])
        .pp.add_quantification(func=sp.percentage_positive, key_added="_percentage_positive")
        .la.threshold_labels(binarization_dict, layer_key="_percentage_positive")
    )

    subtype_dict = {
        "Cell type 1": {"subtypes": [{"name": "nTreg", "markers": ["dummy_marker"]}]},
        "Cell type 2": {"subtypes": [{"name": "T_h", "markers": ["CD4"]}, {"name": "T_tox", "markers": ["CD8"]}]},
    }

    with pytest.raises(
        AssertionError,
        match="All markers must be binarized before predicting cell subtypes",
    ):
        ds.la.predict_cell_subtypes(subtype_dict)


# test no binarization present
def test_predict_cell_subtypes_no_binarization(dataset_labeled):
    with pytest.raises(
        AssertionError,
        match="All markers must be binarized before predicting cell subtypes",
    ):
        dataset_labeled.la.predict_cell_subtypes(subtype_dict)


def test_predict_cell_subtypes_no_labels(dataset):
    with pytest.raises(
        AssertionError,
        match="No cell type labels found in the object",
    ):
        dataset.la.predict_cell_subtypes(subtype_dict)
