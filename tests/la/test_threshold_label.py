import numpy as np
import pytest


# === SPATIALPROTEOMICS BACKEND ===
def test_layer_does_not_exist(ds_labels):
    ds = ds_labels.pp.drop_layers("_percentage_positive")
    with pytest.raises(KeyError, match="Please add it first using pp.add_quantification()."):
        ds.la._threshold_label(channel="CD4", threshold=10.0, layer_key="_percentage_positive")


def test_channel_does_not_exist(ds_labels):
    channel = "NPC"
    ds = ds_labels.pp.drop_layers("_percentage_positive")
    with pytest.raises(KeyError, match=f'No channel "{channel}".'):
        ds.la._threshold_label(channel=channel, threshold=10.0, layer_key="_intensity")


def test_thresholding(ds_labels):
    ds = ds_labels.pp.drop_layers("_percentage_positive").pp.drop_observations("CD4_binarized")
    channel = "CD4"
    threshold = 6.0
    layer_key = "_intensity"

    binarized = ds.la._threshold_label(channel=channel, threshold=threshold, layer_key=layer_key)

    manual = ds._intensity.sel(channels=channel).values >= threshold
    computed = binarized._obs.sel(features=f"{channel}_binarized").values.astype(bool)
    assert np.all(manual == computed)


def test_thresholding_on_label(ds_labels):
    ds = ds_labels.pp.drop_layers("_percentage_positive").pp.drop_observations("CD4_binarized")
    channel = "CD4"
    threshold = 6.0
    layer_key = "_intensity"
    label = "T"

    binarized_label = ds.la._threshold_label(channel=channel, threshold=threshold, label=label, layer_key=layer_key)

    label_pos = ds._intensity.sel(channels=channel).values >= threshold
    label_mask = np.isin(ds.coords["cells"].values, ds.la[label].coords["cells"].values).astype(int)

    manual = label_pos * label_mask
    computed = binarized_label._obs.sel(features=f"{channel}_{label}_binarized").values
    assert np.all(manual == computed)


# === SPATIALDATA BACKEND ===
# def test_threshold_labels_spatialdata(dataset_labeled):
#     threshold_dict = {"CD4": 0.5, "CD8": 0.6}
#     data = dataset_labeled.pp.add_quantification().tl.convert_to_spatialdata()
#     sp.pp.add_quantification(data, func=sp.percentage_positive, layer_key="perc_pos")
#     sp.la.threshold_labels(data, threshold_dict=threshold_dict)


# def test_layer_does_not_exist_spatialdata(dataset_labeled):
#     threshold_dict = {"CD4": 0.5, "CD8": 0.6}
#     data = dataset_labeled.pp.add_quantification().tl.convert_to_spatialdata()
#     with pytest.raises(AssertionError, match="Layer perc_pos not found in adata object."):
#         sp.la.threshold_labels(data, threshold_dict=threshold_dict)


# def test_channel_does_not_exist_spatialdata(dataset_labeled):
#     threshold_dict = {"CD4": 0.5, "Dummy": 0.6}
#     data = dataset_labeled.pp.add_quantification().tl.convert_to_spatialdata()
#     sp.pp.add_quantification(data, func=sp.percentage_positive, layer_key="perc_pos")
#     with pytest.raises(KeyError, match="Channel Dummy not found in the expression matrix"):
#         sp.la.threshold_labels(data, threshold_dict=threshold_dict)
