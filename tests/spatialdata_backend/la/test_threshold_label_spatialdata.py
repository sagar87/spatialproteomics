import pytest

import spatialproteomics as sp


def test_threshold_labels_spatialdata(ds_labels_spatialdata):
    threshold_dict = {"CD4": 0.5, "CD8": 0.6}
    ds = sp.pp.add_quantification(ds_labels_spatialdata, func=sp.percentage_positive, layer_key="perc_pos", copy=True)
    sp.la.threshold_labels(ds, threshold_dict=threshold_dict, copy=True)


def test_layer_does_not_exist_spatialdata(ds_labels_spatialdata):
    threshold_dict = {"CD4": 0.5, "CD8": 0.6}
    with pytest.raises(AssertionError, match="Layer perc_pos not found in adata object."):
        sp.la.threshold_labels(ds_labels_spatialdata, threshold_dict=threshold_dict)


def test_channel_does_not_exist_spatialdata(ds_labels_spatialdata):
    threshold_dict = {"CD4": 0.5, "Dummy": 0.6}
    ds = sp.pp.add_quantification(ds_labels_spatialdata, func=sp.percentage_positive, layer_key="perc_pos", copy=True)
    with pytest.raises(KeyError, match="Channel Dummy not found in the expression matrix"):
        sp.la.threshold_labels(ds, threshold_dict=threshold_dict)
