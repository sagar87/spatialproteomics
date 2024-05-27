import numpy as np
import pytest


def test_layer_does_not_exist(dataset_labeled):
    # add _intensity_layer but not _percentage_positive
    data = dataset_labeled.pp.add_quantification()
    with pytest.raises(KeyError, match="Please add it first using pp.add_quantification()."):
        data.la._threshold_label(channel="CD4", threshold=10.0, layer_key="_percentage_positive")


def test_channel_does_not_exist(dataset_labeled):
    channel = "NPC"
    data = dataset_labeled.pp.add_quantification()
    with pytest.raises(KeyError, match=f'No channel "{channel}".'):
        data.la._threshold_label(channel=channel, threshold=10.0, layer_key="_intensity")


def test_threholding(dataset_labeled):
    data = dataset_labeled.pp.add_quantification()
    channel = "CD4"
    threshold = 6.0
    layer_key = "_intensity"

    binarized = data.la._threshold_label(channel=channel, threshold=threshold, layer_key=layer_key)

    manual = data._intensity.sel(channels=channel).values >= threshold
    computed = binarized._obs.sel(features=f"{channel}_binarized").values.astype(bool)
    assert np.all(manual == computed)


def test_threholding_on_label(dataset_labeled):
    data = dataset_labeled.pp.add_quantification()
    channel = "CD4"
    threshold = 6.0
    layer_key = "_intensity"
    label = "Cell type 5"

    binarized_label = data.la._threshold_label(channel=channel, threshold=threshold, label=label, layer_key=layer_key)

    label_pos = data._intensity.sel(channels=channel).values >= threshold
    label_mask = np.isin(data.coords["cells"].values, data.la[label].coords["cells"].values).astype(int)

    manual = label_pos * label_mask
    computed = binarized_label._obs.sel(features=f"{channel}_binarized").values
    assert np.all(manual == computed)
