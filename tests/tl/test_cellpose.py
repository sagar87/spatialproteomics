import pytest

from spatialproteomics.constants import Layers


def test_cellpose_with_layer_key_segmentation(dataset_full):
    dataset = dataset_full.pp[["Hoechst", "CD4", "CD8"]].pp[250:300, 250:300].pp.drop_layers([Layers.SEGMENTATION])
    with pytest.raises(
        KeyError,
        match=f'The key "{Layers.SEGMENTATION}" is reserved',
    ):
        dataset.tl.cellpose(key_added=Layers.SEGMENTATION, gpu=False)


def test_cellpose_segmentation_already_exists(dataset_full):
    with pytest.raises(
        KeyError,
        match=f'The key "{Layers.SEGMENTATION}" already exists.',
    ):
        dataset_full.tl.cellpose(key_added=Layers.SEGMENTATION)


def test_cellpose_joint_segmentation_multichannel(dataset_full):
    with pytest.raises(
        AssertionError,
        match="Joint segmentation requires exactly two channels.",
    ):
        dataset_full.tl.cellpose(key_added="_dummy_key", channel_settings=[1, 2])
