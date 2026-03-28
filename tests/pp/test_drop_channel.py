import pytest

from spatialproteomics.constants import Dims, Layers


def test_drop_channel_single(ds_labels):
    ds = ds_labels.pp.drop_channel("CD4")
    assert "CD4" not in ds.coords[Dims.CHANNELS].values
    assert "CD8" in ds.coords[Dims.CHANNELS].values
    assert "DAPI" in ds.coords[Dims.CHANNELS].values
    # ensuring that both the image and the intensity layer are updated
    image_shape_old = ds_labels[Layers.IMAGE].shape
    intensity_shape_old = ds_labels[Layers.INTENSITY].shape
    image_shape_new = ds[Layers.IMAGE].shape
    intensity_shape_new = ds[Layers.INTENSITY].shape
    assert (
        image_shape_new[0] == image_shape_old[0] - 1
    ), "The number of channels in the image layer should be reduced by 1 after dropping a channel."
    assert (
        intensity_shape_new[1] == intensity_shape_old[1] - 1
    ), "The number of channels in the intensity layer should be reduced by 1 after dropping a channel."
    # ensuring that nothing happened in-place
    assert "CD4" in ds_labels.coords[Dims.CHANNELS].values


def test_drop_channel_multi(ds_labels):
    ds = ds_labels.pp.drop_channel(["CD4", "DAPI"])
    assert "CD4" not in ds.coords[Dims.CHANNELS].values
    assert "CD8" in ds.coords[Dims.CHANNELS].values
    assert "DAPI" not in ds.coords[Dims.CHANNELS].values
    # ensuring that both the image and the intensity layer are updated
    image_shape_old = ds_labels[Layers.IMAGE].shape
    intensity_shape_old = ds_labels[Layers.INTENSITY].shape
    image_shape_new = ds[Layers.IMAGE].shape
    intensity_shape_new = ds[Layers.INTENSITY].shape
    assert (
        image_shape_new[0] == image_shape_old[0] - 2
    ), "The number of channels in the image layer should be reduced by 2 after dropping two channels."
    assert (
        intensity_shape_new[1] == intensity_shape_old[1] - 2
    ), "The number of channels in the intensity layer should be reduced by 2 after dropping two channels."
    # ensuring that nothing happened in-place
    assert "CD4" in ds_labels.coords[Dims.CHANNELS].values
    assert "DAPI" in ds_labels.coords[Dims.CHANNELS].values


def test_drop_channel_all_channels(ds_labels):
    channels = ds_labels.coords[Dims.CHANNELS].values
    ds = ds_labels.pp.drop_channel(channels)
    assert len(ds.coords[Dims.CHANNELS].values) == 0
    # checking that the image and intensity layers are empty
    assert ds[Layers.IMAGE].shape[0] == 0
    assert ds[Layers.INTENSITY].shape[1] == 0
    # ensuring that nothing happened in-place
    assert len(ds_labels.coords[Dims.CHANNELS].values) == 5


def test_drop_channel_no_channel(ds_labels):
    with pytest.raises(
        AssertionError,
        match="Some of the channels to be dropped do not exist in the object: {'dummy_channel'}. Please set the channel_names argument to a list of existing channels.",
    ):
        ds_labels.pp.drop_channel("dummy_channel")
