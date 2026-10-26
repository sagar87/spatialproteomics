import numpy as np

from spatialproteomics.constants import Dims, Features, Labels, Layers


def test_remove_label_type_with_existing_unlabeled_keeps_compositions(ds_labels):
    # if label 0 (unlabeled) already exists, the removed types are attributed to it in the neighborhood compositions
    df = ds_labels.pp.get_layer_as_df()
    df.loc[df.index[:5], Features.LABELS] = Labels.UNLABELED
    df["cell"] = df.index
    ds = ds_labels.pp.drop_layers(Layers.LA_PROPERTIES).la.add_labels_from_dataframe(df, label_col=Features.LABELS)
    assert 0 in ds.coords[Dims.LABELS].values
    ds = ds.nh.compute_neighborhoods_radius(radius=50)

    removed_id = ds.la._label_name_to_id("B")
    ds_removed = ds.la.remove_label_type("B")

    compositions = ds_removed[Layers.NEIGHBORHOODS]
    assert np.allclose(compositions.sum(Dims.LABELS).values, 1)
    expected = ds[Layers.NEIGHBORHOODS].sel({Dims.LABELS: [0, removed_id]}).sum(Dims.LABELS)
    assert np.allclose(compositions.sel({Dims.LABELS: 0}).values, expected.values)
    # the input is not modified
    assert removed_id in ds[Layers.NEIGHBORHOODS].coords[Dims.LABELS].values
    assert np.allclose(ds[Layers.NEIGHBORHOODS].sum(Dims.LABELS).values, 1)
