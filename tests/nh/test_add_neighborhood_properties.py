from spatialproteomics.constants import Dims, Layers, Props


def test_add_neighborhood_property(ds_neighborhoods):
    # the neighborhood properties of the test file are read from zarr (StringDType with xarray>=2026)
    neighborhoods = ds_neighborhoods.coords[Dims.NEIGHBORHOODS].values
    values = [f"value {i}" for i in neighborhoods]

    ds = ds_neighborhoods.nh.add_properties(values, "dummy_property")

    props = ds.pp.get_layer_as_df(Layers.NH_PROPERTIES)
    assert list(props["dummy_property"]) == values
    # the existing properties are kept
    old_props = ds_neighborhoods.pp.get_layer_as_df(Layers.NH_PROPERTIES)
    assert list(props[Props.NAME]) == list(old_props[Props.NAME])
    assert list(props[Props.COLOR]) == list(old_props[Props.COLOR])
