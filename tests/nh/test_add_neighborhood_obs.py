def test_add_neighborhood_obs(ds_neighborhoods):
    ds_with_nh = ds_neighborhoods.nh.add_neighborhood_obs(features="degree")
    assert "degree" in ds_with_nh.pp.get_layer_as_df().columns
