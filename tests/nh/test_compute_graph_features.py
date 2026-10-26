def test_compute_graph_features(ds_neighborhoods):
    graph_features = ds_neighborhoods.nh.compute_graph_features(features=["num_nodes", "num_edges"])
    assert graph_features["num_nodes"] == 56
    # the stored adjacency matrix contains 1554 edges, 56 of which are self-loops that are not counted
    assert graph_features["num_edges"] == 1498
