import networkx as nx
import numpy as np

from spatialproteomics.nh.utils import _compute_network_features


# === FUNCTIONAL TESTS ===
def test_degree():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (3, 4)])
    expected_degrees = {0: 2, 1: 3, 2: 2, 3: 2, 4: 1}
    features = _compute_network_features(G, "degree")["degree"]
    for node, expected_degree in expected_degrees.items():
        assert features[node] == expected_degree, f"Degree mismatch for node {node}"


# not testing betweenness and centrality, since they are based on networkx functions


def test_homophily():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (3, 4)])
    nx.set_node_attributes(G, {0: "A", 1: "A", 2: "B", 3: "B", 4: "A"}, "_labels")
    expected_homophily = {0: 0.5, 1: 0.3333333333333333, 2: 0.0, 3: 0.0, 4: 0.0}
    features = _compute_network_features(G, "homophily")["homophily"]
    for node, expected_value in expected_homophily.items():
        assert np.isclose(
            features[node], expected_value
        ), f"Homophily mismatch for node {node}. Expected {expected_value}, got {features[node]}"


def test_inter_label_connectivity():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (3, 4)])
    nx.set_node_attributes(G, {0: "A", 1: "A", 2: "B", 3: "B", 4: "A"}, "_labels")
    expected_connectivity = {0: 0.5, 1: 0.66666666666666, 2: 1.0, 3: 1.0, 4: 1.0}
    features = _compute_network_features(G, "inter_label_connectivity")["inter_label_connectivity"]
    for node, expected_value in expected_connectivity.items():
        assert np.isclose(
            features[node], expected_value
        ), f"Inter-label connectivity mismatch for node {node}. Expected {expected_value}, got {features[node]}"


def test_diversity_index():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (3, 4)])
    nx.set_node_attributes(G, {0: "A", 1: "A", 2: "B", 3: "B", 4: "A"}, "_labels")
    # computed semi-manually (calling the entropy function from scipy.stats)
    expected_diversity = {0: 0.6931471805599453, 1: 0.6365141682948128, 2: 0.0, 3: 0.0, 4: 0.0}
    features = _compute_network_features(G, "diversity_index")["diversity_index"]
    for node, expected_value in expected_diversity.items():
        assert np.isclose(
            features[node], expected_value
        ), f"Diversity index mismatch for node {node}. Expected {expected_value}, got {features[node]}"
