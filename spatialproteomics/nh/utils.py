import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from skimage.segmentation import relabel_sequential
from sklearn.neighbors import NearestNeighbors

from ..base_logger import logger
from ..constants import Features


def _format_neighborhoods(neighborhoods):
    """
    Format the neighborhoods array to ensure consecutive numbering.

    Parameters
    ----------
    neighborhoods : numpy.ndarray
        The input array of neighborhoods.

    Returns
    -------
    numpy.ndarray
        The formatted array of neighborhoods with consecutive numbering.
    """

    formatted_neighborhoods = neighborhoods.copy()
    unique_neighborhoods = np.unique(neighborhoods)

    if ~np.all(np.diff(unique_neighborhoods) == 1):
        formatted_neighborhoods, _, _ = relabel_sequential(formatted_neighborhoods)

    return formatted_neighborhoods


def _construct_neighborhood_df_radius(
    df, cell_types, x="centroid-1", y="centroid-0", label_col="labels", radius=100, include_center=True
):
    """
    Constructs a neighborhood profile DataFrame and an adjacency matrix for each cell in the input DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing cell data with spatial coordinates and labels.
    cell_types : list
        List of cell types to consider for the neighborhood profile.
    x : str, optional
        Column name for the x-coordinate (default is 'centroid-1').
    y : str, optional
        Column name for the y-coordinate (default is 'centroid-0').
    label_col : str, optional
        Column name for the cell type labels (default is 'labels').
    radius : float, optional
        Radius within which to search for neighboring cells (default is 100 pixels).
    include_center : bool, optional
        Whether to include the center cell in the neighborhood profile (default is True).

    Returns
    -------
    neighborhood_profile : pandas.DataFrame
        A DataFrame where each row corresponds to a cell and each column corresponds to a cell type.
        The values represent the normalized counts of neighboring cell types within the specified radius.
    adjacency_matrix : np.ndarray
        A square adjacency matrix where each element (i, j) is 1 if cell 'i' is within the radius of cell 'j', 0 otherwise.
    """
    # Build KDTree for efficient neighborhood queries
    tree = KDTree(df[[x, y]])

    # Initialize a DataFrame to store the neighborhood counts
    neighborhood_profile = pd.DataFrame(0, index=range(len(df)), columns=cell_types)

    # Initialize adjacency matrix (N x N, where N is the number of cells)
    N = len(df)
    adjacency_matrix = np.zeros((N, N), dtype=int)

    # Resetting the index to start from 0, makes accessing with loc easier
    original_index = df.index
    df = df.reset_index()

    # Iterate over each cell
    for i in range(N):
        # Query the KDTree for neighbors within the radius (including the center cell)
        indices = tree.query_ball_point(df.loc[i, [x, y]], r=radius)

        # If include_center is False, exclude the center cell (i.e., cell 'i') from the indices
        if not include_center:
            indices = [idx for idx in indices if idx != i]

        # Update adjacency matrix: Set 1 for neighbors within the radius
        adjacency_matrix[i, indices] = 1

        # Count the cell types of the neighbors
        neighbor_counts = df.loc[indices, label_col].value_counts()

        # Update the neighborhood profile for the current cell
        for cell_type, count in neighbor_counts.items():
            neighborhood_profile.at[i, cell_type] = count

    # Normalize the neighborhood profile so that each row sums to 1
    neighborhood_profile = neighborhood_profile.div(neighborhood_profile.sum(axis=1), axis=0)

    # Handle cases where the neighborhood profile contains NaN values (e.g., no neighbors found)
    if neighborhood_profile.isna().sum().sum() > 0:
        logger.warning(
            "Some neighborhoods contained no cells. This may be due to the neighborhood radius being too small, or the center cell not being included. These neighborhoods will be set to 0 everywhere."
        )
        neighborhood_profile.fillna(0, inplace=True)

    # Restore the original index for the neighborhood profile DataFrame
    neighborhood_profile.index = original_index

    return neighborhood_profile, adjacency_matrix


def _construct_neighborhood_df_knn(
    df, cell_types, x="centroid-1", y="centroid-0", label_col="labels", k=10, include_center=True
):
    """
    Constructs a neighborhood profile DataFrame for each cell using k-nearest neighbors.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing cell data with spatial coordinates and labels.
    cell_types : list
        List of cell types to consider for the neighborhood profile.
    x : str, optional
        Column name for the x-coordinate (default is 'centroid-1').
    y : str, optional
        Column name for the y-coordinate (default is 'centroid-0').
    label_col : str, optional
        Column name for the cell type labels (default is 'labels').
    k : int, optional
        The number of nearest neighbors to consider (default is 10).
    include_center : bool, optional
        Whether to include the center cell in the neighborhood profile (default is True).

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a cell and each column corresponds to a cell type.
        The values represent the normalized counts of neighboring cell types within the k-nearest neighbors.
    adjacency_matrix : np.ndarray
        A square adjacency matrix where each element (i, j) is 1 if cell 'i' is within the radius of cell 'j', 0 otherwise.
    """
    # Build NearestNeighbors for k-nearest neighbor queries
    knn = NearestNeighbors(n_neighbors=k + 1)  # +1 because the first neighbor is the cell itself
    knn.fit(df[[x, y]])

    # Get the indices of the k-nearest neighbors for each cell
    distances, indices = knn.kneighbors(df[[x, y]])

    # Initialize a DataFrame to store the neighborhood counts
    neighborhood_profile = pd.DataFrame(0, index=range(len(df)), columns=cell_types)

    # Initialize adjacency matrix (N x N, where N is the number of cells)
    N = len(df)
    adjacency_matrix = np.zeros((N, N), dtype=int)

    # Resetting the index to start from 0, makes accessing with loc easier
    original_index = df.index
    df = df.reset_index()

    # Iterate over each cell
    for i, neighbors in enumerate(indices):
        if not include_center:
            # Exclude the first element (the center cell)
            neighbors = neighbors[1:]

        # Update adjacency matrix: Set 1 for neighbors within the radius
        adjacency_matrix[i, indices] = 1

        # Count the cell types of the neighbors
        neighbor_counts = df.loc[neighbors, label_col].value_counts()

        # Update the neighborhood profile for the current cell
        for cell_type, count in neighbor_counts.items():
            neighborhood_profile.at[i, cell_type] = count

    # Normalize the neighborhood profile so that each row sums to 1
    neighborhood_profile = neighborhood_profile.div(neighborhood_profile.sum(axis=1), axis=0)

    # Setting the index back to the original ones
    neighborhood_profile.index = original_index

    return neighborhood_profile, adjacency_matrix


def _construct_neighborhood_df_delaunay(
    df, cell_types, x="centroid-1", y="centroid-0", label_col="labels", include_center=True
):
    """
    Constructs a neighborhood profile DataFrame for each cell using Delaunay triangulation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing cell data with spatial coordinates and labels.
    cell_types : list
        List of cell types to consider for the neighborhood profile.
    x : str, optional
        Column name for the x-coordinate (default is 'centroid-1').
    y : str, optional
        Column name for the y-coordinate (default is 'centroid-0').
    label_col : str, optional
        Column name for the cell type labels (default is 'labels').
    include_center : bool, optional
        Whether to include the center cell in the neighborhood profile (default is True).

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a cell and each column corresponds to a cell type.
        The values represent the normalized counts of neighboring cell types connected via Delaunay triangulation.
    adjacency_matrix : np.ndarray
        A square adjacency matrix where each element (i, j) is 1 if cell 'i' is within the radius of cell 'j', 0 otherwise.
    """
    # Extract the coordinates from the DataFrame
    coords = df[[x, y]].values

    # Perform Delaunay triangulation
    tri = Delaunay(coords)

    # Initialize a DataFrame to store the neighborhood counts
    neighborhood_profile = pd.DataFrame(0, index=range(len(df)), columns=cell_types)

    # Initialize adjacency matrix (N x N, where N is the number of cells)
    N = len(df)
    adjacency_matrix = np.zeros((N, N), dtype=int)

    # resetting the index to start from 0, makes accessing with loc easier
    original_index = df.index
    df = df.reset_index()

    # Iterate over each cell
    for i in range(len(df)):
        # Get the indices of the neighboring cells connected via Delaunay triangulation
        neighbors = tri.vertex_neighbor_vertices[1][
            tri.vertex_neighbor_vertices[0][i] : tri.vertex_neighbor_vertices[0][i + 1]
        ]

        # Optionally include the center cell (i.e., the cell itself)
        if include_center:
            neighbors = list(neighbors) + [i]

        # Update adjacency matrix: Set 1 for neighbors within the radius
        adjacency_matrix[i, neighbors] = 1

        # Count the cell types of the neighbors
        neighbor_counts = df.loc[neighbors, label_col].value_counts()

        # Update the neighborhood profile for the current cell
        for cell_type, count in neighbor_counts.items():
            neighborhood_profile.at[i, cell_type] = count

    # Normalize the neighborhood profile so that each row sums to 1
    neighborhood_profile = neighborhood_profile.div(neighborhood_profile.sum(axis=1), axis=0)

    # setting the index back to the original ones
    neighborhood_profile.index = original_index

    return neighborhood_profile, adjacency_matrix


def _compute_network_features(G, features):
    """
    Computes various network features for each node in the graph.

    Parameters:
    G (networkx.Graph): Input network graph.
    features (list): List of features to compute.

    Returns:
    pandas.DataFrame: DataFrame with node features.
    """
    import networkx as nx

    # Initialize a dictionary to store features for each node
    feature_df = {}

    # Degree
    if "degree" in features:
        degree_dict = dict(G.degree())
        feature_df["degree"] = degree_dict

    # Closeness Centrality
    if "closeness_centrality" in features:
        closeness_dict = nx.closeness_centrality(G)
        feature_df["closeness_centrality"] = closeness_dict

    # Betweenness Centrality
    if "betweenness_centrality" in features:
        betweenness_dict = nx.betweenness_centrality(G)
        feature_df["betweenness_centrality"] = betweenness_dict

    # Homophily
    if "homophily" in features:
        homophily_dict = _compute_node_homophily(G)
        feature_df["homophily"] = homophily_dict

    # Inter-label Connectivity
    if "inter_label_connectivity" in features:
        inter_label_connectivity_dict = _compute_inter_label_connectivity(G)
        feature_df["inter_label_connectivity"] = inter_label_connectivity_dict

    # Diversity Index
    if "diversity_index" in features:
        diversity_dict = _compute_diversity_index(G)
        feature_df["diversity_index"] = diversity_dict

    # Create DataFrame from the features dictionary
    feature_df = pd.DataFrame(feature_df)

    return feature_df


def _compute_node_homophily(G, label_col=Features.LABELS):
    """
    Computes the homophily score for each node in the graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph with node attributes.
    label_col : str
        The attribute name for node labels (default is '_labels').

    Returns
    -------
    dict
        A dictionary mapping each node to its homophily score.
    """
    homophily_scores = {}

    for node in G.nodes():
        total_edges = G.degree(node)  # Total number of edges connected to the node
        if total_edges == 0:
            homophily_scores[node] = 0  # No edges connected, homophily score is 0
            continue

        homophilic_edges = 0

        for neighbor in G.neighbors(node):
            if G.nodes[node][label_col] == G.nodes[neighbor][label_col]:
                homophilic_edges += 1

        # Compute homophily score
        homophily_score = homophilic_edges / total_edges
        homophily_scores[node] = homophily_score

    return homophily_scores


def _compute_inter_label_connectivity(G, label_col=Features.LABELS):
    """
    Computes the inter-label connectivity score for each node in the graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph with node attributes.
    label_col : str
        The attribute name for node labels (default is '_labels').

    Returns
    -------
    dict
        A dictionary mapping each node to its inter-label connectivity score.
    """
    inter_label_connectivity_scores = {}

    for node in G.nodes():
        total_edges = G.degree(node)
        if total_edges == 0:
            inter_label_connectivity_scores[node] = 0  # No edges connected, score is 0
            continue

        inter_label_edges = 0

        for neighbor in G.neighbors(node):
            if G.nodes[node][label_col] != G.nodes[neighbor][label_col]:
                inter_label_edges += 1

        # Compute inter-label connectivity score
        inter_label_connectivity_score = inter_label_edges / total_edges
        inter_label_connectivity_scores[node] = inter_label_connectivity_score

    return inter_label_connectivity_scores


def _compute_diversity_index(G, label_col=Features.LABELS):
    """
    Computes the diversity index (Shannon's diversity index) for each node in the graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph with node attributes.
    label_col : str
        The attribute name for node labels (default is '_labels').

    Returns
    -------
    dict
        A dictionary mapping each node to its diversity index.
    """
    diversity_index_scores = {}

    for node in G.nodes():
        total_edges = G.degree(node)
        if total_edges == 0:
            diversity_index_scores[node] = 0  # No edges connected, diversity is 0
            continue

        # Count the occurrence of each label among the neighbors
        label_counts = {}
        for neighbor in G.neighbors(node):
            neighbor_label = G.nodes[neighbor][label_col]
            label_counts[neighbor_label] = label_counts.get(neighbor_label, 0) + 1

        # Compute proportions of each label
        proportions = np.array(list(label_counts.values())) / total_edges

        # Compute Shannon's diversity index
        diversity_index = -np.sum(proportions * np.log(proportions))
        diversity_index_scores[node] = diversity_index

    return diversity_index_scores


def _compute_global_network_features(G, features):
    """
    Compute selected network features for the given graph.

    Args:
        G (networkx.Graph): Input graph.
        features (list): List of features to compute. Possible values: ['density', 'diameter', 'modularity', 'assortativity']

    Returns:
        results (dict): Dictionary with feature names as keys and computed values as values.
    """
    import networkx as nx

    results = {}

    if "num_nodes" in features:
        results["num_nodes"] = G.number_of_nodes()

    if "num_edges" in features:
        results["num_edges"] = G.number_of_edges()

    # Compute graph density
    if "density" in features:
        results["density"] = nx.density(G)

    # Compute modularity using the Louvain method
    if "modularity" in features:
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(G)
            results["modularity"] = community_louvain.modularity(partition, G)
        except ImportError:
            raise ImportError("Please install the 'python-louvain' package to compute modularity.")

    # Compute assortativity (degree assortativity coefficient)
    if "assortativity" in features:
        results["assortativity"] = nx.degree_assortativity_coefficient(G)

    return results
