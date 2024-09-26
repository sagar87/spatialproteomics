import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, KDTree
from skimage.segmentation import relabel_sequential
from sklearn.neighbors import NearestNeighbors

from ..base_logger import logger


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
        logger.warning("Neighborhoods are non-consecutive. Relabeling...")
        formatted_neighborhoods, _, _ = relabel_sequential(formatted_neighborhoods)

    return formatted_neighborhoods


def _construct_neighborhood_df_radius(df, cell_types, x="centroid-1", y="centroid-0", label_col="labels", radius=100):
    """
    Constructs a neighborhood profile DataFrame for each cell in the input DataFrame.
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
        Radius within which to search for neighboring cells (default is 100).
    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a cell and each column corresponds to a cell type.
        The values represent the normalized counts of neighboring cell types within the specified radius.
    """
    # Build KDTree for efficient neighborhood queries
    tree = KDTree(df[[x, y]])

    # Initialize a DataFrame to store the neighborhood counts
    neighborhood_profile = pd.DataFrame(0, index=range(len(df)), columns=cell_types)

    # resetting the index to start from 0, makes accessing with loc easier
    original_index = df.index
    df = df.reset_index()

    # Iterate over each cell
    for i in range(len(df)):
        # Query the KDTree for neighbors within the radius (including the center cell)
        indices = tree.query_ball_point(df.loc[i, [x, y]], r=radius)

        # Count the cell types of the neighbors
        neighbor_counts = df.loc[indices, label_col].value_counts()

        # Update the neighborhood profile for the current cell
        for cell_type, count in neighbor_counts.items():
            neighborhood_profile.at[i, cell_type] = count

    # Normalize the neighborhood profile so that each row sums to 1
    neighborhood_profile = neighborhood_profile.div(neighborhood_profile.sum(axis=1), axis=0)

    # setting the index back to the original ones
    neighborhood_profile.index = original_index

    return neighborhood_profile


def _construct_neighborhood_df_knn(df, cell_types, x="centroid-1", y="centroid-0", label_col="labels", k=10):
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

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a cell and each column corresponds to a cell type.
        The values represent the normalized counts of neighboring cell types within the k-nearest neighbors.
    """
    # Build NearestNeighbors for k-nearest neighbor queries
    knn = NearestNeighbors(n_neighbors=k + 1)  # +1 because the first neighbor is the cell itself
    knn.fit(df[[x, y]])

    # Get the indices of the k-nearest neighbors for each cell
    distances, indices = knn.kneighbors(df[[x, y]])

    # Initialize a DataFrame to store the neighborhood counts
    neighborhood_profile = pd.DataFrame(0, index=range(len(df)), columns=cell_types)

    # Resetting the index to start from 0, makes accessing with loc easier
    original_index = df.index
    df = df.reset_index()

    # Iterate over each cell
    for i, neighbors in enumerate(indices):
        # Count the cell types of the neighbors
        neighbor_counts = df.loc[neighbors, label_col].value_counts()

        # Update the neighborhood profile for the current cell
        for cell_type, count in neighbor_counts.items():
            neighborhood_profile.at[i, cell_type] = count

    # Normalize the neighborhood profile so that each row sums to 1
    neighborhood_profile = neighborhood_profile.div(neighborhood_profile.sum(axis=1), axis=0)

    # Setting the index back to the original ones
    neighborhood_profile.index = original_index

    return neighborhood_profile


def _construct_neighborhood_df_delaunay(df, cell_types, x="centroid-1", y="centroid-0", label_col="labels"):
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

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a cell and each column corresponds to a cell type.
        The values represent the normalized counts of neighboring cell types connected via Delaunay triangulation.
    """
    # Extract the coordinates from the DataFrame
    coords = df[[x, y]].values

    # Perform Delaunay triangulation
    tri = Delaunay(coords)

    # Initialize a DataFrame to store the neighborhood counts
    neighborhood_profile = pd.DataFrame(0, index=range(len(df)), columns=cell_types)

    # resetting the index to start from 0, makes accessing with loc easier
    original_index = df.index
    df = df.reset_index()

    # Iterate over each cell
    for i in range(len(df)):
        # Get the indices of the neighboring cells connected via Delaunay triangulation
        neighbors = tri.vertex_neighbor_vertices[1][
            tri.vertex_neighbor_vertices[0][i] : tri.vertex_neighbor_vertices[0][i + 1]
        ]

        # Count the cell types of the neighbors
        neighbor_counts = df.loc[neighbors, label_col].value_counts()

        # Update the neighborhood profile for the current cell
        for cell_type, count in neighbor_counts.items():
            neighborhood_profile.at[i, cell_type] = count

    # Normalize the neighborhood profile so that each row sums to 1
    neighborhood_profile = neighborhood_profile.div(neighborhood_profile.sum(axis=1), axis=0)

    # setting the index back to the original ones
    neighborhood_profile.index = original_index

    return neighborhood_profile
