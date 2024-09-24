import pandas as pd
from scipy.spatial import KDTree


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
