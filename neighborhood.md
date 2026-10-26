# The neighborhood (`nh`) accessor

The neighborhood accessor provides several functions associated with cellular neighborhoods.

## `ds.nh` (NeighborhoodAccessor)

Adds functions for cellular neighborhoods.

Available on any `xarray.Dataset` after `import spatialproteomics as sp`, as `ds.nh`.

### `ds.nh.deselect(indices)`

Deselect specific neighborhood indices from the data object.

This method deselects specific neighborhood indices from the data object, effectively removing them from the selection.
The deselection can be performed using slices, lists, tuples, or individual integers.

**Parameters**

indices : slice, list, tuple, or int
    The neighborhood indices to be deselected. Can be a slice, list, tuple, or an individual integer.

**Returns**

xr.Dataset
    The updated data object with the deselected neighborhood indices.

**Notes**

- The function uses 'indices' to specify which neighborhoods to deselect.
- 'indices' can be provided as slices, lists, tuples, or an integer.
- The function then updates the data object to remove the deselected neighborhood indices.

### `ds.nh.add_properties(array: Union[np.ndarray, list], prop: str='_neighborhoods') -> xr.Dataset`

Adds neighborhood properties to the image container.

**Parameters**

array : Union[np.ndarray, list]
    An array or list of properties to be added to the image container.
prop : str, optional
    The name of the property. Default is Features.NEIGHBORHOODS.

**Returns**

xr.Dataset
    The updated image container with added properties.

### `ds.nh.add_neighborhoods_from_dataframe(df: pd.DataFrame, neighborhood_col: str='_neighborhoods', colors: Optional[list]=None, names: Optional[list]=None) -> xr.Dataset`

Add neighborhoods to the dataset from a DataFrame.

**Parameters**

df : pd.DataFrame
    DataFrame containing neighborhood information.
neighborhood_col : str, optional
    Column name in the DataFrame that contains neighborhood labels, by default '_neighborhoods'.
colors : list or None, optional
    List of colors for the neighborhoods, by default None. If None, random colors will be assigned.
names : list or None, optional
    List of names for the neighborhoods, by default None. If None, default names will be assigned.

**Returns**

xr.Dataset
    Updated dataset with neighborhood information added.

**Raises**

AssertionError
    If neighborhood properties are already present in the object.
    If the number of neighborhoods does not match the number of cells.
    If the specified column does not exist in the DataFrame.
    If neighborhoods contain negative values.
    If the length of colors does not match the number of unique neighborhoods.
    If the length of names does not match the number of unique neighborhoods.

### `ds.nh.set_neighborhood_colors(neighborhoods: Union[str, List[str]], colors: Union[str, List[str]], suppress_warnings: bool=False)`

Set the color of a specific neighborhood.

This method sets the 'color' of a specific neighborhood identified by the 'neighborhood'.
The 'neighborhood' can be either a neighborhood ID or the name of the neighborhood.

**Parameters**

label : int or str
    The ID or name of the neighborhood whose color will be updated.
color : any
    The new color to be assigned to the specified neighborhood.
suppress_warnings : bool, optional
    Whether to suppress warnings. Default is False.

**Returns**

xr.Dataset

**Notes**

- The function converts the 'neighborhood' from its name to the corresponding ID for internal processing.
- It updates the color of the neighborhood in the data object to the new 'color'.

### `ds.nh.set_neighborhood_name(neighborhoods: Union[int, str, List], names: Union[str, List])`

Set the name of one or more neighborhoods.

This method updates the 'name' of specific neighborhoods identified by the 'neighborhood'.
The 'neighborhoods' can be a single neighborhood ID or name, or a list of IDs/names. Similarly,
the 'names' parameter can be a single string or a list of strings.

**Parameters**

neighborhoods : int, str, or list
    The ID(s) or name(s) of the neighborhoods whose names will be updated.
names : str or list
    The new name(s) to be assigned to the specified neighborhoods.

**Returns**

xr.Dataset

**Notes**

- When both parameters are lists, their lengths must match.
- The function converts each 'neighborhood' from its name to the corresponding ID for internal processing.
- It updates the name(s) of the neighborhood(s) in the data object to the new 'name(s)'.

### `ds.nh.compute_neighborhoods_radius(radius: int=100, include_center: bool=True, key_added: str='_neighborhoods', key_adjacency_matrix: str='_adjacency_matrix')`

Compute the neighborhoods of each cell based on a specified radius.

This method defines a radius around each cell and identifies all cells within this radius.
It then examines the predicted cell types of these cells, including the center cell itself,
and computes the frequency of each cell type.

**Parameters**

radius : int, optional
    The radius around each cell to define the neighborhood (in pixels). Default is 100.
include_center : bool, optional
    Whether to include the center cell in the neighborhood. Default is True.
key_added : str, optional
    The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is '_neighborhoods'.
key_adjacency_matrix : str, optional
    The key under which the computed adjacency matrix will be stored in the resulting DataArray. Default is '_adjacency_matrix'.

**Returns**

xarray.Dataset
    A merged xarray Dataset containing the original data and the computed neighborhoods.

### `ds.nh.compute_neighborhoods_knn(k=10, include_center: bool=True, key_added: str='_neighborhoods', key_adjacency_matrix: str='_adjacency_matrix')`

Compute the neighborhoods of each cell based on k-nearest neighbors.

This method identifies the k-nearest neighbors for each cell and computes
the frequency of each cell type within these k neighbors.

**Parameters**

k : int, optional
    The number of nearest neighbors to consider, not counting the center cell. Default is 10.
include_center : bool, optional
    Whether to add the center cell to its k nearest neighbors in the neighborhood. Default is True.
key_added : str, optional
    The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is '_neighborhoods'.
key_adjacency_matrix : str, optional
    The key under which the computed adjacency matrix will be stored in the resulting DataArray. Default is '_adjacency_matrix'.

**Returns**

xarray.Dataset
    A merged xarray Dataset containing the original data and the computed neighborhoods.

### `ds.nh.compute_neighborhoods_delaunay(include_center: bool=True, key_added: str='_neighborhoods', key_adjacency_matrix: str='_adjacency_matrix')`

Compute the neighborhoods of each cell based on a Delaunay triangulation.

This method identifies the neighbors for each cell and computes
the frequency of each cell type within these k neighbors.

**Parameters**

include_center : bool, optional
    Whether to include the center cell in the neighborhood. Default is True.
key_added : str, optional
    The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is '_neighborhoods'.
key_adjacency_matrix : str, optional
    The key under which the computed adjacency matrix will be stored in the resulting DataArray. Default is '_adjacency_matrix'.

**Returns**

xarray.Dataset
    A merged xarray Dataset containing the original data and the computed neighborhoods.

### `ds.nh.add_neighborhood_obs(features: Union[str, List[str]]=['degree', 'homophily', 'inter_label_connectivity', 'di ...])`

Adds neighborhood observations to the object by computing network features
from the adjacency matrix.
This method requires the `networkx` package to be installed. It checks if
the adjacency matrix is present in the object, constructs a graph from the
adjacency matrix, and computes network features.
**Parameters**

features : str or List[str], optional
    The network features to compute. Possible features are ['degree', 'closeness_centrality', 'betweenness_centrality', 'homophily', 'inter_label_connectivity', 'diversity_index'].
**Raises**

ImportError
    If the `networkx` package is not installed.
AssertionError
    If the adjacency matrix is not found in the object.
**Notes**

The adjacency matrix should be computed and stored in the object before
calling this method. You can compute the adjacency matrix using methods
from the `nh` module, such as `nh.compute_neighborhoods_radius()`.

### `ds.nh.compute_graph_features(features: Union[str, List[str]]=['num_nodes', 'num_edges', 'density', 'modularity', 'as ...])`

Compute various graph features from the adjacency matrix of the data.

**Parameters**

features : Union[str, List[str]], optional
    A single feature or a list of features to compute. Valid features are:
    "num_nodes", "num_edges", "density", "modularity", "assortativity".
    Default is ["num_nodes", "num_edges", "density", "modularity", "assortativity"].

**Returns**

dict
    A dictionary where keys are the names of the computed features and values are the corresponding feature values.

**Raises**

ImportError
    If the `networkx` package is not installed.
AssertionError
    If required layers (OBS, LA_PROPERTIES, ADJACENCY_MATRIX) are not found in the object.
    If no valid features are provided.
    If the LABELS feature is not found in the observation layer.

**Notes**

This method requires the `networkx` package to be installed. If you want to compute the modularity of the network,
you will also need to install the `python-louvain` package.
