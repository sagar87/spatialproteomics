# The image container

The image container wraps methods that needs to run on multiple images. For example, clustering neighborhoods is typically performed on all samples simultaneously.

## `sp.ImageContainer`

This class is used to store multiple SpatialProteomics objects and perform operations on them.

### `sp.ImageContainer.compute_neighborhoods(neighborhood_method: str='radius', radius=100, knn=10, k=5, overwrite: bool=False, seed: int=0)`

Compute neighborhoods for spatial proteomics objects using the specified method and perform clustering.

**Parameters**

neighborhood_method : str, optional
    The method to use for computing neighborhoods. Must be one of 'radius', 'knn', or 'delaunay'.
    Default is 'radius'.
radius : int, optional
    The radius to use for the 'radius' neighborhood method. Default is 100.
knn : int, optional
    The number of nearest neighbors to use for the 'knn' neighborhood method. Default is 10.
k : int, optional
    The number of clusters to form using K-Means clustering. Default is 5.
overwrite : bool, optional
    Whether to overwrite existing neighborhoods in the objects. Default is False.
seed : int, optional
    The random seed to use for K-Means clustering. Default is 0.
**Returns**

dict
    A dictionary of spatial proteomics objects with computed neighborhoods and clusters.
**Raises**

ValueError
    If neighborhoods are already present in the objects and `overwrite` is set to False.
    If `neighborhood_method` is not one of 'radius', 'knn', or 'delaunay'.
    If there is an error in the clustering process.

### `sp.ImageContainer.get_neighborhood_composition(standardize: bool=True, neighborhood_col: str='_neighborhoods') -> pd.DataFrame`

Get the composition of neighborhoods across all objects in the ImageContainer.

**Parameters**

standardize : bool, optional
    Whether to standardize the composition of neighborhoods. Default is True.
neighborhood_col : str, optional
    The column in the object's DataFrame that contains the neighborhood labels. Default is 'neighborhoods'.

**Returns**

pd.DataFrame
    A DataFrame containing the composition of neighborhoods across all objects in the ImageContainer.
