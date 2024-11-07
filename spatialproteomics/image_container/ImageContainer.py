from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..constants import COLORS, Layers


class ImageContainer:
    """This class is used to store multiple SpatialProteomics objects and perform operations on them."""

    def __init__(self, sprot_dict: Dict[str, xr.Dataset]):
        # assert that the input is a dictionary
        assert isinstance(sprot_dict, dict), "Input must be a dictionary"
        # assert that the dictionary is not empty
        assert len(sprot_dict) > 0, "Input dictionary must not be empty"
        # assert that the dictionary values are xarray datasets
        assert all(
            [isinstance(v, xr.Dataset) for v in sprot_dict.values()]
        ), "Dictionary values must be xarray Datasets"

        self.objects = sprot_dict
        self.kmeans_df = None
        self.neighborhood_df = None

    def __repr__(self) -> str:
        return f"ImageContainer with {len(self.objects)} objects"

    def compute_neighborhoods(
        self, neighborhood_method: str = "radius", radius=100, knn=10, k=5, overwrite: bool = False, seed: int = 0
    ):
        """
        Compute neighborhoods for spatial proteomics objects using the specified method and perform clustering.

        Parameters
        ----------
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
        Returns
        -------
        dict
            A dictionary of spatial proteomics objects with computed neighborhoods and clusters.
        Raises
        ------
        ValueError
            If neighborhoods are already present in the objects and `overwrite` is set to False.
            If `neighborhood_method` is not one of 'radius', 'knn', or 'delaunay'.
            If there is an error in the clustering process.
        """
        assert neighborhood_method in [
            "radius",
            "knn",
            "delaunay",
        ], "Neighborhood method must be either 'radius', 'knn', or 'delaunay'."

        # before actually computing the neighborhoods, we need to check if the objects already have neighborhoods
        # if they do, we need to check if we are allowed to overwrite them
        for id, sp_obj in self.objects.items():
            # check if there are already some neighborhoods present in the objects
            if Layers.NEIGHBORHOODS in sp_obj and not overwrite:
                raise ValueError(
                    "Neighborhoods are already present in the objects. To overwrite them, set overwrite=True."
                )

        # === NEIGHBORHOOD CONSTRUCTION ===
        neighborhood_df = []
        for id, sp_obj in self.objects.items():
            # if neighborhoods are already present, we remove them from the objects
            if Layers.NEIGHBORHOODS in sp_obj:
                # checking which of these three layers are present, since not all might be (e. g. NH_PROPERTIES could be missing if no colors were set)
                present_layers = [
                    layer
                    for layer in [Layers.NH_PROPERTIES, Layers.NEIGHBORHOODS, Layers.ADJACENCY_MATRIX]
                    if layer in sp_obj
                ]
                self.objects[id] = sp_obj.pp.drop_layers(present_layers, suppress_warnings=True)

            # computing the neighborhood for each object
            if neighborhood_method == "radius":
                self.objects[id] = self.objects[id].nh.compute_neighborhoods_radius(radius=radius)
            elif neighborhood_method == "knn":
                self.objects[id] = self.objects[id].nh.compute_neighborhoods_knn(k=knn)
            elif neighborhood_method == "delaunay":
                self.objects[id] = self.objects[id].nh.compute_neighborhoods_delaunay()
            # adding the neighborhoods to the big neighborhood data frame
            neighborhood_df.append(self.objects[id].pp.get_layer_as_df("_neighborhoods"))

        # replacing nans with 0s (this means that the ct was never called in that specific sample)
        neighborhood_df = pd.concat(neighborhood_df).fillna(0)

        # === CLUSTERING ===
        # Running K-Means clustering
        # at some point, we could also add more clustering algorithms
        clusterer = KMeans(n_clusters=k, random_state=seed)
        clusterer.fit(neighborhood_df)
        kmeans_df = pd.DataFrame({"neighborhood": [f"Neighborhood {x}" for x in clusterer.labels_]})

        # storing the neighborhood and k-means dataframes in the ImageContainer
        self.neighborhood_df = neighborhood_df
        self.kmeans_df = kmeans_df

        # === ADDING CLUSTERS TO OBJECTS ===
        colors = np.random.choice(COLORS, size=k, replace=False)
        for id, sp_obj in self.objects.items():
            # obtaining the number of cells contained in the object we are currently looking at
            num_cells = sp_obj.sizes["cells"]
            # getting the labels for the number of cells
            tmp_df = kmeans_df[:num_cells]
            # removing those cells from the kmeans_df
            kmeans_df = kmeans_df[num_cells:]
            # storing the neighborhoods in the spatialproteomics objects
            # we can also set custom neighborhood colors here using .nh.set_neighborhood_colors()
            self.objects[id] = sp_obj.nh.add_neighborhoods_from_dataframe(tmp_df).nh.set_neighborhood_colors(
                [f"Neighborhood {x}" for x in range(k)], colors, suppress_warnings=True
            )

        # got to make sure the kmeans_df is empty, otherwise there was a critical error
        assert (
            kmeans_df.empty
        ), "There was an error in the clustering process. If you encounter this issue, please report it to the developers."

        # returning the objects in the form of a dictionary
        return self.objects

    def get_neighborhood_composition(self, standardize: bool = True) -> pd.DataFrame:
        """
        Get the composition of neighborhoods across all objects in the ImageContainer.

        Parameters
        ----------
        standardize : bool, optional
            Whether to standardize the composition of neighborhoods. Default is True.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the composition of neighborhoods across all objects in the ImageContainer.
        """
        assert (
            self.neighborhood_df is not None
        ), "Neighborhoods have not been computed yet. Please call compute_neighborhoods() first."

        # computing the composition of neighborhoods (grouping by the neighborhood labels and summing them up)
        neighborhood_composition = self.neighborhood_df
        neighborhood_composition["neighborhood"] = self.kmeans_df["neighborhood"].values

        neighborhood_composition = neighborhood_composition.groupby("neighborhood").mean()

        if standardize:
            scaler = StandardScaler()
            neighborhood_composition = pd.DataFrame(
                scaler.fit_transform(neighborhood_composition),
                index=neighborhood_composition.index,
                columns=neighborhood_composition.columns,
            )

        return neighborhood_composition
