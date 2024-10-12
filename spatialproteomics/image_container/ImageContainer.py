from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans

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

    def __repr__(self) -> str:
        return f"ImageContainer with {len(self.objects)} objects"

    def compute_neighborhoods(
        self, neighborhood_method: str = "radius", radius=100, knn=10, k=5, overwrite: bool = False, seed: int = 0
    ):
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
                self.objects[id] = sp_obj.pp.drop_layers(
                    [Layers.NH_PROPERTIES, Layers.NEIGHBORHOODS, Layers.ADJACENCY_MATRIX], suppress_warnings=True
                )

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
                [f"Neighborhood {x}" for x in range(k)], colors
            )

        # got to make sure the kmeans_df is empty, otherwise there was a critical error
        assert (
            kmeans_df.empty
        ), "There was an error in the clustering process. If you encounter this issue, please report it to the developers."

        return self.objects
