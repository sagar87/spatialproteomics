from typing import Dict
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class ImageContainer():
    def __init__ (self, sprot_dict: Dict[str, xr.Dataset]):
        # assert that the input is a dictionary
        assert isinstance(sprot_dict, dict), "Input must be a dictionary"
        # assert that the dictionary is not empty
        assert len(sprot_dict) > 0, "Input dictionary must not be empty"
        # assert that the dictionary values are xarray datasets
        assert all([isinstance(v, xr.Dataset) for v in sprot_dict.values()]), "Dictionary values must be xarray Datasets"

        self.objects = sprot_dict
        
    def __repr__(self) -> str:
        return f"ImageContainer with {len(self.objects)} objects"
  
    def neighborhoods(self, k=5):
        # === NEIGHBORHOOD CONSTRUCTION ===
        neighborhood_df = []
        for id, sp_obj in self.objects.items():
            # TODO: add different methods for computing neighborhoods
            # TODO: these should also be added into the objects directly
            # computing the neighborhood for each object
            self.objects[id] = sp_obj.nh.compute_neighborhoods_radius()
            # adding the neighborhoods to the big neighborhood data frame
            neighborhood_df.append(self.objects[id].pp.get_layer_as_df('_neighborhoods'))
            
        # replacing nans with 0s (this means that the ct was never called in that specific sample)
        neighborhood_df = pd.concat(neighborhood_df).fillna(0)
        
        # === CLUSTERING ===
        # TODO: implement different clustering methods?
        # Running K-Means clustering
        clusterer = KMeans(n_clusters=k, random_state=0)
        clusterer.fit(neighborhood_df)
        kmeans_df = pd.DataFrame({f'neighborhood': clusterer.labels_})
        
        # === ADDING CLUSTERS TO OBJECTS ===
        # TODO: add the clusters into the objects
        for id, sp_obj in self.objects.items():
            # obtaining the number of cells contained in the object we are currently looking at
            num_cells = sp_obj.sizes['cells']
            # getting the labels for the number of cells
            tmp_df = kmeans_df[:num_cells]
            # removing those cells from the kmeans_df
            kmeans_df = kmeans_df[num_cells:]
            # storing the neighborhoods in the spatialproteomics objects
            # we can also set custom neighborhood colors here using .nh.set_neighborhood_colors()
            self.objects[id] = sp_obj.nh.add_neighborhoods_from_dataframe(tmp_df)
            
        # got to make sure the kmeans_df is empty, otherwise there was an error
        assert kmeans_df.empty, "There was an error in the clustering process. If you encounter this issue, please report it to the developers."
        
        return self.objects