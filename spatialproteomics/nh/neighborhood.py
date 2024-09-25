from typing import List, Union

import numpy as np
import xarray as xr

from ..base_logger import logger
from ..constants import COLORS, Dims, Features, Layers, Props
from .utils import _construct_neighborhood_df_radius, _construct_neighborhood_df_knn, _construct_neighborhood_df_delaunay


@xr.register_dataset_accessor("nh")
class NeighborhoodAccessor:
    """Adds functions for cellular neighborhoods."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __contains__(self, key):
        if Layers.NH_PROPERTIES not in self._obj:
            return False

        neighborhood_dict = self._obj.nh._neighborhood_to_dict(Props.NAME)
        return key in neighborhood_dict.keys() or key in neighborhood_dict.values()

    def __getitem__(self, indices):
        # type checking
        if isinstance(indices, float):
            raise TypeError("Neighborhood indices must be valid integers, str, slices, List[int] or List[str].")

        if isinstance(indices, int):
            if indices not in self._obj.coords[Dims.NEIGHBORHOODS].values:
                raise ValueError(
                    f"Neighborhood type {indices} not found. Neighborhoods available for integer indexing are: {self._obj.coords[Dims.NEIGHBORHOODS].values}."
                )

            sel = [indices]

        if isinstance(indices, str):
            neighborhood_dict = self._obj.nh._neighborhood_to_dict(Props.NAME, reverse=True)

            if indices not in neighborhood_dict:
                raise ValueError(f"Neighborhood type {indices} not found.")

            sel = [neighborhood_dict[indices]]

        if isinstance(indices, slice):
            l_start = indices.start if indices.start is not None else 1
            l_stop = indices.stop if indices.stop is not None else self._obj.sizes[Dims.NEIGHBORHOODS]
            sel = [i for i in range(l_start, l_stop)]

        if isinstance(indices, (list, tuple)):
            if not all([isinstance(i, (str, int)) for i in indices]):
                raise TypeError("Neighborhood indices must be valid integers, str, slices, List[int] or List[str].")

            sel = []
            for i in indices:
                if isinstance(i, str):
                    neighborhood_dict = self._obj.nh._neighborhood_to_dict(Props.NAME, reverse=True)

                    if i not in neighborhood_dict:
                        raise ValueError(f"Neighborhood {i} not found.")

                    sel.append(neighborhood_dict[i])

                if isinstance(i, int):
                    if i not in self._obj.coords[Dims.NEIGHBORHOODS].values:
                        raise ValueError(f"Neighborhood {i} not found.")

                    sel.append(i)

        cells = self._obj.nh._filter_cells_by_neighborhood(sel)

        obj = self._obj.sel({Dims.NEIGHBORHOODS: sel, Dims.CELLS: cells})

        # removing all cells from the segmentation mask that are not in the cells array
        # we need the copy() here, as this will otherwise modify the original self._obj due to the array referencing it
        segmentation = self._obj[Layers.SEGMENTATION].values.copy()
        mask = np.isin(segmentation, cells)
        segmentation[~mask] = 0
        # removing the old segmentation
        obj = obj.drop_vars(Layers.SEGMENTATION)
        # adding the new segmentation
        obj = obj.pp.add_segmentation(segmentation, reindex=False)

        return obj

    def deselect(self, indices):
        """
        Deselect specific neighborhood indices from the data object.

        This method deselects specific neighborhood indices from the data object, effectively removing them from the selection.
        The deselection can be performed using slices, lists, tuples, or individual integers.

        Parameters
        ----------
        indices : slice, list, tuple, or int
            The neighborhood indices to be deselected. Can be a slice, list, tuple, or an individual integer.

        Returns
        -------
        any
            The updated data object with the deselected neighborhood indices.

        Notes
        -----
        - The function uses 'indices' to specify which neighborhoods to deselect.
        - 'indices' can be provided as slices, lists, tuples, or an integer.
        - The function then updates the data object to remove the deselected neighborhood indices.
        """
        if isinstance(indices, slice):
            l_start = indices.start if indices.start is not None else 1
            l_stop = indices.stop if indices.stop is not None else self._obj.sizes[Dims.NEIGHBORHOODS]
            sel = [i for i in range(l_start, l_stop)]
        elif isinstance(indices, list):
            assert all(
                [isinstance(i, (int, str)) for i in indices]
            ), "All neighborhood indices must be integers or strings."
            if all([isinstance(i, int) for i in indices]):
                sel = indices
            else:
                neighborhood_dict = self._obj.nh._neighborhood_to_dict(Props.NAME, reverse=True)
                for idx in indices:
                    if idx not in neighborhood_dict:
                        raise ValueError(f"Neighborhood {indices} not found.")
                sel = [neighborhood_dict[idx] for idx in indices]
        elif isinstance(indices, tuple):
            indices = list(indices)
            all_int = all([type(i) is int for i in indices])
            assert all_int, "All neighborhood indices must be integers."
            sel = indices
        elif isinstance(indices, str):
            neighborhood_dict = self._obj.nh._neighborhood_to_dict(Props.NAME, reverse=True)
            if indices not in neighborhood_dict:
                raise ValueError(f"Neighborhood {indices} not found.")
            sel = [neighborhood_dict[indices]]
        else:
            assert type(indices) is int, "Neighborhood must be provided as slices, lists, tuple or int."
            sel = [indices]

        inv_sel = [i for i in self._obj.coords[Dims.NEIGHBORHOODS] if i not in sel]

        cells = self._obj.nh._filter_cells_by_neighborhood(inv_sel)

        obj = self._obj.sel({Dims.NEIGHBORHOODS: inv_sel, Dims.CELLS: cells})

        # removing all cells from the segmentation mask that are not in the cells array
        segmentation = obj[Layers.SEGMENTATION].values.copy()
        mask = np.isin(segmentation, cells)
        segmentation[~mask] = 0
        # removing the old segmentation
        obj = obj.drop_vars(Layers.SEGMENTATION)
        # adding the new segmentation
        obj = obj.pp.add_segmentation(segmentation)

        return obj

    def _filter_cells_by_neighborhood(self, items: Union[int, List[int]]):
        """
        Filter cells by neighborhood.

        Parameters
        ----------
        items : int or List[int]
            The neighborhood(s) to filter cells by. If an integer is provided, only cells with that neighborhood will be returned.
            If a list of integers is provided, cells with any of the neighborhoods in the list will be returned.

        Returns
        -------
        numpy.ndarray
            An array containing the selected cells.
        """
        if type(items) is int:
            items = [items]

        cells = self._obj[Layers.OBS].loc[:, Features.NEIGHBORHOODS].values.copy()
        cells_bool = np.isin(cells, items)
        cells_sel = self._obj.coords[Dims.CELLS][cells_bool].values

        return cells_sel

    def _neighborhood_name_to_id(self, neighborhood):
        """
        Convert a neighborhood name to its corresponding ID.

        Parameters
        ----------
        label : str
            The name of the neighborhood to convert.

        Returns
        -------
        int
            The ID corresponding to the given neighborhood name.

        Raises
        ------
        ValueError
            If the given neighborhood name is not found in the neighborhood names dictionary.
        """
        neighborhood_names_reverse = self._obj.nh._neighborhood_to_dict(Props.NAME, reverse=True)
        if neighborhood not in neighborhood_names_reverse:
            raise ValueError(f"Neighborhood {neighborhood} not found.")

        return neighborhood_names_reverse[neighborhood]

    def _cells_to_neighborhood(self, relabel: bool = False) -> dict:
        """
        Returns a dictionary that maps each neighborhood to a list of cells.

        Parameters
        ----------
        relabel : bool, optional
            If True, relabels the dictionary keys to consecutive integers starting from 1.
            Default is False.

        Returns
        -------
        dict
            A dictionary that maps each neighborhood to a list of cells. The keys are neighborhood names,
            and the values are lists of cell indices.
        """
        neighborhood_dict = {
            neighborhood.item(): self._obj.nh._filter_cells_by_neighborhood(neighborhood.item())
            for neighborhood in self._obj.coords[Dims.NEIGHBORHOODS]
        }

        if relabel:
            return self._obj.nh._relabel_dict(neighborhood_dict)

        return neighborhood_dict

    def add_properties(self, array: Union[np.ndarray, list], prop: str = Features.NEIGHBORHOODS) -> xr.Dataset:
        """
        Adds neighborhood properties to the image container.

        Parameters
        ----------
        array : Union[np.ndarray, list]
            An array or list of properties to be added to the image container.
        prop : str, optional
            The name of the property. Default is Features.NEIGHBORHOODS.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The updated image container with added properties or the properties as a separate xarray.DataArray.
        """
        unique_neighborhoods = np.unique(self._obj[Layers.OBS].sel({Dims.FEATURES: Features.NEIGHBORHOODS}))

        if type(array) is list:
            array = np.array(array)

        da = xr.DataArray(
            array.reshape(-1, 1),
            coords=[unique_neighborhoods.astype(int), [prop]],
            dims=[Dims.NEIGHBORHOODS, Dims.NH_PROPS],
            name=Layers.NH_PROPERTIES,
        )

        if Layers.NH_PROPERTIES in self._obj:
            da = xr.concat(
                [self._obj[Layers.NH_PROPERTIES], da],
                dim=Dims.NH_PROPS,
            )

        return xr.merge([da, self._obj])

    def add_neighborhoods_from_dataframe(
        self, neighborhoods: Union[np.ndarray, list], colors: Union[list, None] = None, names: Union[list, None] = None
    ) -> xr.Dataset:
        # check if properties are already present
        assert (
            Layers.NH_PROPERTIES not in self._obj
        ), f"Already found neighborhood properties in the object. Please remove them with pp.drop_layers('{Layers.NH_PROPERTIES}') first."

        # check that the labels have the same length as the cells
        assert len(neighborhoods) == len(
            self._obj.coords[Dims.CELLS]
        ), "The number of neighborhoods does not match the number of cells."

        if type(neighborhoods) is list:
            neighborhoods = neighborhoods.to_numpy()

        neighborhoods = neighborhoods.squeeze()
        unique_neighborhoods = np.unique(neighborhoods)

        # adding the neighborhoods into obs
        obj = self._obj.copy()
        obj = obj.pp.add_feature(Features.NEIGHBORHOODS, neighborhoods)

        # adding colors to the neighborhoods
        if colors is not None:
            assert len(colors) == len(
                unique_neighborhoods
            ), "Colors does not have the same length as there are neighborhoods."
        else:
            colors = np.random.choice(COLORS, size=len(unique_neighborhoods), replace=False)

        # adding a lookup table for the neighborhood colors
        obj = obj.nh.add_properties(colors, Props.COLOR)

        # adding names
        if names is not None:
            assert len(names) == len(unique_neighborhoods), "Names do not match the number of neighborhoods."
        else:
            names = [f"Neighborhood {i}" for i in unique_neighborhoods]

        obj = obj.nh.add_properties(names, Props.NAME)

        return obj

    def _neighborhood_to_dict(self, prop: str, reverse: bool = False, relabel: bool = False) -> dict:
        """
        Returns a dictionary that maps each neighborhood to a property.

        Parameters
        ----------
        prop : str
            The property to map to the labels.
        reverse : bool, optional
            If True, the dictionary will be reversed. Default is False.
        relabel : bool, optional
            If True, relabels the dictionary keys to consecutive integers starting from 1.
            Default is False.

        Returns
        -------
        label_dict : dict
            A dictionary that maps each label to a property.
        """
        neighborhood_layer = self._obj[Layers.NH_PROPERTIES]
        neighborhoods = self._obj.coords[Dims.NEIGHBORHOODS]

        neighborhood_dict = {}

        for neighborhood in neighborhoods:
            current_row = neighborhood_layer.loc[neighborhood, prop]
            neighborhood_dict[neighborhood.values.item()] = current_row.values.item()

        if relabel:
            return self._obj.nh._relabel_dict(neighborhood_dict)

        if reverse:
            neighborhood_dict = {v: k for k, v in neighborhood_dict.items()}

        return neighborhood_dict

    def _relabel_dict(self, dictionary: dict):
        unique_keys = sorted(set(dictionary.keys()))  # Get unique keys and sort them
        relabel_map = {
            key: idx + 1 for idx, key in enumerate(unique_keys)
        }  # Create a mapping to consecutive numbers starting from 1
        return {relabel_map[k]: v for k, v in dictionary.items()}  # Apply the relabeling

    def set_neighborhood_colors(self, neighborhoods: Union[str, List[str]], colors: Union[str, List[str]]):
        """
        Set the color of a specific neighborhood.

        This method sets the 'color' of a specific neighborhood identified by the 'neighborhood'.
        The 'neighborhood' can be either a neighborhood ID or the name of the neighborhood.

        Parameters
        ----------
        label : int or str
            The ID or name of the neighborhood whose color will be updated.
        color : any
            The new color to be assigned to the specified neighborhood.

        Returns
        -------
        None

        Notes
        -----
        - The function converts the 'neighborhood' from its name to the corresponding ID for internal processing.
        - It updates the color of the neighborhood in the data object to the new 'color'.
        """
        if isinstance(neighborhoods, str):
            neighborhoods = [neighborhoods]
        if isinstance(colors, str):
            colors = [colors]

        # checking that there are as many colors as labels
        assert len(neighborhoods) == len(colors), "The number of neighborhoods and colors must be the same."

        # checking that a neighborhood layer is already present
        assert (
            Layers.NH_PROPERTIES in self._obj
        ), "No neighborhoods layer found in the data object. Please add neighborhoods before setting colors, e. g. by using nh.compute_neighborhoods_radius()."

        # obtaining the current properties
        props_layer = self._obj.coords[Dims.NH_PROPS].values.tolist()
        neighborhoods_layer = self._obj.coords[Dims.NEIGHBORHOODS].values.tolist()
        array = self._obj[Layers.NH_PROPERTIES].values.copy()

        for neighborhood, color in zip(neighborhoods, colors):
            # if the neighborhoods does not exist in the object, a warning is thrown and we continue
            if neighborhood not in self._obj.nh:
                logger.warning(f"Neighborhood {neighborhood} not found in the data object. Skipping.")
                continue

            # getting the id for the label
            neighborhood = self._obj.nh._neighborhood_name_to_id(neighborhood)

            # setting the new color for the given label
            array[neighborhoods_layer.index(neighborhood), props_layer.index(Props.COLOR)] = color

        da = xr.DataArray(
            array,
            coords=[neighborhoods_layer, props_layer],
            dims=[Dims.NEIGHBORHOODS, Dims.NH_PROPS],
            name=Layers.NH_PROPERTIES,
        )

        return xr.merge([self._obj.drop_vars(Layers.NH_PROPERTIES), da])

    def set_neighborhood_name(self, neighborhood, name):
        """
        Set the name of a specific neighborhood.

        This method sets the 'name' of a specific neighborhood identified by the 'neighborhood'.
        The 'neighborhood' can be either a neighborhood ID or the name of the neighborhood.

        Parameters
        ----------
        label : int or str
            The ID or name of the neighborhood whose name will be updated.
        name : str
            The new name to be assigned to the specified neighborhood.

        Returns
        -------
        None

        Notes
        -----
        - The function converts the 'neighborhood' from its name to the corresponding ID for internal processing.
        - It updates the name of the neighborhood in the data object to the new 'name'.
        """
        # checking that a neighborhood layer is already present
        assert Layers.NH_PROPERTIES in self._obj, "No neighborhood layer found in the data object."
        # checking if the old neighborhood exists
        assert (
            neighborhood in self._obj.nh
        ), f"Neighborhood {neighborhood} not found. Existing cell types: {self._obj.nh}"
        # checking if the new label already exists
        assert name not in self._obj[Layers.NH_PROPERTIES].sel(
            {Dims.NH_PROPS: Props.NAME}
        ), f"Neighborhood name {neighborhood} already exists."

        # getting the original neighborhood properties
        property_layer = self._obj[Layers.NH_PROPERTIES].copy()

        if isinstance(neighborhood, str):
            neighborhood = self._obj.nh._neighborhood_name_to_id(neighborhood)

        property_layer.loc[neighborhood, Props.NAME] = name

        # removing the old property layer
        obj = self._obj.pp.drop_layers(Layers.NH_PROPERTIES)

        # adding the new property layer
        return xr.merge([property_layer, obj])

    def compute_neighborhoods_radius(self, radius=100, key_added: str = Layers.NEIGHBORHOODS):
        """
        Compute the neighborhoods of each cell based on a specified radius.

        This method defines a radius around each cell and identifies all cells within this radius.
        It then examines the predicted cell types of these cells, including the center cell itself,
        and computes the frequency of each cell type.

        Parameters
        ----------
        radius : int, optional
            The radius around each cell to define the neighborhood. Default is 100.
        key_added : str, optional
            The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is Layers.NEIGHBORHOODS.

        Returns
        -------
        xarray.Dataset
            A merged xarray Dataset containing the original data and the computed neighborhoods.
        """

        # this method computes the neighborhoods of each cell based on some radius around that cell
        # it works by defining a radius around each cell and finding all cells that are within this radius
        # then, it looks at the predicted cell types of these cells (including the center cell itself) and computes the frequency of each cell type

        assert radius > 0, "Radius must be greater than 0."
        assert Layers.OBS in self._obj, "No observations found in the object."
        assert Features.LABELS in self._obj.coords[Dims.FEATURES].values, "No cell type labels found in the object."

        # here we use the numeric labels in order to keep them synchronized with the rest of the object
        neighborhood_df = _construct_neighborhood_df_radius(
            self._obj.pp.get_layer_as_df(celltypes_to_str=False),
            cell_types=self._obj.coords[Dims.LABELS].values,
            x=Features.X,
            y=Features.Y,
            label_col=Features.LABELS,
            radius=radius,
        )

        # putting the df into a data array
        da = xr.DataArray(
            neighborhood_df.values,
            coords=[neighborhood_df.index, neighborhood_df.columns],
            dims=[Dims.CELLS, Dims.LABELS],
            name=key_added,
        )

        return xr.merge([self._obj, da])


    def compute_neighborhoods_knn(self, k=10, key_added: str = Layers.NEIGHBORHOODS):
        """
        Compute the neighborhoods of each cell based on k-nearest neighbors.

        This method identifies the k-nearest neighbors for each cell and computes
        the frequency of each cell type within these k neighbors.

        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors to consider. Default is 10.
        key_added : str, optional
            The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is Layers.NEIGHBORHOODS.

        Returns
        -------
        xarray.Dataset
            A merged xarray Dataset containing the original data and the computed neighborhoods.
        """

        assert k > 0, "k must be greater than 0."
        assert Layers.OBS in self._obj, "No observations found in the object."
        assert Features.LABELS in self._obj.coords[Dims.FEATURES].values, "No cell type labels found in the object."
            
        # here we use the numeric labels in order to keep them synchronized with the rest of the object
        neighborhood_df = _construct_neighborhood_df_knn(
            self._obj.pp.get_layer_as_df(celltypes_to_str=False),
            cell_types=self._obj.coords[Dims.LABELS].values,
            x=Features.X,
            y=Features.Y,
            label_col=Features.LABELS,
            k=k,
        )

        # Convert the DataFrame to an xarray DataArray
        # putting the df into a data array
        da = xr.DataArray(
            neighborhood_df.values,
            coords=[neighborhood_df.index, neighborhood_df.columns],
            dims=[Dims.CELLS, Dims.LABELS],
            name=key_added,
        )

        return xr.merge([self._obj, da])
    
    
    def compute_neighborhoods_delaunay(self, key_added: str = Layers.NEIGHBORHOODS):
        """
        Compute the neighborhoods of each cell based on a Delaunay triangulation.

        This method identifies the neighbors for each cell and computes
        the frequency of each cell type within these k neighbors.

        Parameters
        ----------
        key_added : str, optional
            The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is Layers.NEIGHBORHOODS.

        Returns
        -------
        xarray.Dataset
            A merged xarray Dataset containing the original data and the computed neighborhoods.
        """
        assert Layers.OBS in self._obj, "No observations found in the object."
        assert Features.LABELS in self._obj.coords[Dims.FEATURES].values, "No cell type labels found in the object."
            
        # here we use the numeric labels in order to keep them synchronized with the rest of the object
        neighborhood_df = _construct_neighborhood_df_delaunay(
            self._obj.pp.get_layer_as_df(celltypes_to_str=False),
            cell_types=self._obj.coords[Dims.LABELS].values,
            x=Features.X,
            y=Features.Y,
            label_col=Features.LABELS,
        )

        # Convert the DataFrame to an xarray DataArray
        # putting the df into a data array
        da = xr.DataArray(
            neighborhood_df.values,
            coords=[neighborhood_df.index, neighborhood_df.columns],
            dims=[Dims.CELLS, Dims.LABELS],
            name=key_added,
        )

        return xr.merge([self._obj, da])