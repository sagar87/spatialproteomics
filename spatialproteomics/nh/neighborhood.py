from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..base_logger import logger
from ..constants import COLORS, Dims, Features, Layers, Props
from .utils import (
    _compute_global_network_features,
    _compute_network_features,
    _construct_neighborhood_df_delaunay,
    _construct_neighborhood_df_knn,
    _construct_neighborhood_df_radius,
    _format_neighborhoods,
)


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
        # checking if the user provided dict_values or dict_keys and turns them into a list if that is the case
        if type(indices) is {}.keys().__class__ or type(indices) is {}.values().__class__:
            indices = list(indices)

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

        # ensuring that cells and cells_2 are synchronized
        if Dims.CELLS_2 in obj.coords:
            obj = obj.sel({Dims.CELLS_2: cells})

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
        xr.Dataset
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
        xr.Dataset
            The updated image container with added properties.
        """
        unique_neighborhoods = np.unique(self._obj[Layers.OBS].sel({Dims.FEATURES: Features.NEIGHBORHOODS}))

        if type(array) is list:
            array = np.array(array)

        if prop == Features.NEIGHBORHOODS:
            unique_neighborhoods = np.unique(_format_neighborhoods(array))

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
        self,
        df: pd.DataFrame,
        neighborhood_col: str = Features.NEIGHBORHOODS,
        colors: Union[list, None] = None,
        names: Union[list, None] = None,
    ) -> xr.Dataset:
        """
        Add neighborhoods to the dataset from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing neighborhood information.
        neighborhood_col : str, optional
            Column name in the DataFrame that contains neighborhood labels, by default '_neighborhoods'.
        colors : list or None, optional
            List of colors for the neighborhoods, by default None. If None, random colors will be assigned.
        names : list or None, optional
            List of names for the neighborhoods, by default None. If None, default names will be assigned.

        Returns
        -------
        xr.Dataset
            Updated dataset with neighborhood information added.

        Raises
        ------
        AssertionError
            If neighborhood properties are already present in the object.
            If the number of neighborhoods does not match the number of cells.
            If the specified column does not exist in the DataFrame.
            If neighborhoods contain negative values.
            If the length of colors does not match the number of unique neighborhoods.
            If the length of names does not match the number of unique neighborhoods.
        """
        # check if properties are already present
        assert (
            Layers.NH_PROPERTIES not in self._obj
        ), f"Already found neighborhood properties in the object. Please remove them with pp.drop_layers('{Layers.NH_PROPERTIES}') first."

        # check that the labels have the same length as the cells
        assert df.shape[0] == len(
            self._obj.coords[Dims.CELLS]
        ), "The number of neighborhoods does not match the number of cells."

        # check that the column exists in the data frame
        assert neighborhood_col in df.columns, f"Column {neighborhood_col} not found in the data frame."

        neighborhoods = df.loc[:, neighborhood_col].to_numpy().squeeze()
        unique_neighborhoods = np.unique(neighborhoods)

        if np.all([isinstance(i, str) for i in neighborhoods]):
            unique_neighborhoods = np.unique(neighborhoods)

            # converting the neighborhood labels to numeric values
            neighborhood_to_num = dict(zip(unique_neighborhoods, range(1, len(unique_neighborhoods) + 1)))

            neighborhoods = np.array([neighborhood_to_num[neighborhood] for neighborhood in neighborhoods])
            names = [k for k, v in sorted(neighborhood_to_num.items(), key=lambda x: x[1])]

        assert ~np.all(neighborhoods < 0), "Neighborhoods must be >= 0."

        formated_neighborhoods = _format_neighborhoods(neighborhoods)
        unique_neighborhoods = np.unique(formated_neighborhoods)

        # adding the neighborhoods into obs
        obj = self._obj.copy()
        obj = obj.pp.add_feature(Features.NEIGHBORHOODS, neighborhoods)

        # adding colors to the neighborhoods
        if colors is not None:
            assert len(colors) == len(
                unique_neighborhoods
            ), f"Colors does not have the same length as there are neighborhoods. Got {len(colors)} colors for {len(unique_neighborhoods)} neighborhoods."
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

        # finally converting the properties to strings, because str and np.str cannot be mixed
        obj[Layers.NH_PROPERTIES] = obj[Layers.NH_PROPERTIES].astype(str)

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

    def set_neighborhood_colors(
        self, neighborhoods: Union[str, List[str]], colors: Union[str, List[str]], suppress_warnings: bool = False
    ):
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
        suppress_warnings : bool, optional
            Whether to suppress warnings. Default is False.

        Returns
        -------
        xr.Dataset

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
                if not suppress_warnings:
                    logger.warning(f"Neighborhood {neighborhood} not found in the data object. Skipping.")
                continue

            # getting the id for the label (if the input was not numeric already)
            if isinstance(neighborhood, str):
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

    def set_neighborhood_name(self, neighborhoods: Union[int, str, List], names: Union[str, List]):
        """
        Set the name of one or more neighborhoods.

        This method updates the 'name' of specific neighborhoods identified by the 'neighborhood'.
        The 'neighborhoods' can be a single neighborhood ID or name, or a list of IDs/names. Similarly,
        the 'names' parameter can be a single string or a list of strings.

        Parameters
        ----------
        neighborhoods : int, str, or list
            The ID(s) or name(s) of the neighborhoods whose names will be updated.
        names : str or list
            The new name(s) to be assigned to the specified neighborhoods.

        Returns
        -------
        xr.Dataset

        Notes
        -----
        - When both parameters are lists, their lengths must match.
        - The function converts each 'neighborhood' from its name to the corresponding ID for internal processing.
        - It updates the name(s) of the neighborhood(s) in the data object to the new 'name(s)'.
        """
        # Ensure the neighborhood layer exists
        assert Layers.NH_PROPERTIES in self._obj, "No neighborhood layer found in the data object."

        # checking if the user provided dict_values or dict_keys and turns them into a list if that is the case
        if type(neighborhoods) is {}.keys().__class__ or type(neighborhoods) is {}.values().__class__:
            neighborhoods = list(neighborhoods)
        if type(names) is {}.keys().__class__ or type(names) is {}.values().__class__:
            names = list(names)

        # Handle single inputs by converting them into lists for uniform processing
        if not isinstance(neighborhoods, list):
            neighborhoods = [neighborhoods]
        if not isinstance(names, list):
            names = [names]

        # Ensure the lengths of neighborhoods and names match
        assert len(neighborhoods) == len(
            names
        ), f"Mismatch in lengths: {len(neighborhoods)} neighborhoods and {len(names)} names provided."

        # ensure that the neighborhoods are provided as either strings or integers, but not mixed
        assert all([isinstance(n, str) for n in neighborhoods]) or all(
            [isinstance(n, int) for n in neighborhoods]
        ), "Neighborhoods must be provided as either strings or integers, but not mixed."

        # ensure that the names are provided as strings
        assert all([isinstance(n, str) for n in names]), "Names must be provided as strings."

        # ensure that there are no duplicates in the names
        assert len(names) == len(set(names)), "Names must be unique."

        # Check that all neighborhoods exist
        invalid_neighborhoods = [n for n in neighborhoods if n not in self._obj.nh]

        # if the neighborhoods are provided as strings
        if all([isinstance(n, str) for n in neighborhoods]):
            existing_names = self._obj[Layers.NH_PROPERTIES].sel({Dims.NH_PROPS: Props.NAME}).values
            assert not invalid_neighborhoods, (
                f"Neighborhood(s) {invalid_neighborhoods} not found. " f"Existing neighborhoods: {existing_names}"
            )

        # if they are provided as integers
        if all([isinstance(n, int) for n in neighborhoods]):
            existing_names = self._obj.coords["neighborhoods"].values
            assert not invalid_neighborhoods, (
                f"Neighborhood(s) {invalid_neighborhoods} not found. " f"Existing neighborhoods: {existing_names}"
            )

        # Check that all new names are unique and do not already exist
        existing_names = set(self._obj[Layers.NH_PROPERTIES].sel({Dims.NH_PROPS: Props.NAME}).values)
        duplicate_names = [n for n in names if n in existing_names]
        assert not duplicate_names, f"Neighborhood name(s) {duplicate_names} already exist in the data object."

        # Retrieve the original neighborhood properties
        property_layer = self._obj[Layers.NH_PROPERTIES].copy()

        for n, new_name in zip(neighborhoods, names):
            # Convert neighborhood name to ID if necessary
            if isinstance(n, str):
                n = self._obj.nh._neighborhood_name_to_id(n)

            # Update the name
            property_layer.loc[n, Props.NAME] = new_name

        # Remove the old property layer
        obj = self._obj.pp.drop_layers(Layers.NH_PROPERTIES, drop_obs=False)

        # Add the updated property layer
        return xr.merge([property_layer, obj])

    def compute_neighborhoods_radius(
        self,
        radius: int = 100,
        include_center: bool = True,
        key_added: str = Layers.NEIGHBORHOODS,
        key_adjacency_matrix: str = Layers.ADJACENCY_MATRIX,
    ):
        """
        Compute the neighborhoods of each cell based on a specified radius.

        This method defines a radius around each cell and identifies all cells within this radius.
        It then examines the predicted cell types of these cells, including the center cell itself,
        and computes the frequency of each cell type.

        Parameters
        ----------
        radius : int, optional
            The radius around each cell to define the neighborhood (in pixels). Default is 100.
        include_center : bool, optional
            Whether to include the center cell in the neighborhood. Default is True.
        key_added : str, optional
            The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is '_neighborhoods'.
        key_adjacency_matrix : str, optional
            The key under which the computed adjacency matrix will be stored in the resulting DataArray. Default is '_adjacency_matrix'.

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
        assert (
            Layers.ADJACENCY_MATRIX not in self._obj
        ), "Adjacency matrix already found in the object. Please remove it first with pp.drop_layers('_adjacency_matrix')."

        # here we use the numeric labels in order to keep them synchronized with the rest of the object
        neighborhood_df, adjacency_matrix = _construct_neighborhood_df_radius(
            self._obj.pp.get_layer_as_df(celltypes_to_str=False),
            cell_types=self._obj.coords[Dims.LABELS].values,
            x=Features.X,
            y=Features.Y,
            label_col=Features.LABELS,
            radius=radius,
            include_center=include_center,
        )

        # putting the df into a data array
        da = xr.DataArray(
            neighborhood_df.values,
            coords=[neighborhood_df.index, neighborhood_df.columns],
            dims=[Dims.CELLS, Dims.LABELS],
            name=key_added,
        )

        obj = xr.merge([self._obj, da])

        # adding the adjacency matrix to the object
        cells = obj.coords[Dims.CELLS].values
        da = xr.DataArray(
            adjacency_matrix,
            coords=[cells, cells],
            # xarray does not support duplicate dimension names, hence we need to introduce a second cell variable here
            dims=[Dims.CELLS, Dims.CELLS_2],
            name=key_adjacency_matrix,
        )

        return xr.merge([obj, da])

    def compute_neighborhoods_knn(
        self,
        k=10,
        include_center: bool = True,
        key_added: str = Layers.NEIGHBORHOODS,
        key_adjacency_matrix: str = Layers.ADJACENCY_MATRIX,
    ):
        """
        Compute the neighborhoods of each cell based on k-nearest neighbors.

        This method identifies the k-nearest neighbors for each cell and computes
        the frequency of each cell type within these k neighbors.

        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors to consider. Default is 10.
        include_center : bool, optional
            Whether to include the center cell in the neighborhood. Default is True.
        key_added : str, optional
            The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is '_neighborhoods'.
        key_adjacency_matrix : str, optional
            The key under which the computed adjacency matrix will be stored in the resulting DataArray. Default is '_adjacency_matrix'.

        Returns
        -------
        xarray.Dataset
            A merged xarray Dataset containing the original data and the computed neighborhoods.
        """

        assert k > 0, "K must be greater than 0."
        assert Layers.OBS in self._obj, "No observations found in the object."
        assert Features.LABELS in self._obj.coords[Dims.FEATURES].values, "No cell type labels found in the object."
        assert (
            Layers.ADJACENCY_MATRIX not in self._obj
        ), "Adjacency matrix already found in the object. Please remove it first with pp.drop_layers('_adjacency_matrix')."

        # here we use the numeric labels in order to keep them synchronized with the rest of the object
        neighborhood_df, adjacency_matrix = _construct_neighborhood_df_knn(
            self._obj.pp.get_layer_as_df(celltypes_to_str=False),
            cell_types=self._obj.coords[Dims.LABELS].values,
            x=Features.X,
            y=Features.Y,
            label_col=Features.LABELS,
            k=k,
            include_center=include_center,
        )

        # Convert the DataFrame to an xarray DataArray
        # putting the df into a data array
        da = xr.DataArray(
            neighborhood_df.values,
            coords=[neighborhood_df.index, neighborhood_df.columns],
            dims=[Dims.CELLS, Dims.LABELS],
            name=key_added,
        )

        obj = xr.merge([self._obj, da])

        # adding the adjacency matrix to the object
        cells = obj.coords[Dims.CELLS].values
        da = xr.DataArray(
            adjacency_matrix,
            coords=[cells, cells],
            # xarray does not support duplicate dimension names, hence we need to introduce a second cell variable here
            dims=[Dims.CELLS, Dims.CELLS_2],
            name=key_adjacency_matrix,
        )

        return xr.merge([obj, da])

    def compute_neighborhoods_delaunay(
        self,
        include_center: bool = True,
        key_added: str = Layers.NEIGHBORHOODS,
        key_adjacency_matrix: str = Layers.ADJACENCY_MATRIX,
    ):
        """
        Compute the neighborhoods of each cell based on a Delaunay triangulation.

        This method identifies the neighbors for each cell and computes
        the frequency of each cell type within these k neighbors.

        Parameters
        ----------
        include_center : bool, optional
            Whether to include the center cell in the neighborhood. Default is True.
        key_added : str, optional
            The key under which the computed neighborhoods will be stored in the resulting DataArray. Default is '_neighborhoods'.
        key_adjacency_matrix : str, optional
            The key under which the computed adjacency matrix will be stored in the resulting DataArray. Default is '_adjacency_matrix'.

        Returns
        -------
        xarray.Dataset
            A merged xarray Dataset containing the original data and the computed neighborhoods.
        """
        assert Layers.OBS in self._obj, "No observations found in the object."
        assert Features.LABELS in self._obj.coords[Dims.FEATURES].values, "No cell type labels found in the object."
        assert (
            Layers.ADJACENCY_MATRIX not in self._obj
        ), "Adjacency matrix already found in the object. Please remove it first with pp.drop_layers('_adjacency_matrix')."

        # here we use the numeric labels in order to keep them synchronized with the rest of the object
        neighborhood_df, adjacency_matrix = _construct_neighborhood_df_delaunay(
            self._obj.pp.get_layer_as_df(celltypes_to_str=False),
            cell_types=self._obj.coords[Dims.LABELS].values,
            x=Features.X,
            y=Features.Y,
            label_col=Features.LABELS,
            include_center=include_center,
        )

        # Convert the DataFrame to an xarray DataArray
        # putting the df into a data array
        da = xr.DataArray(
            neighborhood_df.values,
            coords=[neighborhood_df.index, neighborhood_df.columns],
            dims=[Dims.CELLS, Dims.LABELS],
            name=key_added,
        )

        obj = xr.merge([self._obj, da])

        # adding the adjacency matrix to the object
        cells = obj.coords[Dims.CELLS].values
        da = xr.DataArray(
            adjacency_matrix,
            coords=[cells, cells],
            # xarray does not support duplicate dimension names, hence we need to introduce a second cell variable here
            dims=[Dims.CELLS, Dims.CELLS_2],
            name=key_adjacency_matrix,
        )

        return xr.merge([obj, da])

    def add_neighborhood_obs(
        self, features: Union[str, List[str]] = ["degree", "homophily", "inter_label_connectivity", "diversity_index"]
    ):
        """
        Adds neighborhood observations to the object by computing network features
        from the adjacency matrix.
        This method requires the `networkx` package to be installed. It checks if
        the adjacency matrix is present in the object, constructs a graph from the
        adjacency matrix, and computes network features.
        Parameters
        ----------
        features : str or List[str], optional
            The network features to compute. Possible features are ['degree', 'closeness_centrality', 'betweenness_centrality', 'homophily', 'inter_label_connectivity', 'diversity_index'].
        Raises
        ------
        ImportError
            If the `networkx` package is not installed.
        AssertionError
            If the adjacency matrix is not found in the object.
        Notes
        -----
        The adjacency matrix should be computed and stored in the object before
        calling this method. You can compute the adjacency matrix using methods
        from the `nh` module, such as `nh.compute_neighborhoods_radius()`.
        """

        try:
            import networkx as nx

            assert Layers.OBS in self._obj, "No observations found in the object."
            assert Layers.LA_PROPERTIES in self._obj, "No cell type labels found in the object."
            assert (
                Layers.ADJACENCY_MATRIX in self._obj
            ), "No adjacency matrix found in the object. Please compute the adjacency matrix first by running either of the methods contained in the nh module (e. g. nh.compute_neighborhoods_radius())."

            if isinstance(features, str):
                features = [features]
            assert len(features) > 0, "At least one feature must be provided."
            # ensuring that all features are valid
            valid_features = [
                "degree",
                "closeness_centrality",
                "betweenness_centrality",
                "homophily",
                "inter_label_connectivity",
                "diversity_index",
            ]
            assert all(
                [feature in valid_features for feature in features]
            ), f"Invalid feature provided. Valid features are: {valid_features}."

            adjacency_matrix = self._obj[Layers.ADJACENCY_MATRIX].values
            G = nx.from_numpy_array(adjacency_matrix)

            # adding labels as node attributes
            spatial_df = self._obj.pp.get_layer_as_df(Layers.OBS)
            assert (
                Features.LABELS in spatial_df.columns
            ), f"Feature {Features.LABELS} not found in the observation layer."
            # need to reset the index here to ensure that the labels are correctly assigned to the nodes
            labels_dict = spatial_df[Features.LABELS].reset_index(drop=True).to_dict()
            nx.set_node_attributes(G, labels_dict, Features.LABELS)
            network_features = _compute_network_features(G, features)

            # if the object already has neighborhood observations, we need to remove them from the obs
            existing_obs = [x for x in self._obj[Dims.FEATURES].values if x in valid_features]
            obj = self._obj.copy()
            if len(existing_obs) > 0:
                logger.warning(f"Overwriting existing neighborhood observations: {existing_obs}")
                obs = obj[Layers.OBS]
                obj[Layers.OBS] = obs.drop_sel(features=existing_obs)

            return obj.pp.add_obs_from_dataframe(network_features)

        except ImportError:
            raise ImportError("The networkx package is required for this function. Please install it first.")

    def compute_graph_features(
        self, features: Union[str, List[str]] = ["num_nodes", "num_edges", "density", "modularity", "assortativity"]
    ):
        """
        Compute various graph features from the adjacency matrix of the data.

        Parameters
        ----------
        features : Union[str, List[str]], optional
            A single feature or a list of features to compute. Valid features are:
            "num_nodes", "num_edges", "density", "modularity", "assortativity".
            Default is ["num_nodes", "num_edges", "density", "modularity", "assortativity"].

        Returns
        -------
        dict
            A dictionary where keys are the names of the computed features and values are the corresponding feature values.

        Raises
        ------
        ImportError
            If the `networkx` package is not installed.
        AssertionError
            If required layers (OBS, LA_PROPERTIES, ADJACENCY_MATRIX) are not found in the object.
            If no valid features are provided.
            If the LABELS feature is not found in the observation layer.

        Notes
        -----
        This method requires the `networkx` package to be installed. If you want to compute the modularity of the network,
        you will also need to install the `python-louvain` package.
        """
        try:
            import networkx as nx

            assert Layers.OBS in self._obj, "No observations found in the object."
            assert Layers.LA_PROPERTIES in self._obj, "No cell type labels found in the object."
            assert (
                Layers.ADJACENCY_MATRIX in self._obj
            ), "No adjacency matrix found in the object. Please compute the adjacency matrix first by running either of the methods contained in the nh module (e. g. nh.compute_neighborhoods_radius())."

            if isinstance(features, str):
                features = [features]
            assert len(features) > 0, "At least one feature must be provided."
            # ensuring that all features are valid
            valid_features = ["num_nodes", "num_edges", "density", "modularity", "assortativity"]
            assert all(
                [feature in valid_features for feature in features]
            ), f"Invalid feature provided. Valid features are: {valid_features}."

            adjacency_matrix = self._obj[Layers.ADJACENCY_MATRIX].values
            G = nx.from_numpy_array(adjacency_matrix)

            # adding labels as node attributes
            spatial_df = self._obj.pp.get_layer_as_df(Layers.OBS)
            assert (
                Features.LABELS in spatial_df.columns
            ), f"Feature {Features.LABELS} not found in the observation layer."
            # need to reset the index here to ensure that the labels are correctly assigned to the nodes
            labels_dict = spatial_df[Features.LABELS].reset_index(drop=True).to_dict()
            nx.set_node_attributes(G, labels_dict, Features.LABELS)

            # computing the features
            network_features = _compute_global_network_features(G, features)

            # for now this only gets returned
            # could later be added to the object as a new layer
            return network_features

        except ImportError:
            raise ImportError(
                "The networkx package is required for this function. Please install it first. If you want to compute modularity of the network, you will also need to install 'python-louvain'."
            )
