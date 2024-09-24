from typing import List, Union

import numpy as np
import xarray as xr

from ..constants import COLORS, Dims, Features, Layers, Props
from .utils import _construct_neighborhood_df_radius


@xr.register_dataset_accessor("nh")
class NeighborhoodAccessor:
    """Adds functions for cellular neighborhoods."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, indices):
        # type checking
        if isinstance(indices, float):
            raise TypeError("Neighborhood indices must be valid integers, str, slices, List[int] or List[str].")

        if isinstance(indices, int):
            if indices not in self._obj.coords[Dims.NEIGHBORHOODS].values:
                raise ValueError(f"Neighborhood type {indices} not found.")

            sel = [indices]

        if isinstance(indices, str):
            raise NotImplementedError("String indexing is not yet implemented.")
            label_dict = self._obj.la._label_to_dict(Props.NAME, reverse=True)

            if indices not in label_dict:
                raise ValueError(f"Label type {indices} not found.")

            sel = [label_dict[indices]]

        if isinstance(indices, slice):
            raise NotImplementedError("Slice indexing is not yet implemented.")
            l_start = indices.start if indices.start is not None else 1
            l_stop = indices.stop if indices.stop is not None else self._obj.sizes[Dims.LABELS]
            sel = [i for i in range(l_start, l_stop)]

        if isinstance(indices, (list, tuple)):
            raise NotImplementedError("List indexing is not yet implemented.")
            if not all([isinstance(i, (str, int)) for i in indices]):
                raise TypeError("Label indices must be valid integers, str, slices, List[int] or List[str].")

            sel = []
            for i in indices:
                if isinstance(i, str):
                    label_dict = self._obj.la._label_to_dict(Props.NAME, reverse=True)

                    if i not in label_dict:
                        raise ValueError(f"Label type {i} not found.")

                    sel.append(label_dict[i])

                if isinstance(i, int):
                    if i not in self._obj.coords[Dims.LABELS].values:
                        raise ValueError(f"Label type {i} not found.")

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

    # TODO: reimplement contains
    # TODO: reimplement deselect
    # TODO: reimplement colorization of neighborhoods
    # TODO: reimplement renaming of neighborhoods

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
            names = [f"Neighborhood {i+1}" for i in range(len(unique_neighborhoods))]

        obj = obj.nh.add_properties(names, Props.NAME)

        return obj
