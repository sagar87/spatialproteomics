from typing import List, Union

import numpy as np
import xarray as xr

from ..constants import Dims, Features, Layers, Props


@xr.register_dataset_accessor("nh")
class NeighborhoodAccessor:
    """Adds functions for cellular neighborhoods."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # TODO: UPDATE THIS
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

        # TODO: THIS NEEDS TO BE BY NH
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
