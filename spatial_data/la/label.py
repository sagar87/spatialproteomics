from typing import Callable, List, Union

import numpy as np
import xarray as xr
from skimage.segmentation import relabel_sequential

from ..base_logger import logger
from ..constants import Dims, Features, Layers, Props


@xr.register_dataset_accessor("la")
class LabelAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __contains__(self, key):
        if Layers.LABELS not in self._obj:
            return False

        label_dict = self._obj.la._label_to_dict(Props.NAME)
        return key in label_dict.keys() or key in label_dict.values()

    def _relabel_dict(self, dictionary: dict):
        _, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values)
        return {fw[k]: v for k, v in dictionary.items()}

    def _label_to_dict(self, prop: str, reverse: bool = False, relabel: bool = False) -> dict:
        """Returns a dictionary that maps each label to a list to their property.

        Parameters
        ----------
        prop : str
            The property to map to the labels.
        reverse : bool
            If True, the dictionary will be reversed.
        relabel : bool
            Deprecated.

        Returns
        -------
        label_dict : dict
            A dictionary that maps each label to a list to their property.
        """
        labels_layer = self._obj[Layers.LABELS]
        labels = self._obj.coords[Dims.LABELS]

        label_dict = {}

        for label in labels:
            current_row = labels_layer.loc[label, prop]
            label_dict[label.values.item()] = current_row.values.item()

        if relabel:
            return self._obj.la._relabel_dict(label_dict)

        if reverse:
            label_dict = {v: k for k, v in label_dict.items()}

        return label_dict

    def _cells_to_label(self, relabel: bool = False, include_unlabeled: bool = False):
        """Returns a dictionary that maps each label to a list of cells."""

        label_dict = {
            label.item(): self._obj.la._filter_cells_by_label(label.item()) for label in self._obj.coords[Dims.LABELS]
        }

        if include_unlabeled:
            label_dict[0] = self._obj.la._filter_cells_by_label(0)

        if relabel:
            return self._obj.la._relabel_dict(label_dict)

        return label_dict

    def _filter_cells_by_label(self, items: Union[int, List[int]]):
        """Returns the list of cells with the labels from items."""
        if type(items) is int:
            items = [items]

        cells = self._obj[Layers.OBS].loc[:, Features.LABELS].values.copy()
        cells_bool = np.isin(cells, items)
        cells_sel = self._obj.coords[Dims.CELLS][cells_bool].values

        return cells_sel

    def _label_name_to_id(self, label):
        """Given a label name return its id."""
        label_names_reverse = self._obj.la._label_to_dict(Props.NAME, reverse=True)
        if label not in label_names_reverse:
            raise ValueError(f"Cell type {label} not found.")

        return label_names_reverse[label]

    def filter_by_intensity(self, col: str, func: Callable, layer_key: str):
        """Returns the list of cells with the labels from items."""
        cells = self._obj[layer_key].sel({Dims.CHANNELS: col}).values.copy()
        cells_bool = func(cells)
        cells_sel = self._obj.coords[Dims.CELLS][cells_bool].values

        return self._obj.sel({Dims.CELLS: cells_sel})

    def __getitem__(self, indices):
        """
        Sub selects labels.
        """
        # type checking
        if isinstance(indices, float):
            raise TypeError("Label indices must be valid integers, str, slices, List[int] or List[str].")

        if isinstance(indices, int):
            if indices not in self._obj.coords[Dims.LABELS].values:
                raise ValueError(f"Label type {indices} not found.")

            sel = [indices]

        if isinstance(indices, str):
            label_dict = self._obj.la._label_to_dict(Props.NAME, reverse=True)

            if indices not in label_dict:
                raise ValueError(f"Label type {indices} not found.")

            sel = [label_dict[indices]]

        if isinstance(indices, slice):
            l_start = indices.start if indices.start is not None else 1
            l_stop = indices.stop if indices.stop is not None else self._obj.dims[Dims.LABELS]
            sel = [i for i in range(l_start, l_stop)]

        if isinstance(indices, (list, tuple)):
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

        cells = self._obj.la._filter_cells_by_label(sel)
        return self._obj.sel({Dims.LABELS: sel, Dims.CELLS: cells})

    def deselect(self, indices):
        """
        Deselect specific label indices from the data object.

        This method deselects specific label indices from the data object, effectively removing them from the selection.
        The deselection can be performed using slices, lists, tuples, or individual integers.

        Parameters
        ----------
        indices : slice, list, tuple, or int
            The label indices to be deselected. Can be a slice, list, tuple, or an individual integer.

        Returns
        -------
        any
            The updated data object with the deselected label indices.

        Notes
        -----
        - The function uses 'indices' to specify which labels to deselect.
        - 'indices' can be provided as slices, lists, tuples, or an integer.
        - The function then updates the data object to remove the deselected label indices.
        """
        # REFACTOR
        if type(indices) is slice:
            l_start = indices.start if indices.start is not None else 1
            l_stop = indices.stop if indices.stop is not None else self._obj.dims[Dims.LABELS]
            sel = [i for i in range(l_start, l_stop)]
        elif type(indices) is list:
            assert all([isinstance(i, (int, str)) for i in indices]), "All label indices must be integers."
            sel = indices

        elif type(indices) is tuple:
            indices = list(indices)
            all_int = all([type(i) is int for i in indices])
            assert all_int, "All label indices must be integers."
            sel = indices
        else:
            assert type(indices) is int, "Label must be provided as slices, lists, tuple or int."

            sel = [indices]

        total_labels = self._obj.dims[Dims.LABELS] + 1
        inv_sel = [i for i in range(1, total_labels) if i not in sel]
        cells = self._obj.la._filter_cells_by_label(inv_sel)
        return self._obj.sel({Dims.LABELS: inv_sel, Dims.CELLS: cells})

    def add_label_type(self, name: str, color: str = "w") -> xr.Dataset:
        """
        Add a new label type to the data object.

        This method adds a new label type with the specified 'name' and 'color' to the data object.
        The label type is used to identify and categorize cells in the segmentation mask.

        Parameters
        ----------
        name : str
            The name of the new label type to be added.
        color : str, optional
            The color code to represent the new label type in the visualization. Default is "white" ("w").

        Returns
        -------
        xr.Dataset
            The updated data object with the newly added label type.

        Raises
        ------
        ValueError
            If the segmentation mask or observation table is not found in the data object.
            If the provided 'name' already exists as a label type.

        Notes
        -----
        - The function checks for the existence of the segmentation mask and observation table in the data object.
        - It ensures that the 'name' of the new label type does not already exist in the label types.
        - The function then adds the new label type with the given 'name' and 'color' to the data object.
        """

        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found.")
        if Layers.OBS not in self._obj:
            raise ValueError("No observation table found.")

        # Assert that label type is not already present
        if Layers.LABELS in self._obj:
            if name in self._obj[Layers.LABELS].sel({Dims.PROPS: Props.NAME}):
                raise ValueError("Label type already exists.")

        array = np.array([name, color]).reshape(1, -1)

        # if label annotations (Layers.LABELS) are not present, create them
        if Layers.LABELS not in self._obj:
            da = xr.DataArray(
                array,
                coords=[np.array([1]), [Props.NAME, Props.COLOR]],
                dims=[Dims.LABELS, Dims.PROPS],
                name=Layers.LABELS,
            )

            db = xr.DataArray(
                np.zeros(self._obj.coords[Dims.CELLS].shape[0]).reshape(-1, 1),
                coords=[self._obj.coords[Dims.CELLS], [Features.LABELS]],
                dims=[Dims.CELLS, Dims.FEATURES],
                name=Layers.OBS,
            )

            obj = xr.merge([self._obj, da, db])
        else:
            new_coord = self._obj.coords[Dims.LABELS].values.max() + 1
            da = xr.DataArray(
                array,
                coords=[np.array([new_coord]), [Props.NAME, Props.COLOR]],
                dims=[Dims.LABELS, Dims.PROPS],
            )

            da = xr.concat(
                [self._obj[Layers.LABELS], da],
                dim=Dims.LABELS,
            )
            obj = xr.merge([self._obj, da])

        return obj

    def remove_label_type(self, cell_type: Union[int, List[int]]) -> xr.Dataset:
        """
        Remove specific cell type label(s) from the data object.

        This method removes specific cell type label(s) identified by 'cell_type' from the data object.
        The cell type label(s) are effectively removed, and their associated cells are assigned to the 'Unlabeled' category.

        Parameters
        ----------
        cell_type : int or list of int
            The ID(s) of the cell type label(s) to be removed.

        Returns
        -------
        xr.Dataset
            The updated data object with the specified cell type label(s) removed.

        Raises
        ------
        ValueError
            If the data object does not contain any cell type labels.
            If the specified 'cell_type' is not found among the existing cell type labels.

        Notes
        -----
        - The function first checks for the existence of cell type labels in the data object.
        - It then removes the specified 'cell_type' from the cell type labels, setting their cells to the 'Unlabeled' category.
        """
        if isinstance(cell_type, int):
            cell_type = [cell_type]

        if isinstance(cell_type, str):
            cell_type = [self._obj.la._label_name_to_id(cell_type)]
        # TODO: If list should properly get cell -type
        # TODO: should call reset label type prior to removing the cell type

        if Layers.LABELS not in self._obj:
            raise ValueError("No cell type labels found.")

        for i in cell_type:
            if i not in self._obj.coords[Dims.LABELS].values:
                raise ValueError(f"Cell type {i} not found.")

        cells_bool = (self._obj[Layers.OBS].sel({Dims.FEATURES: Features.LABELS}) == cell_type).values
        cells = self._obj.coords[Dims.CELLS][cells_bool].values

        self._obj[Layers.OBS].loc[{Dims.FEATURES: Features.LABELS, Dims.CELLS: cells}] = 0

        return self._obj.sel({Dims.LABELS: [i for i in self._obj.coords[Dims.LABELS] if i not in cell_type]})

    def add_label_property(self, array: Union[np.ndarray, list], prop: str):
        """
        Add a label property for each unique cell type label.

        This method adds a property, specified by 'prop', for each unique cell type label in the data object.
        The property values are taken from the 'array' argument and assigned to each corresponding cell type label.

        Parameters
        ----------
        array : numpy.ndarray or list
            An array or list containing property values to be assigned to each unique cell type label.
        prop : str
            The name of the property to be added to the cell type labels.

        Returns
        -------
        any
            The updated data object with the added label property.

        Notes
        -----
        - The function ensures that 'array' is converted to a NumPy array.
        - It creates a DataArray 'da' with the given 'array' as the property values and unique cell type labels as coords.
        - The DataArray 'da' is then merged into the data object, associating properties with cell type labels.
        - If the label property already exists in the data object, it will be updated with the new property values.
        """
        unique_labels = self._obj.coords[
            Dims.LABELS
        ].values  # np.unique(self._obj[Layers.OBS].sel({Dims.FEATURES: Features.LABELS}))

        if type(array) is list:
            array = np.array(array)

        da = xr.DataArray(
            array.reshape(-1, 1),
            coords=[unique_labels.astype(int), [prop]],
            dims=[Dims.LABELS, Dims.PROPS],
            name=Layers.LABELS,
        )

        if Layers.LABELS in self._obj:
            # import pdb;pdb.set_trace()
            da = xr.concat(
                [self._obj[Layers.LABELS], da],
                dim=Dims.PROPS,
            )

        return xr.merge([da, self._obj])

    def set_label_name(self, label, name):
        """
        Set the name of a specific cell type label.

        This method sets the 'name' of a specific cell type label identified by the 'label'.
        The 'label' can be either a label ID or the name of the cell type label.

        Parameters
        ----------
        label : int or str
            The ID or name of the cell type label whose name will be updated.
        name : str
            The new name to be assigned to the specified cell type label.

        Returns
        -------
        None

        Notes
        -----
        - The function converts the 'label' from its name to the corresponding ID for internal processing.
        - It updates the name of the cell type label in the data object to the new 'name'.
        """
        if isinstance(label, str):
            label = self._obj.la._label_name_to_id(label)

        self._obj[Layers.LABELS].loc[label, Props.NAME] = name

    def set_label_color(self, label, color):
        """
        Set the color of a specific cell type label.

        This method sets the 'color' of a specific cell type label identified by the 'label'.
        The 'label' can be either a label ID or the name of the cell type label.

        Parameters
        ----------
        label : int or str
            The ID or name of the cell type label whose color will be updated.
        color : any
            The new color to be assigned to the specified cell type label.

        Returns
        -------
        None

        Notes
        -----
        - The function converts the 'label' from its name to the corresponding ID for internal processing.
        - It updates the color of the cell type label in the data object to the new 'color'.
        """
        if label not in self._obj.la:
            logger.info(f"Did not find {label}.")
            return self._obj

        if isinstance(label, str):
            label = self._obj.la._label_name_to_id(label)

        props = self._obj.coords[Dims.PROPS].values.tolist()
        labels = self._obj.coords[Dims.LABELS].values.tolist()
        array = self._obj._labels.values.copy()
        array[labels.index(label), props.index(Props.COLOR)] = color

        da = xr.DataArray(
            array,
            coords=[labels, props],
            dims=[Dims.LABELS, Dims.PROPS],
            name=Layers.LABELS,
        )

        return xr.merge([self._obj.drop_vars(Layers.LABELS), da])


def predict_cell_types(self, marker_dict: dict, key: str = Layers.INTENSITY, overwrite_existing_labels: bool = False):
    # first, we only want to get cells for which we do not have a classification yet (unless overwrite_existing_labels is True, in that case we reclassify all cells)
    if not overwrite_existing_labels:
        # TODO: this is incorrect
        cells = self._obj.la._filter_cells_by_label(0)
    else:
        cells = self._obj.coords[Dims.CELLS].values
    # preds = dict(zip(dapi_cells.coords['cells'].values, np.array(labels)[np.argmax(dapi_cells._arcsinh_median.values,1)]))
