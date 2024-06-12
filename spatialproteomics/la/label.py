from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from skimage.segmentation import relabel_sequential

from ..base_logger import logger
from ..constants import Dims, Features, Layers, Props


@xr.register_dataset_accessor("la")
class LabelAccessor:
    """Adds functions for cell phenotyping."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __contains__(self, key):
        if Layers.PROPERTIES not in self._obj:
            return False

        label_dict = self._obj.la._label_to_dict(Props.NAME)
        return key in label_dict.keys() or key in label_dict.values()

    def __getitem__(self, indices):
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
            l_stop = indices.stop if indices.stop is not None else self._obj.sizes[Dims.LABELS]
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
        if isinstance(indices, slice):
            l_start = indices.start if indices.start is not None else 1
            l_stop = indices.stop if indices.stop is not None else self._obj.sizes[Dims.LABELS]
            sel = [i for i in range(l_start, l_stop)]
        elif isinstance(indices, list):
            assert all([isinstance(i, (int, str)) for i in indices]), "All label indices must be integers or strings."
            if all([isinstance(i, int) for i in indices]):
                sel = indices
            else:
                label_dict = self._obj.la._label_to_dict(Props.NAME, reverse=True)
                for idx in indices:
                    if idx not in label_dict:
                        raise ValueError(f"Label type {indices} not found.")
                sel = [label_dict[idx] for idx in indices]
        elif isinstance(indices, tuple):
            indices = list(indices)
            all_int = all([type(i) is int for i in indices])
            assert all_int, "All label indices must be integers."
            sel = indices
        elif isinstance(indices, str):
            label_dict = self._obj.la._label_to_dict(Props.NAME, reverse=True)
            if indices not in label_dict:
                raise ValueError(f"Label type {indices} not found.")
            sel = [label_dict[indices]]
        else:
            assert type(indices) is int, "Label must be provided as slices, lists, tuple or int."
            sel = [indices]

        inv_sel = [i for i in self._obj.coords[Dims.LABELS] if i not in sel]

        cells = self._obj.la._filter_cells_by_label(inv_sel)
        return self._obj.sel({Dims.LABELS: inv_sel, Dims.CELLS: cells})

    def _relabel_dict(self, dictionary: dict):
        _, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values)
        return {fw[k]: v for k, v in dictionary.items()}

    def _label_to_dict(self, prop: str, reverse: bool = False, relabel: bool = False) -> dict:
        """
        Returns a dictionary that maps each label to a list to their property.

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
        labels_layer = self._obj[Layers.PROPERTIES]
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

    def _cells_to_label(self, relabel: bool = False, include_unlabeled: bool = False) -> dict:
        """
        Returns a dictionary that maps each label to a list of cells.

        Parameters
        ----------
        relabel : bool, optional
            If True, relabels the dictionary keys to consecutive integers starting from 1.
            Default is False.
        include_unlabeled : bool, optional
            If True, includes cells that are unlabeled in the dictionary.
            Default is False.

        Returns
        -------
        dict
            A dictionary that maps each label to a list of cells. The keys are label values,
            and the values are lists of cell indices.
        """
        label_dict = {
            label.item(): self._obj.la._filter_cells_by_label(label.item()) for label in self._obj.coords[Dims.LABELS]
        }

        if include_unlabeled:
            label_dict[0] = self._obj.la._filter_cells_by_label(0)

        if relabel:
            return self._obj.la._relabel_dict(label_dict)

        return label_dict

    def _filter_cells_by_label(self, items: Union[int, List[int]]):
        """
        Filter cells by label.

        Parameters
        ----------
        items : int or List[int]
            The label(s) to filter cells by. If an integer is provided, only cells with that label will be returned.
            If a list of integers is provided, cells with any of the labels in the list will be returned.

        Returns
        -------
        numpy.ndarray
            An array containing the selected cells.
        """
        if type(items) is int:
            items = [items]

        cells = self._obj[Layers.OBS].loc[:, Features.LABELS].values.copy()
        cells_bool = np.isin(cells, items)
        cells_sel = self._obj.coords[Dims.CELLS][cells_bool].values

        return cells_sel

    def _label_name_to_id(self, label):
        """
        Convert a label name to its corresponding ID.

        Parameters
        ----------
        label : str
            The name of the label to convert.

        Returns
        -------
        int
            The ID corresponding to the given label name.

        Raises
        ------
        ValueError
            If the given label name is not found in the label names dictionary.
        """
        label_names_reverse = self._obj.la._label_to_dict(Props.NAME, reverse=True)
        if label not in label_names_reverse:
            raise ValueError(f"Cell type {label} not found.")

        return label_names_reverse[label]

    def _filter_by_intensity(
        self, channel: str, func: Callable, layer_key: str = Layers.INTENSITY, return_int_array: bool = True
    ) -> xr.Dataset:
        """
        Filter the cells based on intensity values for a specific channel. Useful for binarizing markers.

        Parameters:
            channel (str): The channel to filter on.
            func (Callable): A function that takes in intensity values and returns a boolean array indicating which cells to keep.
            layer_key (str, optional): The key of the layer containing the intensity values. Defaults to Layers.INTENSITY.
            return_int_array (bool, optional): Whether to return the filtered cells as a numeric array (0 for False, 1 for True). Defaults to True.

        Returns:
            xarray.Dataset: The filtered cells as a DataArray if return_int_array is False, otherwise a numeric array.
        """
        cells = self._obj[layer_key].sel({Dims.CHANNELS: channel}).values.copy()
        cells_bool = func(cells)

        if return_int_array:
            # turning the boolean array into a numeric array (where 0 is False, 1 is True)
            return cells_bool.astype(int)

        cells_sel = self._obj.coords[Dims.CELLS][cells_bool].values

        return self._obj.sel({Dims.CELLS: cells_sel})

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
        if Layers.PROPERTIES in self._obj:
            if name in self._obj[Layers.PROPERTIES].sel({Dims.PROPS: Props.NAME}):
                raise ValueError("Label type already exists.")

        array = np.array([name, color]).reshape(1, -1)

        # if label properties (Layers.PROPERTIES) are not present, create them
        if Layers.PROPERTIES not in self._obj:
            da = xr.DataArray(
                array,
                coords=[np.array([1]), [Props.NAME, Props.COLOR]],
                dims=[Dims.LABELS, Dims.PROPS],
                name=Layers.PROPERTIES,
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
                [self._obj[Layers.PROPERTIES], da],
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

        if Layers.PROPERTIES not in self._obj:
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
        # checking that we already have properties
        assert (
            Layers.PROPERTIES in self._obj
        ), "No label layer found in the data object. Please add labels, e. g. via la.predict_cell_types_argmax() or tl.astir()."
        # making sure the property does not exist already
        assert prop not in self._obj.coords[Dims.PROPS].values, f"Property {prop} already exists."

        # checking that the length of the array matches the number of labels
        assert len(array) == len(
            self._obj.coords[Dims.LABELS].values
        ), "The length of the array must match the number of labels."

        unique_labels = self._obj.coords[Dims.LABELS].values

        if type(array) is list:
            array = np.array(array)

        da = xr.DataArray(
            array.reshape(-1, 1),
            coords=[unique_labels.astype(int), [prop]],
            dims=[Dims.LABELS, Dims.PROPS],
            name=Layers.PROPERTIES,
        )

        if Layers.PROPERTIES in self._obj:
            da = xr.concat(
                [self._obj[Layers.PROPERTIES], da],
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
        # checking that a label layer is already present
        assert Layers.PROPERTIES in self._obj, "No label layer found in the data object."
        # checking if the old label exists
        assert label in self._obj.la, f"Cell type {label} not found. Existing cell types: {self._obj.la}"
        # checking if the new label already exists
        assert name not in self._obj[Layers.PROPERTIES].sel(
            {Dims.PROPS: Props.NAME}
        ), f"Label name {name} already exists."

        # getting the original label properties
        property_layer = self._obj[Layers.PROPERTIES].copy()

        if isinstance(label, str):
            label = self._obj.la._label_name_to_id(label)

        property_layer.loc[label, Props.NAME] = name

        # removing the old property layer
        obj = self._obj.pp.drop_layers(Layers.PROPERTIES)

        # adding the new property layer
        return xr.merge([property_layer, obj])

    def set_label_colors(self, labels: Union[str, List[str]], colors: Union[str, List[str]]):
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
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(colors, str):
            colors = [colors]

        # checking that there are as many colors as labels
        assert len(labels) == len(colors), "The number of labels and colors must be the same."

        # checking that a label layer is already present
        assert (
            Layers.PROPERTIES in self._obj
        ), "No label layer found in the data object. Please add labels before setting colors, e. g. by using la.predict_cell_types_argmax() or tl.astir()."

        # obtaining the current properties
        props_layer = self._obj.coords[Dims.PROPS].values.tolist()
        labels_layer = self._obj.coords[Dims.LABELS].values.tolist()
        array = self._obj[Layers.PROPERTIES].values.copy()

        for label, color in zip(labels, colors):
            # if the label does not exist in the object, a warning is thrown and we continue
            if label not in self._obj.la:
                logger.warning(f"Label {label} not found in the data object. Skipping.")
                continue

            # getting the id for the label
            label = self._obj.la._label_name_to_id(label)

            # setting the new color for the given label
            array[labels_layer.index(label), props_layer.index(Props.COLOR)] = color

        da = xr.DataArray(
            array,
            coords=[labels_layer, props_layer],
            dims=[Dims.LABELS, Dims.PROPS],
            name=Layers.PROPERTIES,
        )

        return xr.merge([self._obj.drop_vars(Layers.PROPERTIES), da])

    def predict_cell_types_argmax(
        self,
        marker_dict: dict,
        key: str = Layers.INTENSITY,
        overwrite_existing_labels: bool = False,
        cell_col: str = "cell",
        label_col: str = "label",
    ):
        """
        Predicts cell types based on the argmax classification of marker intensities.

        Parameters:
            marker_dict (dict): A dictionary mapping cell types to markers.
            key (str, optional): The key of the quantification layer to use for classification. Defaults to Layers.INTENSITY.
            overwrite_existing_labels (bool, optional): Whether to overwrite existing labels. Defaults to False.
            cell_col (str, optional): The name of the column to store cell IDs in the output dataframe. Defaults to "cell".
            label_col (str, optional): The name of the column to store predicted cell types in the output dataframe. Defaults to "label".

        Returns:
            spatial_data.SpatialData: A new SpatialData object with the predicted cell types added as labels.

        Raises:
            AssertionError: If the quantification layer with the specified key is not found.
            AssertionError: If any of the markers specified in the marker dictionary are not present in the quantification layer.
        """
        # asserting that a quantification with the key exists
        assert (
            key in self._obj
        ), f"Quantification layer with key {key} not found. Please run pp.add_quantification() before classifying cell types."
        celltypes, markers = list(marker_dict.keys()), list(marker_dict.values())
        # asserting that all markers are present in the quantification layer
        markers_present = self._obj.coords[Dims.CHANNELS].values
        assert (
            len(set(markers) - set(markers_present)) == 0
        ), f"The following markers were not found in quantification layer: {set(markers) - set(markers_present)}."

        # only looking at the markers specified in the marker dict
        obj = self._obj.copy()
        # getting the argmax for each cell
        argmax_classification = np.argmax(obj.pp[markers][key].values, axis=1)

        # translating the argmax classification into cell types
        celltypes_argmax = np.array(celltypes)[argmax_classification]

        # putting everything into a dataframe
        celltype_prediction_df = pd.DataFrame(
            zip(obj.coords[Dims.CELLS].values, celltypes_argmax), columns=[cell_col, label_col]
        )

        # if there already exist partial annotations, and overwrite is set to False, we merge the two dataframes
        if Features.LABELS in obj.coords[Dims.FEATURES].values and not overwrite_existing_labels:
            id_to_label_dict = obj.la._label_to_dict(Props.NAME)
            existing_labels_numeric = obj[Layers.OBS].sel(features=Features.LABELS).values
            # getting a boolean array that is 0 if there exists a label, and 1 if there is no label
            ct_exists = np.where(existing_labels_numeric != 0, 1, 0)
            existing_labels = [id_to_label_dict[x] for x in existing_labels_numeric]
            celltype_prediction_df["ct_exists"] = ct_exists
            celltype_prediction_df["old_ct_prediction"] = existing_labels

            # keeping original predictions and using the argmax otherwise
            celltype_prediction_df[label_col] = celltype_prediction_df.apply(
                lambda row: row[label_col] if row["ct_exists"] == 0 else row["old_ct_prediction"], axis=1
            )

            # removing the intermediate columns
            celltype_prediction_df = celltype_prediction_df.drop(columns=["ct_exists", "old_ct_prediction"])

        # need to remove the old labels first (if there are old labels)
        obs = obj[Layers.OBS]
        if Features.LABELS in obs.coords[Dims.FEATURES].values:
            # ideally, we should select by Dims.FEATURES here, but that does not work syntactically
            obj[Layers.OBS] = obs.drop_sel(features=Features.LABELS)
            # removing the old colors
            obj = obj.pp.drop_layers(Layers.PROPERTIES)

        # adding the new labels
        return obj.pp.add_labels_from_dataframe(celltype_prediction_df)

    def _threshold_label(
        self, channel: str, threshold: float, layer_key: str = Layers.INTENSITY, label: Optional[str] = None
    ):
        if layer_key not in self._obj:
            raise KeyError(f'No layer "{layer_key}" found. Please add it first using pp.add_quantification().')

        if channel not in self._obj.coords[Dims.CHANNELS]:
            raise KeyError(f'No channel "{channel}".')

        obj = self._obj.copy()
        label_pos = obj.la._filter_by_intensity(channel, lambda x: x >= threshold, layer_key=layer_key)

        column = f"{channel}_binarized"

        if label is not None:
            # getting all of the cells that should have a 1 in the binary vector
            label_idx = obj.la[label].cells.values
            # creating a boolean mask indicating whether each element of cells is in cells_subset
            label_mask = np.isin(obj.cells.values, label_idx).astype(int)
            label_pos *= label_mask
            column = f"{channel}_{label}_binarized"

        obj = obj.pp.add_feature(column, label_pos)

        return obj

    def threshold_labels(self, threshold_dict: dict, label: Optional[str] = None, layer_key: str = Layers.INTENSITY):
        """
        Binarise based on a threshold.
        If a label is specified, the binarization is only applied to this cell type.

        Parameters
        ----------
        threshold_dict : dict
            A dictionary mapping channels to threshold values.
        label : str, optional
            The specified cell type for which the binarization is applied, by default None.
        layer_key : str, optional
            The key for the new binary feature layer, by default "_percentage_positive_intensity".

        Returns
        -------
        xr.Dataset
            A new dataset object with the binary features added.

        Notes
        -----
        - If a label is specified, the binarization is only applied to the cells of that specific cell type.
        - The binary feature is computed by comparing the intensity values of each channel to the threshold value.
        - The binary feature is added as a new layer to the dataset object.
        """
        obj = self._obj.copy()
        for channel, threshold in threshold_dict.items():
            obj = obj.la._threshold_label(channel=channel, threshold=threshold, layer_key=layer_key, label=label)

        return obj
