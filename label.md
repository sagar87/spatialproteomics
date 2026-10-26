# The label (`la`) accessor

The label accessor provides several functions associated with cell type annotations and gating.

## `ds.la` (LabelAccessor)

Adds functions for cell phenotyping.

Available on any `xarray.Dataset` after `import spatialproteomics as sp`, as `ds.la`.

### `ds.la.deselect(indices)`

Deselect specific label indices from the data object.

This method deselects specific label indices from the data object, effectively removing them from the selection.
The deselection can be performed using slices, lists, tuples, or individual integers.

**Parameters**

indices : slice, list, tuple, or int
    The label indices to be deselected. Can be a slice, list, tuple, or an individual integer.

**Returns**

xr.Dataset
    The updated data object with the deselected label indices.

**Notes**

- The function uses 'indices' to specify which labels to deselect.
- 'indices' can be provided as slices, lists, tuples, or an integer.
- The function then updates the data object to remove the deselected label indices.

### `ds.la.add_label_type(name: str, color: str='w') -> xr.Dataset`

Add a new label type to the data object.

This method adds a new label type with the specified 'name' and 'color' to the data object.
The label type is used to identify and categorize cells in the segmentation mask.

**Parameters**

name : str
    The name of the new label type to be added.
color : str, optional
    The color code to represent the new label type in the visualization. Default is "white" ("w").

**Returns**

xr.Dataset
    The updated data object with the newly added label type.

**Raises**

ValueError
    If the segmentation mask or observation table is not found in the data object.
    If the provided 'name' already exists as a label type.

**Notes**

- The function checks for the existence of the segmentation mask and observation table in the data object.
- It ensures that the 'name' of the new label type does not already exist in the label types.
- The function then adds the new label type with the given 'name' and 'color' to the data object.

### `ds.la.remove_label_type(cell_type: Union[int, str, List[Union[int, str]]]) -> xr.Dataset`

Remove specific cell type label(s) from the data object.

This method removes specific cell type label(s) identified by 'cell_type' from the data object.
The cell type label(s) are effectively removed, and their associated cells are assigned to the 'Unlabeled' category.

**Parameters**

cell_type : int, str or list of int or str
    The ID(s) or name(s) of the cell type label(s) to be removed.

**Returns**

xr.Dataset
    The updated data object with the specified cell type label(s) removed.

**Raises**

ValueError
    If the data object does not contain any cell type labels.
    If the specified 'cell_type' is not found among the existing cell type labels.

**Notes**

- The function first checks for the existence of cell type labels in the data object.
- It then removes the specified 'cell_type' from the cell type labels, setting their cells to the 'Unlabeled' category.

### `ds.la.add_label_property(array: Union[np.ndarray, list], prop: str)`

Add a label property for each unique cell type label.

This method adds a property, specified by 'prop', for each unique cell type label in the data object.
The property values are taken from the 'array' argument and assigned to each corresponding cell type label.

**Parameters**

array : numpy.ndarray or list
    An array or list containing property values to be assigned to each unique cell type label.
prop : str
    The name of the property to be added to the cell type labels.

**Returns**

xr.Dataset
    The updated data object with the added label property.

### `ds.la.set_label_name(label, name)`

Set the name of a specific cell type label.

This method sets the 'name' of a specific cell type label identified by the 'label'.
The 'label' can be either a label ID or the name of the cell type label.

**Parameters**

label : int or str
    The ID or name of the cell type label whose name will be updated.
name : str
    The new name to be assigned to the specified cell type label.

**Returns**

None

**Notes**

- The function converts the 'label' from its name to the corresponding ID for internal processing.
- It updates the name of the cell type label in the data object to the new 'name'.

### `ds.la.set_label_colors(labels: Union[str, List[str]], colors: Union[str, List[str]])`

Set the color of a specific cell type label.

This method sets the 'color' of a specific cell type label identified by the 'label'.
The 'label' can be either a label ID or the name of the cell type label.

**Parameters**

label : int or str
    The ID or name of the cell type label whose color will be updated.
color : any
    The new color to be assigned to the specified cell type label.

**Returns**

xr.Dataset

**Notes**

- The function converts the 'label' from its name to the corresponding ID for internal processing.
- It updates the color of the cell type label in the data object to the new 'color'.

### `ds.la.predict_cell_types_argmax(marker_dict: dict, key: str='_intensity', overwrite_existing_labels: bool=False, cell_col: str='cell', label_col: str='label', min_intensity: float=0.0, ignore_neighborhoods: bool=False)`

Predicts cell types based on the argmax classification of marker intensities.

Parameters:
    marker_dict (dict): A dictionary mapping markers to cell types. Each marker should be associated to one specific cell type.
    key (str, optional): The key of the quantification layer to use for classification. Defaults to "_intensity".
    overwrite_existing_labels (bool, optional): Whether to overwrite existing labels. Defaults to False.
    cell_col (str, optional): The name of the column to store cell IDs in the output dataframe. Defaults to "cell".
    label_col (str, optional): The name of the column to store predicted cell types in the output dataframe. Defaults to "label".
    min_intensity (float, optional): The minimum intensity threshold for assigning cells to a type. Defaults to 0.0.
    ignore_neighborhoods (bool, optional): Whether to remove existing neighborhoods when predicting cell types. Defaults to False.

Returns:
    xr.Dataset: A new spatialproteomics object with the predicted cell types added as labels.

Raises:
    AssertionError: If the quantification layer with the specified key is not found.
    AssertionError: If any of the markers specified in the marker dictionary are not present in the quantification layer.

### `ds.la.threshold_labels(threshold_dict: dict, label: Optional[str]=None, layer_key: str='_intensity')`

Binarise based on a threshold.
If a label is specified, the binarization is only applied to this cell type.

**Parameters**

threshold_dict : dict
    A dictionary mapping channels to threshold values.
label : str, optional
    The specified cell type for which the binarization is applied, by default None.
layer_key : str, optional
    The key for the new binary feature layer, by default "_percentage_positive_intensity".

**Returns**

xr.Dataset
    A new dataset object with the binary features added.

**Notes**

- If a label is specified, the binarization is only applied to the cells of that specific cell type.
- The binary feature is computed by comparing the intensity values of each channel to the threshold value.
- The binary feature is added as a new layer to the dataset object.

### `ds.la.add_labels(labels: Optional[dict]=None) -> xr.Dataset`

Add labels from a mapping (cell -> label) to the spatialproteomics object.

**Parameters**

labels : Union[dict, None]
    A dictionary containing cell labels as keys and corresponding labels as values.
    If None, a default labeling will be added. Default is None.

**Returns**

xr.Dataset
    The spatialproteomics object with added labels.

Notes:
------
This method converts the input dictionary into a pandas DataFrame and then adds the labels to the object
using the `la.add_labels_from_dataframe` method.

### `ds.la.add_labels_from_dataframe(df: Optional[pd.DataFrame]=None, cell_col: str='cell', label_col: str='label', colors: Optional[list]=None, names: Optional[list]=None, ignore_neighborhoods: bool=False) -> xr.Dataset`

Adds labels to the image container.

**Parameters**

df : Union[pd.DataFrame, None], optional
    A dataframe with the cell and label information. If None, a default labeling will be applied.
cell_col : str, optional
    The name of the column in the dataframe representing cell coordinates. Default is "cell".
label_col : str, optional
    The name of the column in the dataframe representing cell labels. Default is "label".
colors : Union[list, None], optional
    A list of colors corresponding to the cell labels. If None, random colors will be assigned. Default is None.
names : Union[list, None], optional
    A list of names corresponding to the cell labels. If None, default names will be assigned. Default is None.
ignore_neighborhoods : bool, optional
    If True, the function will ignore the neighborhoods in the object. Default is False.

**Returns**

xr.Dataset
    The updated image container with added labels.

### `ds.la.add_properties(array: Union[np.ndarray, list], prop: str='_labels', return_xarray: bool=False) -> xr.Dataset`

Adds properties to the image container. In order to add properties to an already existing property layer, use the la.add_label_property() method.

**Parameters**

array : Union[np.ndarray, list]
    An array or list of properties to be added to the image container.
prop : str, optional
    The name of the property. Default is Features.LABELS.
return_xarray : bool, optional
    If True, the function returns an xarray.DataArray with the properties instead of adding them to the image container.

**Returns**

xr.Dataset or xr.DataArray
    The updated image container with added properties or the properties as a separate xarray.DataArray.

### `ds.la.predict_cell_subtypes(subtype_dict: Union[dict, str], overwrite_existing_labels: bool=True) -> xr.Dataset`

Predict cell subtypes based on the binarized marker intensities.

**Parameters**

subtype_dict : dict
    A dictionary mapping cell subtypes to the binarized markers used for prediction.
    Instead of a dictionary, a path to a yaml file containing the subtype dictionary can be provided.
overwrite_existing_labels : bool, optional
    If True, existing labels will be overwritten by the new, more granular cell type predictions. Default is True.

**Returns**

xr.Dataset
    The updated image container with the predicted cell subtypes.

### `ds.la.set_label_level(level: str, ignore_neighborhoods: bool=False) -> xr.Dataset`

Set the label level to a specific level.

**Parameters**

level : str
    The name of the label level to set.
ignore_neighborhoods : bool, optional
    If True, the function will ignore the neighborhoods in the object. Default is False.

**Returns**

xr.Dataset
    The updated image container with the label level set to the specified level.

## Functions for the spatialdata backend (`sp.la`)

These functions take a `spatialdata.SpatialData` object as first argument and modify it in place unless `copy=True`.

### `sp.la.threshold_labels(sdata, threshold_dict: dict, key_added: str='binarization', table_key: str='table', layer_key: str='perc_pos', copy: bool=False)`

Binarise based on a threshold. If a label is specified, the binarization is only applied to this cell type.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
    threshold_dict (dict): A dictionary containing the threshold values for each channel.
    key_added (str, optional): The key under which the processed expression matrix will be stored in the obsm attribute of the spatialdata object. Defaults to "binarization".
    table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
    layer_key (str, optional): The key under which the expression matrix is stored in the layers attribute of the spatialdata object. Defaults to "perc_pos".
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.

### `sp.la.predict_cell_types_argmax(sdata, marker_dict: dict, min_intensity: float=0.0, table_key: str='table', label_key: str='celltype', copy: bool=False)`

This function predicts cell types based on the expression matrix using the argmax method.
It extracts the expression matrix from the spatialdata object, applies the argmax method,
and adds the predicted cell types to the spatialdata object.
The predicted cell types are stored in the obs attribute of the AnnData object in the tables attribute of the spatialdata object.
The argmax method assigns the cell type with the highest expression value to each cell.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
    marker_dict (dict): A dictionary containing the marker genes for each cell type.
    min_intensity (float, optional): The minimum intensity threshold for assigning cells to a type. Cells with all marker intensities below this threshold will be assigned to "Unlabeled". Defaults to 0.0.
    table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
    label_key (str, optional): The key under which the cell type labels are stored in the obs attribute of the spatialdata object. Defaults to "celltype".
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.

### `sp.la.predict_cell_subtypes(sdata, subtype_dict: Union[dict, str], table_key: str='table', layer_key: str='binarization', label_key: str='celltype', copy: bool=False)`

This function predicts cell subtypes based on the expression matrix using a subtype dictionary.
It extracts the expression matrix from the spatialdata object, applies the subtype prediction method,
and adds the predicted cell subtypes to the spatialdata object.
The predicted cell subtypes are stored in the obs attribute of the AnnData object in the tables attribute of the spatialdata object.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
    subtype_dict (Union[dict, str]): A dictionary mapping cell subtypes to the binarized markers used for prediction. Instead of a dictionary, a path to a yaml file containing the subtype dictionary can be provided.
    table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
    layer_key (str, optional): The key under which the binarized expression matrix is stored in the obsm attribute of the spatialdata object. Defaults to "binarization".
    label_key (str, optional): The key under which the cell type labels are stored in the obs attribute of the spatialdata object. Defaults to "celltype".
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
