# The preprocessing (`pp`) accessor

The preprocessing accessor provides several methods to subset and process image data.

## `ds.pp` (PreprocessingAccessor)

The image accessor enables fast indexing and preprocessing of the spatialproteomics object.

Available on any `xarray.Dataset` after `import spatialproteomics as sp`, as `ds.pp`.

### `ds.pp[...]`

Fast subsetting the image container. The following examples show how
the user can subset the image container:

Subset the image container using x and y coordinates:
>>> ds.pp[0:50, 0:50]

Subset the image container using x and y coordinates and channels:
>>> ds.pp['Hoechst', 0:50, 0:50]

Subset the image container using channels:
>>> ds.pp['Hoechst']

Multiple channels can be selected by passing a list of channels:
>>> ds.pp[['Hoechst', 'CD4']]

**Parameters**

indices : str, slice, list, tuple
    The indices to subset the image container.

**Returns**

xr.Dataset
    The subsetted image container.

### `ds.pp.get_bbox(x_slice: slice, y_slice: slice) -> xr.Dataset`

Returns the bounds of the image container.

**Parameters**

x_slice : slice
    The slice representing the x-coordinates for the bounding box.
y_slice : slice
    The slice representing the y-coordinates for the bounding box.

**Returns**

xr.Dataset
    The updated image container.

### `ds.pp.get_channels(channels: Union[List[str], str]) -> xr.Dataset`

Retrieve the specified channels from the dataset.

**Parameters**

channels : Union[List[str], str]
    The channels to retrieve. Can be a single channel name or a list of channel names.

**Returns**

xr.Dataset
    The dataset containing the specified channels.

### `ds.pp.add_channel(channels: Union[str, list], array: np.ndarray, layer_key: str='_image') -> xr.Dataset`

Adds channel(s) to an existing spatialproteomics object.

**Parameters**

channels : Union[str, list]
    The name of the channel or a list of channel names to be added.
array : np.ndarray
    The numpy array representing the channel(s) to be added.
layer_key : str
    The layer key where the channel(s) should be added. Default is '_image'.

**Returns**

xr.Dataset
    The updated spatialproteomics object with added channel(s).

### `ds.pp.drop_channel(channels: Union[str, list]) -> xr.Dataset`

Drops channel(s) from an existing spatialproteomics object.

**Parameters**

channels : Union[str, list]
    The name of the channel or a list of channel names to be dropped.

**Returns**

xr.Dataset
    The updated spatialproteomics object with dropped channel(s).

### `ds.pp.add_segmentation(segmentation: Union[str, np.ndarray]=None, reindex: bool=False, keep_labels: bool=True, add_obs: bool=True) -> xr.Dataset`

Adds a segmentation mask field to the xarray dataset. This will be stored in the '_segmentation' layer.

**Parameters**

segmentation : str or np.ndarray
    A segmentation mask, i.e., a np.ndarray with image.shape = (x, y),
    that indicates the location of each cell, or a layer key.
mask_growth : int
    The number of pixels by which the segmentation mask should be grown.
reindex : bool
    If true the segmentation mask is relabeled to have continuous numbers from 1 to n.
keep_labels : bool
    When using cellpose on multiple channels, you may already get some initial celltype annotations from those.
    If you want to keep those annotations, set this to True. Default is True.
add_obs : bool
    If True, centroids are added to the xarray. Default is True.

**Returns**

xr.Dataset
    The amended xarray.

### `ds.pp.add_layer(array: np.ndarray, key_added: str='_mask') -> xr.Dataset`

Adds a layer (such as a mask highlighting artifacts) to the xarray dataset.

**Parameters**

array : np.ndarray
    The array representing the layer to be added. Can either be 2D or 3D (in this case, the first dimension should be the number of channels).
key_added : str, optional
    The name of the added layer in the xarray dataset. Default is '_mask'.
**Returns**

xr.Dataset
    The updated dataset with the added layer.
**Raises**

AssertionError
    If the array is not 2-dimensional or its shape does not match the image shape.
**Notes**

This method adds a layer to the xarray dataset, where the layer has the same shape as the image field.
The array should be a 2-dimensional numpy array representing the segmentation mask or layer to be added.
The layer is created as a DataArray with the same coordinates and dimensions as the image field.
The name of the added layer in the xarray dataset can be specified using the `key_added` parameter.
The amended xarray dataset is returned after merging the original dataset with the new layer.

### `ds.pp.add_layer_from_dataframe(df: pd.DataFrame, key_added: str='_la_layers') -> xr.Dataset`

Adds a dataframe as a layer to the xarray object. This is similar to add_obs, with the only difference that it can be used to add any kind of data to the xarray object.
Useful to add things like string-based labels or other metadata.

**Parameters**

df : pd.DataFrame
    A dataframe with the observation values.

**Returns**

xr.Dataset
    The amended image container.

### `ds.pp.add_observations(properties: Union[str, list, tuple]=('label', 'centroid'), layer_key: str='_segmentation', return_xarray: bool=False) -> xr.Dataset`

Adds properties derived from the segmentation mask to the image container.

**Parameters**

properties : Union[str, list, tuple]
    A list of properties to be added to the image container. See
    skimage.measure.regionprops_table for a list of available properties.
layer_key : str
    The key of the layer that contains the segmentation mask.
return_xarray : bool
    If true, the function returns an xarray.DataArray with the properties
    instead of adding them to the image container.

**Returns**

xr.Dataset
    The amended image container.

### `ds.pp.drop_observations(properties: Union[str, list, tuple], key: str='features') -> xr.Dataset`

### `ds.pp.add_feature(feature_name: str, feature_values: Union[list, np.ndarray])`

Adds a feature to the image container.

**Parameters**

feature_name : str
    The name of the feature to be added.
feature_values :
    The values of the feature to be added.

**Returns**

xr.Dataset
    The updated image container with the added feature.

### `ds.pp.add_obs_from_dataframe(df: pd.DataFrame) -> xr.Dataset`

Adds an observation table to the image container. Columns of the
dataframe have to match the feature coordinates of the image
container, and the index of the dataframe has to match the cell coordinates
of the image container.

**Parameters**

df : pd.DataFrame
    A dataframe with the observation values.

**Returns**

xr.Dataset
    The amended image container.

### `ds.pp.add_quantification(func: Union[str, Callable]='intensity_mean', key_added: str='_intensity', layer_key: str='_image', return_xarray=False) -> xr.Dataset`

Quantify channel intensities over the segmentation mask.

**Parameters**

func : Callable or str, optional
    The function used for quantification. Can either be a string to specify a function from skimage.measure.regionprops_table or a custom function. Default is 'intensity_mean'.
key_added : str, optional
    The key under which the quantification data will be stored in the image container. Default is '_intensity'.
layer_key : str, optional
    The key of the layer to be quantified. Default is '_image'.
return_xarray : bool, optional
    If True, the function returns an xarray.DataArray with the quantification data instead of adding it to the image container.

**Returns**

xr.Dataset or xr.DataArray
    The updated image container with added quantification data or the quantification data as a separate xarray.DataArray.

### `ds.pp.add_quantification_from_dataframe(df: pd.DataFrame, key_added: str='_intensity') -> xr.Dataset`

Adds an observation table to the image container. Columns of the
dataframe have to match the channel coordinates of the image
container, and the index of the dataframe has to match the cell coordinates
of the image container.

**Parameters**

df : pd.DataFrame
    A dataframe with the quantification values.
key_added : str, optional
    The key under which the quantification data will be added to the image container.

**Returns**

xr.Dataset
    The amended image container.

### `ds.pp.drop_layers(layers: Union[str, list, NoneType]=None, keep: Union[str, list, NoneType]=None, drop_obs: bool=True, suppress_warnings: bool=False) -> xr.Dataset`

Drops layers from the image container. Can either drop all layers specified in layers or drop all layers but the ones specified in keep.

**Parameters**

layers : Union[str, list]
    The name of the layer or a list of layer names to be dropped.
keep : Union[str, list]
    The name of the layer or a list of layer names to be kept.
drop_obs : bool
    If True, the observations are removed when the label or neighborhood properties are dropped. Default is True.
suppress_warnings : bool
    If True, warnings are suppressed. Default is False.

**Returns**

xr.Dataset
    The updated image container with dropped layers.

### `ds.pp.threshold(quantile: Union[float, list]=None, intensity: Union[int, list]=None, key_added: Optional[str]=None, channels: Union[str, list, NoneType]=None, shift: bool=True, **kwargs)`

Apply thresholding to the image layer of the object.
By default, shift is set to true. This means that the threshold value is subtracted from the image, and all negative values are set to 0.
If you instead want to set all values below the threshold to 0 while retaining the rest of the image at the original values, set shift to False.

**Parameters**

quantile : float
    The quantile value used for thresholding. If provided, the pixels below this quantile will be set to 0.
intensity : int
    The absolute intensity value used for thresholding. If provided, the pixels below this intensity will be set to 0.
key_added : Optional[str])
    The name of the new image layer after thresholding. If not provided, the original image layer will be replaced.
channels : Optional[Union[str, list]])
    The channels to apply the thresholding to. If None, the thresholding will be applied to all channels.
shift : bool
    If True, the thresholded image will be shifted so that values do not start at an arbitrary value. Default is True.

**Returns**

xr.Dataset
    The object with the thresholding applied to the image layer.

**Raises**

ValueError
    If both quantile and intensity are None or if both quantile and intensity are provided.

### `ds.pp.apply(func: Callable, key: str='_image', key_added: str='_image', **kwargs)`

Apply a function to each channel independently.

**Parameters**

func : Callable
    The function to apply to the layer.
key : str
    The key of the layer to apply the function to. Default is '_image'.
key_added : str
    The key under which the updated layer will be stored. Default is '_image' (i. e. the original image will be overwritten).
**kwargs : dict, optional
    Additional keyword arguments to pass to the function.

**Returns**

xr.Dataset
    The updated image container with the applied function.

### `ds.pp.normalize()`

Performs a percentile normalization on each channel using the 3- and 99.8-percentile. Resulting values are in the range of 0 to 1.

**Returns**

xr.Dataset
    The image container with the normalized image stored in '_plot'.

### `ds.pp.downsample(rate: int)`

Downsamples the entire dataset by selecting every `rate`-th element along the x and y dimensions.

**Parameters**

rate : int
    The downsampling rate. Only every `rate`-th pixel (or coordinate) is kept.

**Returns**

xr.Dataset
    The downsampled dataset with updated x and y coordinates.

### `ds.pp.rescale(scale: float)`

Rescales the image and segmentation mask in the object by a given scale.

The image is interpolated, while the segmentation mask is rescaled with nearest-neighbor interpolation so
that the cell ids are preserved. The x and y coordinates keep their origin and are spaced so that they still
cover the same region as before (analogous to `pp.downsample`). Cells that disappear when downscaling are
removed from all layers, and observations that can be derived from the segmentation mask (e.g. centroids or
area) are recomputed on the rescaled mask.

**Parameters**

scale : float
    The scale factor by which to rescale the image and segmentation mask.

**Returns**

xr.Dataset
    The rescaled dataset with updated image and segmentation mask.

### `ds.pp.filter_by_obs(col: str, func: Callable, segmentation_key: str='_segmentation', reindex: bool=False)`

Filter the object by observations based on a given feature and filtering function.

Parameters:
    col (str): The name of the feature to filter by.
    func (Callable): A filtering function that takes in the values of the feature and returns a boolean array.
    segmentation_key (str): The key of the segmentation mask in the object. Default is Layers.SEGMENTATION.
    reindex (bool): Whether to reindex the cell IDs after filtering. Default is False.

Returns:
    xr.Dataset: The filtered object with the selected cells and updated segmentation mask.

Raises:
    AssertionError: If the feature does not exist in the object's observations.

Notes:
    - This method filters the object by selecting only the cells that satisfy the filtering condition.

Example:
    To filter the object by the feature "area" and keep only the cells with an area greater than 70px:
    `obj = obj.pp.add_observations('area').pp.filter_by_obs('area', lambda x: x > 70)`

### `ds.pp.remove_outlying_cells(dilation_size: int=25, threshold: int=5, segmentation_key: str='_segmentation', reindex: bool=False)`

Removes outlying cells from the image container. It does so by dilating the segmentation mask and removing cells that belong to a connected component with less than 'threshold' cells.

**Parameters**

dilation_size : int
    The size of the dilation kernel. Default is 25.
threshold : int
    The minimum number of cells in a connected component required for the cells to be kept. Default is 5.
segmentation_key : str
    The key of the segmentation mask in the object. Default is '_segmentation'.
reindex : bool
    Whether to reindex the cell IDs after filtering. Default is False.

**Returns**

xr.Dataset
    The updated image container with the outlying cells removed.

**Raises**

ValueError
    If the object does not contain a segmentation mask.

### `ds.pp.grow_cells(iterations: int=2, suppress_warning: bool=False) -> xr.Dataset`

Grows the segmentation masks by expanding the labels in the object.

**Parameters**

iterations : int
    The number of iterations to grow the segmentation masks. Default is 2.
suppress_warning :bool
    Whether to suppress the warning about recalculating the observations. Used internally, default is False.

**Raises**

ValueError
    If the object does not contain a segmentation mask.

**Returns**

xr.Dataset
    The object with the grown segmentation masks and updated observations.

### `ds.pp.merge_segmentation(layer_key: str, key_added: str='_merged_segmentation', labels: Optional[List[str]]=None, threshold1: float=0.7, threshold2: float=0.8)`

Merge segmentation masks. Merging works as follows:
1. Assume two segmentation masks, S1 (higher priority, e.g., macrophages) and S2 (lower priority, e.g., lymphocytes)
2. We start by copying the segmentation regions from S2 into S1, but only for pixels where S1 has background
3. Some S2 cells may now be partially overwritten by S1 cells — we drop any S2 cell where less than threshold1
fraction of its original area survived (e.g. with threshold1=0.7,
an S2 cell that lost more than 30% of its area to S1 is discarded)
4. Of the remaining S2 cells, we additionally drop any that overlap with threshold2 or
more of any single S1 cell's area — this prevents S2 cells from being retained when they
largely encapsulate an S1 cell (e.g. with threshold2=0.8, an S2 cell is discarded if it
covers more than 80% of any S1 cell)
5. The surviving S1 and S2 regions are combined into a final mask with sequential labels

**Parameters**

layer_key : Union[str, List[str]]
    The key(s) of the segmentation mask(s) to merge. Can be a single key (must be 3D) or a list of keys (each 2D).
key_added : str
    The name of the new segmentation mask to be added to the xarray object. Default is "_merged_segmentation".
labels : Optional[List[str]]
    Optional. Labels corresponding to each segmentation mask. If provided, must match number of arrays.
threshold1: float
    Optional. Threshold for merging cells. Default is 0.7.
threshold2: float
    Optional. Threshold for merging cells. Default is 0.8.

**Returns**

xr.Dataset
    The xarray object with the merged segmentation mask.

**Raises**

    AssertionError
        If specified keys are not found or other input inconsistencies exist.

**Notes**

    - If the input array is 2D, it will be expanded to 3D.
    - If labels are provided, they need to match the number of arrays.
    - The merging process starts with merging the biggest cells first, then the smaller ones.
    - Disconnected cells in the input are handled based on the specified method.

### `ds.pp.get_layer_as_df(layer: str='_obs', celltypes_to_str: bool=True, neighborhoods_to_str: bool=True, idx_to_str: bool=False) -> pd.DataFrame`

Returns the specified layer as a pandas DataFrame.

Parameters:
    layer (str): The name of the layer to retrieve. Defaults to Layers.OBS.
    celltypes_to_str (bool): Whether to convert celltype labels to strings. Defaults to True.
    neighborhoods_to_str (bool): Whether to convert neighborhood labels to strings. Defaults to True.
    idx_to_str (bool): Whether to convert the index to strings. Defaults to False.

Returns:
    pandas.DataFrame: The layer data as a DataFrame.

### `ds.pp.get_disconnected_cell() -> int`

Returns the first disconnected cell from the segmentation layer.

Returns:
    np.ndarray: The first disconnected cell from the segmentation layer.

### `ds.pp.transform_expression_matrix(method: str='arcsinh', key: str='_intensity', key_added: str='_intensity', cofactor: float=5.0, min_percentile: float=1.0, max_percentile: float=99.0, **kwargs)`

Transforms the expression matrix based on the specified mode.

Parameters:
    method (str): The transformation method. Available options are "arcsinh", "zscore", "minmax", "double_zscore", and "clip".
    key (str): The key of the expression matrix in the object.
    key_added (str): The key to assign to the transformed matrix in the object.
    cofactor (float): The cofactor to use for the "arcsinh" transformation.
    min_percentile (float): The minimum percentile value to use for the "clip" transformation.
    max_percentile (float): The maximum percentile value to use for the "clip" transformation.

Returns:
    xr.Dataset: The object with the transformed matrix added.

Raises:
    ValueError: If an unknown transformation mode is specified.
    AssertionError: If no expression matrix is found at the specified layer.

### `ds.pp.mask_region(key: str='_mask', image_key='_image', key_added='_image') -> xr.Dataset`

Mask a region in the image.

Parameters:
    key (str): The key of the region to mask.
    image_key (str): The key of the image layer in the object. Default is Layers.IMAGE.
    key_added (str): The key to assign to the masked image in the object. Default is Layers.IMAGE, which overwrites the original image.

Returns:
    xr.Dataset: The object with the masked region in the image.

### `ds.pp.mask_cells(mask_key: str='_mask', segmentation_key='_segmentation', reindex: bool=False) -> xr.Dataset`

Mask cells in the segmentation mask.

Parameters:
    mask_key (str): The key of the mask to use for masking.
    segmentation_key (str): The key of the segmentation mask in the object. Default is Layers.SEGMENTATION.
    reindex (bool): Whether to reindex the cell IDs after filtering. Default is False.

Returns:
    xr.Dataset: The object with the masked cells in the segmentation mask.

### `ds.pp.convert_to_8bit(key: str='_image', key_added: str='_image')`

Convert the image to 8-bit.

Parameters:
    key (str): The key of the image layer in the object. Default is '_image'.
    key_added (str): The key to assign to the 8-bit image in the object. Default is '_image', which overwrites the original image.

Returns:
    xr.Dataset: The object with the image converted to 8-bit.

### `ds.pp.merge_channels(channels: List[str], key_added: str='merged_channel', normalize: bool=True, method: Union[str, Callable]='max', layer_key: str='_image') -> xr.Dataset`

Merge specified channels into a single channel by summing their values.

**Parameters**

channels : List[str]
    The list of channel names to merge.
key_added : str
    The name of the new channel.
normalize : bool
    Whether to normalize the images before merging. Default is True.
method : Union[str, Callable]
    The method to use for merging. Can be "max", "sum", "mean", or a custom callable function. Default is "max".
layer_key : str
    The key of the image layer in the object. Default is Layers.IMAGE.

**Returns**

xr.Dataset
    The object with the merged channels in the image layer.

### `ds.pp.find_bright_cells(channels: Union[str, list, NoneType]=None, frac_channels: float=0.5, quantile: float=0.95, key: str='_intensity', key_added: str='is_bright') -> xr.Dataset`

Flags cells that are bright across multiple channels (e.g., due to a lot of antibodies binding to that cell).
A cell is considered bright in a channel if its intensity exceeds the given quantile threshold
for that channel. Cells that are bright in at least frac_channels of the
considered channels are flagged.

**Parameters**

channels : Optional[Union[str, list]]
    The channels to consider. If None, all channels are used.
frac_channels : float
    The fraction of channels in which a cell must be bright to be flagged.
    Default is 0.5.
quantile : float
    The quantile threshold above which a cell is considered bright in a
    given channel. Computed per channel. Default is 0.95.
key : str
    The key of the intensity layer to use. Default is Layers.INTENSITY.
key_added : str
    The name of the feature added to obs. Default is 'is_bright'.

**Returns**

xr.Dataset
    The updated image container with the 'is_bright' flag added to obs.

**Raises**

ValueError
    If no intensity matrix is found at the specified layer.

**Example**

Flag cells that are bright in at least 50% of all channels at the 95th percentile:
>>> ds = ds.pp.find_bright_cells()

Flag cells that are bright in at least 2 out of 3 specific channels:
>>> ds = ds.pp.find_bright_cells(channels=['CD3', 'CD8', 'CD20'], frac_channels=0.66)

## Functions for the spatialdata backend (`sp.pp`)

These functions take a `spatialdata.SpatialData` object as first argument and modify it in place unless `copy=True`.

### `sp.pp.add_quantification(sdata, func: Union[str, Callable]='intensity_mean', key_added: str='table', image_key: str='image', segmentation_key: str='segmentation', layer_key: Optional[str]=None, data_key: Optional[str]=None, copy: bool=False, **kwargs)`

This function computes the quantification of the image data based on the provided segmentation masks.
It extracts the image data and segmentation masks from the spatialdata object, applies the quantification function,
and adds the quantification results to the spatialdata object.
The quantification results are stored in an AnnData object, which is added to the tables attribute of the spatialdata object.
The quantification function can be specified as a string or a callable function.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the image data and segmentation masks.
    func (Union[str, Callable], optional): The quantification function to be applied. Defaults to "intensity_mean". Can be a string or a callable function.
    key_added (str, optional): The key under which the quantification results will be stored in the tables attribute of the spatialdata object. Defaults to table.
    image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
    segmentation_key (str, optional): The key for the segmentation masks in the spatialdata object. Defaults to segmentation.
    layer_key (Optional[str], optional): The key for the quantification results in the AnnData object. If None, a new layer will be created. Defaults to None.
    data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.

### `sp.pp.add_observations(sdata, properties: Union[str, list, tuple]=('label', 'centroid'), segmentation_key: str='segmentation', table_key: str='table', copy: bool=False, **kwargs)`

This function computes the observations for each region in the segmentation masks.
It extracts the segmentation masks from the spatialdata object, computes the region properties,
and adds the observations to the AnnData object stored in the tables attribute of the spatialdata object.
The observations are computed using the regionprops_table function from skimage.measure.
The properties to be computed can be specified as a string or a list/tuple of strings.
The default properties are "label" and "centroid", but other properties can be added as well.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the segmentation masks.
    properties (Union[str, list, tuple], optional): The properties to be computed for each region. Defaults to ("label", "centroid").
    segmentation_key (str, optional): The key for the segmentation masks in the spatialdata object. Defaults to segmentation.
    table_key (str, optional): The key under which the AnnData object is stored in the tables attribute of the spatialdata object. Defaults to table.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.

### `sp.pp.apply(sdata, func: Callable, key_added: str='image', image_key: str='image', data_key: Optional[str]=None, copy: bool=False, **kwargs)`

This function applies a given function to the image data in the spatialdata object.
It extracts the image data from the spatialdata object, applies the function,
and adds the processed image back to the spatialdata object.
The processed image is stored in the images attribute of the spatialdata object.
The function can be any callable function that takes an image as input and returns a processed image.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
    func (Callable): The function to be applied to the image data. It should take an image as input and return a processed image.
    image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
    data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    **kwargs: Additional keyword arguments to be passed to the function.

### `sp.pp.threshold(sdata, image_key: str='image', quantile: Union[float, list]=None, intensity: Union[int, list]=None, key_added: str='image', channels: Union[str, list, NoneType]=None, shift: bool=True, copy: bool=False, **kwargs)`

This function applies a threshold to the image data in the spatialdata object.
It extracts the image data from the spatialdata object, applies the thresholding function,
and adds the processed image back to the spatialdata object.
The processed image is stored in the images attribute of the spatialdata object.
The thresholding function can be specified using the quantile or intensity parameters.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
    image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
    quantile (Union[float, list], optional): The quantile value(s) to be used for thresholding. If None, the intensity parameter will be used. Defaults to None.
    intensity (Union[int, list], optional): The intensity value(s) to be used for thresholding. If None, the quantile parameter will be used. Defaults to None.
    key_added (str, optional): The key under which the processed image will be stored in the images attribute of the spatialdata object. Defaults to image.
    channels (Optional[Union[str, list]], optional): The channel(s) to be used for thresholding. If None, all channels will be used. Defaults to None.
    shift (bool, optional): Whether to shift the intensities towards 0 after thresholding. Defaults to True.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    **kwargs: Additional keyword arguments to be passed to the thresholding function.

### `sp.pp.transform_expression_matrix(sdata, method: str='arcsinh', table_key: str='table', cofactor: float=5.0, min_percentile: float=1.0, max_percentile: float=99.0, copy: bool=False, **kwargs)`

This function applies a transformation to the expression matrix in the spatialdata object.
It extracts the expression matrix from the spatialdata object, applies the transformation function,
and adds the transformed expression matrix back to the spatialdata object.
The transformed expression matrix is stored in the tables attribute of the spatialdata object.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
    method (str, optional): The transformation method to be applied. Defaults to "arcsinh".
    table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
    cofactor (float, optional): The cofactor to be used for the transformation. Defaults to 5.0.
    min_percentile (float, optional): The minimum percentile to be used for the transformation. Defaults to 1.0.
    max_percentile (float, optional): The maximum percentile to be used for the transformation. Defaults to 99.0.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    **kwargs: Additional keyword arguments to be passed to the transformation function.

### `sp.pp.filter_by_obs(sdata, col: str, func: Callable, segmentation_key: str='segmentation', table_key: str='table', reindex: bool=False, copy: bool=False)`

Filter the object by observations based on a given feature and filtering function.

Parameters:
    sdata (spatialdata.SpatialData): The spatialdata object to filter.
    col (str): The name of the feature to filter by.
    func (Callable): A filtering function that takes in the values of the feature and returns a boolean array.
    segmentation_key (str): The key of the segmentation mask in the object. Default is SDLayers.SEGMENTATION.
    table_key (str): The key of the table in the object. Default is SDLayers.TABLE.
    reindex (bool): Whether to reindex the obs after filtering. Default is False.
    copy (bool): If True, a copy of the object is returned. Default is False.

### `sp.pp.grow_cells(sdata, iterations: int=2, segmentation_key: str='segmentation', table_key: str='table', suppress_warning: bool=False, copy: bool=False) -> xr.Dataset`

Grows the segmentation masks by expanding the labels in the object.

**Parameters**

sdata : spatialdata.SpatialData
    The spatialdata object containing the segmentation masks.
iterations : int
    The number of iterations to grow the segmentation masks. Default is 2.
segmentation_key : str
    The key of the segmentation mask in the object. Default is segmentation.
suppress_warning :bool
    Whether to suppress the warning about recalculating the observations. Used internally, default is False.
copy : bool
    If True, a copy of the object is returned. Default is False.

**Raises**

ValueError
    If the object does not contain a segmentation mask.

**Returns**

xr.Dataset
    The object with the grown segmentation masks and updated observations.

### `sp.pp.merge_channels(sdata, channels: List[str], key_added: str, normalize: bool=True, method: str='sum', image_key: str='image', copy: bool=False)`

Merges multiple channels into a single channel in the spatialdata object.

**Parameters**

sdata : spatialdata.SpatialData
    The spatialdata object containing the image data.
channels : List[str]
    The list of channels to be merged.
key_added : str
    The key under which the merged channel will be stored in the images attribute of the spatialdata object.
normalize : bool
    Whether to normalize the channels before merging. Default is True.
method : str
    The method to use for merging the channels. Options are "sum", "mean", and "max". Default is "sum".
image_key : str
    The key for the image data in the spatialdata object. Default is SDLayers.IMAGE.
copy : bool
    If True, a copy of the object is returned. Default is False.

**Raises**

AssertionError
    If key_added already exists in the object.
KeyError
    If any of the specified channels do not exist in the image data.
ValueError
    If less than two channels are provided to merge.
    If an unknown merging method is provided.

**Returns**

xr.Dataset
    The object with the merged channel added.
