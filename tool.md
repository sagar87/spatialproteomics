# The tool (`tl`) accessor

The tool accessor contains methods that utilize third-party tools to enable operations such as segmentation or cell type prediction.

## `ds.tl` (ToolAccessor)

The tool accessor enables the application of external tools such as StarDist or Astir.

Available on any `xarray.Dataset` after `import spatialproteomics as sp`, as `ds.tl`.

### `ds.tl.cellpose(channel: Optional[str]=None, key_added: str='_segmentation', diameter: float=None, channel_settings: list=None, num_iterations: int=2000, cellprob_threshold: float=0.0, flow_threshold: float=0.4, batch_size: int=8, gpu: bool=True, model_type: str='cyto3', pretrained_model: str='cpsam', postprocess_func: Callable=<lambda>, return_diameters: bool=False, **kwargs)`

Segment cells using Cellpose. Adds a layer to the spatialproteomics object
with dimension (X, Y) or (C, X, Y) dependent on whether channel argument
is specified or not.

**Parameters**

channel : str, optional
    Channel to use for segmentation. If None, all channels are used for independent segmentation.
key_added : str, optional
    Key to assign to the segmentation results.
diameter : float, optional
    Expected cell diameter in pixels.
channel_settings : List[int], optional
    Channels for Cellpose to use for segmentation. If [0, 0], independent segmentation is performed on all channels. If it is anything else (e. g. [1, 2]), joint segmentation is attempted.
num_iterations : int, optional
    Maximum number of iterations for segmentation.
cellprob_threshold : float, optional
    Threshold for cell probability.
flow_threshold : float, optional
    Threshold for flow.
batch_size : int, optional
    Batch size for segmentation.
gpu : bool, optional
    Whether to use GPU for segmentation.
model_type : str, optional
    Type of Cellpose model to use.
pretrained_model : str, optional
    Pretrained model to use for Cellpose (4+).
postprocess_func : Callable, optional
    Function to apply to the segmentation masks after prediction.
return_diameters : bool, optional
    Whether to return the cell diameters.

**Returns**

xr.Dataset
    Dataset containing original data and segmentation mask.

**Notes**

This method requires the 'cellpose' package to be installed.

### `ds.tl.stardist(channel: Optional[str]=None, key_added: str='_segmentation', scale: float=3, n_tiles: int=12, normalize: bool=True, predict_big: bool=False, postprocess_func: Callable=<lambda>, **kwargs) -> xr.Dataset`

Apply StarDist algorithm to perform instance segmentation on the nuclear image.

**Parameters**

channel : str, optional
    Channel to use for segmentation. If None, all channels are used.
key_added : str, optional
    Key to write the segmentation results to.
scale : float, optional
    Scaling factor for the StarDist model (default is 3).
n_tiles : int, optional
    Number of tiles to split the image into for prediction (default is 12).
normalize : bool, optional
    Flag indicating whether to normalize the nuclear image (default is True).
nuclear_channel : str, optional
    Name of the nuclear channel in the image (default is "DAPI").
predict_big : bool, optional
    Flag indicating whether to use the 'predict_instances_big' method for large images (default is False).
postprocess_func : Callable, optional
    Function to apply to the segmentation masks after prediction (default is lambda x: x).
**kwargs : dict, optional
    Additional keyword arguments to be passed to the StarDist prediction method.

**Returns**

obj : xr.Dataset
    Xarray dataset containing the segmentation mask and centroids.

**Raises**

ValueError
    If the object already contains a segmentation mask.

### `ds.tl.mesmer(key_added: str='_segmentation', channel: Optional[List]=None, postprocess_func: Callable=<lambda>, **kwargs)`

Segment cells using Mesmer. Adds a layer to the spatialproteomics object
with dimension (C, X, Y). Assumes C is two and has the order (nuclear, membrane).

**Parameters**

key_added : str, optional
    Key to assign to the segmentation results.
channel : List, optional
    Channel to use for segmentation. If None, all channels are used.
postprocess_func : Callable, optional
    Function to apply to the segmentation masks after prediction.

**Returns**

xr.Dataset
    Dataset containing original data and segmentation mask.

**Notes**

This method requires the 'mesmer' package to be installed.

### `ds.tl.astir(marker_dict: dict, key: str='_intensity', threshold: float=0, seed: int=42, learning_rate: float=0.001, batch_size: float=64, n_init: int=5, n_init_epochs: int=5, max_epochs: int=500, cell_id_col: str='cell_id', cell_type_col: str='cell_type', **kwargs)`

This method predicts cell types from an expression matrix using the Astir algorithm.

**Parameters**

marker_dict : dict
    Dictionary mapping markers to cell types. Can also include cell states. Example: {"cell_type": {'B': ['PAX5'], 'T': ['CD3'], 'Myeloid': ['CD11b']}}
key : str, optional
    Layer to use as expression matrix.
threshold : float, optional
    Certainty threshold for astir to assign a cell type. Defaults to 0.
seed : int, optional
    Random seed. Defaults to 42.
learning_rate : float, optional
    Learning rate. Defaults to 0.001.
batch_size : float, optional
    Batch size. Defaults to 64.
n_init : int, optional
    Number of initializations. Defaults to 5.
n_init_epochs : int, optional
    Number of epochs for each initialization. Defaults to 5.
max_epochs : int, optional
    Maximum number of epochs. Defaults to 500.
cell_id_col : str, optional
    Column name for cell IDs. Defaults to "cell_id".
cell_type_col : str, optional
    Column name for cell types. Defaults to "cell_type".

**Raises**

ValueError
    If no expression matrix was present or the image is not of type uint8.

**Returns**

DataArray
    A DataArray with the assigned cell types.

### `ds.tl.spotiflow(channel: Optional[str]=None, image_key: str='_image', segmentation_key: str='_segmentation', key_added_suffix: str='_spots', pretrained_name: str='general', return_spots: bool=False, model_kwargs: Optional[dict]={'verbose': False}, predict_kwargs: Optional[dict]={'verbose': False})`

This method detects spots in the image using the Spotiflow algorithm and counts the number of spots per cell, per channel.

**Parameters**

channel : str, optional
    Channel to use for spot detection. If None, all channels are used.
image_key : str, optional
    Key to the image layer in the spatialproteomics object. Default is 'image'.
segmentation_key : str, optional
    Key to the segmentation layer in the spatialproteomics object. Default is 'segmentation'.
key_added_suffix : str, optional
    Suffix to add to the channel names in the resulting counts. Default is '_spots'.
pretrained_name : str, optional
    Name of the pretrained model to use for spot detection. Default is 'general'.
return_spots : bool, optional
    Whether to return the detected spots as a DataFrame. Default is False.
model_kwargs : dict, optional
    Additional keyword arguments to pass to the Spotiflow model. Default is {'verbose': False}.
predict_kwargs : dict, optional
    Additional keyword arguments to pass to the Spotiflow prediction method. Default is {'verbose': False}.

**Returns**

xr.Dataset
    Dataset containing the counts of spots per cell, per channel. If return_spots is True, also returns a DataFrame of the detected spots with their coordinates and assigned cell IDs.

### `ds.tl.convert_to_anndata(expression_matrix_key: str='_intensity', obs_key: str='_obs', additional_layers: Optional[dict]=None, additional_obsm: Optional[dict]=None, additional_uns: Optional[dict]=None)`

Convert the spatialproteomics object to an anndata.AnnData object. The resulting AnnData object does not store the original image or segmentation mask.

**Parameters**

expression_matrix_key : str, optional
    The key of the expression matrix in the spatialproteomics object. Default is '_intensity'.
obs_key : str, optional
    The key of the observation data in the spatialproteomics object. Default is '_obs'.
additional_layers : dict, optional
    Additional layers to include in the anndata.AnnData object. The keys are the names of the layers and the values are the corresponding keys in the spatialproteomics object.
additional_obsm : dict, optional
    Additional obsm data to include in the anndata.AnnData object. The keys are the names of the obsm data and the values are the corresponding keys in the spatialproteomics object.
additional_uns : dict, optional
    Additional uns data to include in the anndata.AnnData object. The keys are the names of the uns data and the values are the corresponding keys in the spatialproteomics object.

**Returns**

adata : anndata.AnnData
    The converted anndata.AnnData object.

**Raises**

AssertionError
    If additional layers, obsm entries, or uns entries are not found in the spatialproteomics object, or if additional layers or obsm entries have incompatible shapes.

**Notes**

- The expression matrix is extracted from the spatialproteomics object using the provided expression matrix key.
- If additional layers are specified, they are extracted from the spatialproteomics object and added to the anndata.AnnData object.
- If obs_key is present in the spatialproteomics object, it is used to create the obs DataFrame of the anndata.AnnData object.
- If additional_uns is specified, the corresponding uns data is extracted from the spatialproteomics object and added to the anndata.AnnData object.

### `ds.tl.convert_to_spatialdata(image_key: str='_image', segmentation_key: str='_segmentation', **kwargs)`

Convert the spatialproteomics object to a spatialdata object.

Parameters:
    image_key (str): The key of the image data in the object. Defaults to Layers.IMAGE.
    segmentation_key (str): The key of the segmentation data in the object. Defaults to Layers.SEGMENTATION.
    **kwargs: Additional keyword arguments to be passed to the convert_to_anndata method.

Returns:
    spatial_data_object (spatialdata.SpatialData): The converted spatialdata object.

## Functions for the spatialdata backend (`sp.tl`)

These functions take a `spatialdata.SpatialData` object as first argument and modify it in place unless `copy=True`.

### `sp.tl.cellpose(sdata, channel: Optional[str]=None, image_key: str='image', key_added: str='segmentation', data_key: Optional[str]=None, copy: bool=False, **kwargs)`

This function runs the cellpose segmentation algorithm on the provided image data.
It extracts the image data from the spatialdata object, applies the cellpose algorithm,
and adds the segmentation masks to the spatialdata object.
The segmentation masks are stored in the labels attribute of the spatialdata object.
The function also handles multiple channels by iterating over the channels and applying the segmentation algorithm to each channel separately.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
    channel (Optional[str]): The channel(s) to be used for segmentation. If None, all channels will be used.
    image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
    key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
    data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    **kwargs: Additional keyword arguments to be passed to the cellpose algorithm.

### `sp.tl.stardist(sdata, channel: Optional[str]=None, image_key: str='image', key_added: str='segmentation', data_key: Optional[str]=None, copy: bool=False, **kwargs)`

This function runs the stardist segmentation algorithm on the provided image data.
It extracts the image data from the spatialdata object, applies the stardist algorithm,
and adds the segmentation masks to the spatialdata object.
The segmentation masks are stored in the labels attribute of the spatialdata object.
The function also handles multiple channels by iterating over the channels and applying the segmentation algorithm to each channel separately.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
    channel (Optional[str]): The channel(s) to be used for segmentation. If None, all channels will be used.
    image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
    key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
    data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    **kwargs: Additional keyword arguments to be passed to the stardist algorithm.

### `sp.tl.mesmer(sdata, channel: Optional[str]=None, image_key: str='image', key_added: str='segmentation', data_key: Optional[str]=None, copy: bool=False, **kwargs)`

This function runs the mesmer segmentation algorithm on the provided image data.
It extracts the image data from the spatialdata object, applies the mesmer algorithm,
and adds the segmentation masks to the spatialdata object.
The segmentation masks are stored in the labels attribute of the spatialdata object.
The first channel is assumed to be nuclear and the second one membraneous.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
    channel (Optional[str]): The channel(s) to be used for segmentation.
    image_key (str, optional): The key for the image data in the spatialdata object. Defaults to image.
    key_added (str, optional): The key under which the segmentation masks will be stored in the labels attribute of the spatialdata object. Defaults to segmentation.
    data_key (Optional[str], optional): The key for the image data in the spatialdata object. If None, the image_key will be used. Defaults to None.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    **kwargs: Additional keyword arguments to be passed to the mesmer algorithm.

### `sp.tl.astir(sdata, marker_dict: dict, table_key='table', threshold: float=0, seed: int=42, learning_rate: float=0.001, batch_size: float=64, n_init: int=5, n_init_epochs: int=5, max_epochs: int=500, cell_id_col: str='cell_id', cell_type_col: str='cell_type', copy: bool=False, **kwargs)`

This function applies the ASTIR algorithm to predict cell types based on the expression matrix.
It extracts the expression matrix from the spatialdata object, applies the ASTIR algorithm,
and adds the predicted cell types to the spatialdata object.
The predicted cell types are stored in the obs attribute of the AnnData object in the tables attribute of the spatialdata object.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the expression matrix.
    marker_dict (dict): A dictionary containing the marker genes for each cell type.
    table_key (str, optional): The key under which the expression matrix is stored in the tables attribute of the spatialdata object. Defaults to "table".
    threshold (float, optional): The threshold value to be used for the ASTIR algorithm. Defaults to 0.
    seed (int, optional): The random seed to be used for the ASTIR algorithm. Defaults to 42.
    learning_rate (float, optional): The learning rate to be used for the ASTIR algorithm. Defaults to 0.001.
    batch_size (float, optional): The batch size to be used for the ASTIR algorithm. Defaults to 64.
    n_init (int, optional): The number of initializations to be used for the ASTIR algorithm. Defaults to 5.
    n_init_epochs (int, optional): The number of initial epochs to be used for the ASTIR algorithm. Defaults to 5.
    max_epochs (int, optional): The maximum number of epochs to be used for the ASTIR algorithm. Defaults to 500.
    cell_id_col (str, optional): The name of the column containing the cell IDs in the expression matrix. Defaults to "cell_id".
    cell_type_col (str, optional): The name of the column containing the cell types in the expression matrix. Defaults to "cell_type".
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.
    **kwargs: Additional keyword arguments to be passed to the ASTIR algorithm.

### `sp.tl.spotiflow(sdata, channel: Optional[str]=None, image_key: str='image', segmentation_key: str='segmentation', table_key: str='table', key_added: str='spotiflow', key_added_suffix: str='_spots', pretrained_name: str='general', model_kwargs: Optional[dict]={'verbose': False}, predict_kwargs: Optional[dict]={'verbose': False}, copy: bool=False)`

This function applies the Spotiflow algorithm to detect spots in the provided image data.
It extracts the image data from the spatialdata object, applies the Spotiflow algorithm,
and adds the detected spots to the spatialdata object.
The detected spots are stored in the points attribute of the spatialdata object.

Args:
    sdata (spatialdata.SpatialData): The spatialdata object containing the image data.
    channel (Optional[str]): The channel(s) to be used for spot detection. If None, all channels will be used.
    image_key (str, optional): The key for the image data in the spatialdata object. Defaults to "image".
    segmentation_key (str, optional): The key for the segmentation mask in the spatialdata object. Defaults to "segmentation".
    table_key (str, optional): The key for the table in the spatialdata object. Defaults to "table".
    key_added (str, optional): The key under which the detected spots will be stored in the points attribute of the spatialdata object. Defaults to "spotiflow".
    key_added_suffix (str, optional): The suffix to be added to the channel names in the detected spots. Defaults to "_spots".
    pretrained_name (str, optional): The name of the pretrained model to be used for spot detection. Defaults to "general".
    model_kwargs (Optional[dict], optional): Additional keyword arguments to be passed to the Spotiflow model. Defaults to {'verbose': False}.
    predict_kwargs (Optional[dict], optional): Additional keyword arguments to be passed to the Spotiflow prediction method. Defaults to {'verbose': False}.
    copy (bool, optional): Whether to create a copy of the spatialdata object. Defaults to False.

Returns:
    spatialdata.SpatialData: The spatialdata object with the detected spots added to the points attribute.
