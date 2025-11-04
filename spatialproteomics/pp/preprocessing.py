import copy as cp
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import skimage
import xarray as xr
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

from ..base_logger import logger
from ..constants import Dims, Features, Layers, Props, SDFeatures, SDLayers
from ..sd.utils import _process_adata, _process_image, _process_segmentation
from .utils import (
    _apply,
    _compute_quantification,
    _convert_to_8bit,
    _get_disconnected_cell,
    _merge_channels,
    _merge_segmentation,
    _normalize,
    _relabel_cells,
    _remove_outlying_cells,
    _remove_unlabeled_cells,
    _threshold,
    _transform_expression_matrix,
    _validate_and_clamp_slice,
)


# === SPATIALDATA PREPROCESSING ===
def add_quantification(
    sdata,
    func: Union[str, Callable] = "intensity_mean",
    key_added: str = SDLayers.TABLE,
    image_key: str = SDLayers.IMAGE,
    segmentation_key: str = SDLayers.SEGMENTATION,
    layer_key: Optional[str] = None,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
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
    """
    from anndata import AnnData

    if copy:
        sdata = cp.deepcopy(sdata)

    # sanity checks for image and segmentation
    image = _process_image(
        sdata, image_key=image_key, channels=None, key_added=None, data_key=data_key, return_values=False
    )
    segmentation = _process_segmentation(sdata, segmentation_key)

    # computing the quantification
    measurements, cell_idx = _compute_quantification(image.values, segmentation, func)

    # checking if there already is an anndata object in the spatialdata object
    if key_added in sdata.tables:
        # if an anndata object (and hence a quantification) already exists, we add the new quantification to a new layer
        assert (
            layer_key is not None
        ), "An expression matrix already exists in your spatialdata object. Please provide a layer_key to add the new quantification to."
        assert (
            layer_key not in sdata.tables[key_added].layers
        ), f"Layer {layer_key} already exists in spatial data object. Please choose another key."
        sdata.tables[key_added].layers[layer_key] = measurements.T
    else:
        # if there is no anndata object yet, we create one
        adata = AnnData(measurements.T)
        adata.obs[SDFeatures.ID] = cell_idx
        adata.obs[SDFeatures.REGION] = segmentation_key
        adata.var_names = image.coords["c"].values
        # to be consistent with anndata standards, we add the Cell_ prefix to the obs_names
        adata.obs_names = [f"Cell_{x}" for x in cell_idx]

        # adding uns
        adata.uns = {
            "spatialdata_attrs": {
                "region": SDLayers.SEGMENTATION,
                "region_key": SDFeatures.REGION,
                "instance_key": SDFeatures.ID,
            }
        }

        # putting the anndata object into the spatialdata object
        sdata.tables[key_added] = adata

    if copy:
        return sdata


def add_observations(
    sdata,
    properties: Union[str, list, tuple] = ("label", "centroid"),
    segmentation_key: str = SDLayers.SEGMENTATION,
    table_key: str = SDLayers.TABLE,
    copy: bool = False,
    **kwargs,
):
    """
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
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    segmentation = _process_segmentation(sdata, segmentation_key)
    adata = _process_adata(sdata, table_key=table_key)
    existing_features = adata.obs.columns

    if type(properties) is str:
        properties = [properties]

    if "label" not in properties:
        properties = ["label", *properties]

    table = regionprops_table(segmentation, properties=properties)

    # remove existing features
    table = pd.DataFrame({k: v for k, v in table.items() if k not in existing_features})

    # setting the label to be the index and removing it from the table
    table.index = table["label"]
    table = table.drop(columns="label")

    # add data into adata.obs
    adata.obs = adata.obs.merge(table, left_on="id", right_index=True, how="left")

    if copy:
        return sdata


def apply(
    sdata,
    func: Callable,
    key_added: str = SDLayers.IMAGE,
    image_key: str = SDLayers.IMAGE,
    data_key: Optional[str] = None,
    copy: bool = False,
    **kwargs,
):
    """
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
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    image = _process_image(
        sdata, image_key=image_key, channels=None, key_added=None, data_key=data_key, return_values=False
    )
    processed_image = _apply(image.values, func, **kwargs)
    channels = image.coords["c"].values

    # get transformations
    transformation = get_transformation(sdata.images[image_key])
    # add the image to the spatial data object
    sdata.images[key_added] = spatialdata.models.Image2DModel.parse(
        processed_image, c_coords=channels, transformations=None, dims=("c", "y", "x")
    )
    set_transformation(sdata.images[key_added], transformation)

    if copy:
        return sdata


def threshold(
    sdata,
    image_key: str = SDLayers.IMAGE,
    quantile: Union[float, list] = None,
    intensity: Union[int, list] = None,
    key_added: str = SDLayers.IMAGE,
    channels: Optional[Union[str, list]] = None,
    shift: bool = True,
    copy: bool = False,
    **kwargs,
):
    """
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
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    # this gets the image as an xarray object
    image = _process_image(sdata, image_key=image_key, channels=None, key_added=None, return_values=False)
    processed_image = _threshold(
        image, quantile=quantile, intensity=intensity, channels=channels, shift=shift, channel_coord="c", **kwargs
    )
    channels = sdata.images[image_key].coords["c"].values

    # get transformations
    transformation = get_transformation(sdata.images[image_key])

    # add the image to the spatial data object
    sdata.images[key_added] = spatialdata.models.Image2DModel.parse(
        processed_image, c_coords=channels, transformations=None, dims=("c", "y", "x")
    )
    set_transformation(sdata.images[key_added], transformation)

    if copy:
        return sdata


def transform_expression_matrix(
    sdata,
    method: str = "arcsinh",
    table_key: str = SDLayers.TABLE,
    cofactor: float = 5.0,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
    copy: bool = False,
    **kwargs,
):
    """
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
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    adata = _process_adata(sdata, table_key=table_key)
    transformed_matrix = _transform_expression_matrix(
        adata.X,
        method=method,
        cofactor=cofactor,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        **kwargs,
    )
    adata.X = transformed_matrix

    if copy:
        return sdata


def filter_by_obs(
    sdata,
    col: str,
    func: Callable,
    segmentation_key: str = SDLayers.SEGMENTATION,
    table_key: str = SDLayers.TABLE,
    copy: bool = False,
):
    """
    Filter the object by observations based on a given feature and filtering function.

    Parameters:
        sdata (spatialdata.SpatialData): The spatialdata object to filter.
        col (str): The name of the feature to filter by.
        func (Callable): A filtering function that takes in the values of the feature and returns a boolean array.
        segmentation_key (str): The key of the segmentation mask in the object. Default is SDLayers.SEGMENTATION.
        table_key (str): The key of the table in the object. Default is SDLayers.TABLE.
        copy (bool): If True, a copy of the object is returned. Default is False.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    segmentation = _process_segmentation(sdata, segmentation_key)
    adata = _process_adata(sdata, table_key=table_key)
    existing_features = adata.obs.columns

    # checking if the feature exists in obs
    assert col in existing_features, f"Feature {col} not found in obs. You can add it with pp.add_observations()."

    # select the right column from the observations
    cells = adata.obs[col].values.copy()
    cells_bool = func(cells)
    cells_sel = adata.obs.loc[cells_bool, SDFeatures.ID].values

    # setting all cells that are not in cells to 0
    segmentation = _remove_unlabeled_cells(segmentation, cells_sel)
    # relabeling cells in the segmentation mask so the IDs go from 1 to n again
    segmentation, relabel_dict = _relabel_cells(segmentation)

    # removing the cells which are not in cells_sel
    adata = adata[adata.obs[SDFeatures.ID].isin(cells_sel), :].copy()
    # updating the cell coords of the object
    adata.obs[SDFeatures.ID] = [relabel_dict[cell] for cell in adata.obs[SDFeatures.ID].values]
    adata.obs.index = [f"Cell_{x}" for x in adata.obs[SDFeatures.ID].values]

    # overwriting the segmentation mask in the object
    # get transformations
    transformation = get_transformation(sdata[segmentation_key])
    # add the segmentation masks to the spatial data object
    sdata.labels[segmentation_key] = spatialdata.models.Labels2DModel.parse(
        segmentation, transformations=None, dims=("y", "x")
    )
    set_transformation(sdata.labels[segmentation_key], transformation)

    # overwriting the anndata object in the object
    sdata.tables[table_key] = adata

    if copy:
        return sdata


def grow_cells(
    sdata,
    iterations: int = 2,
    segmentation_key: str = SDLayers.SEGMENTATION,
    table_key: str = SDLayers.TABLE,
    suppress_warning: bool = False,
    copy: bool = False,
) -> xr.Dataset:
    """
    Grows the segmentation masks by expanding the labels in the object.

    Parameters
    ----------
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

    Raises
    ------
    ValueError
        If the object does not contain a segmentation mask.

    Returns
    -------
    xr.Dataset
        The object with the grown segmentation masks and updated observations.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    if copy:
        sdata = cp.deepcopy(sdata)

    segmentation = _process_segmentation(sdata, segmentation_key)

    # growing segmentation masks
    masks_grown = expand_labels(segmentation, iterations)

    # overwriting the segmentation mask in the object
    # get transformations
    transformation = get_transformation(sdata[segmentation_key])
    # add the segmentation masks to the spatial data object
    sdata.labels[segmentation_key] = spatialdata.models.Labels2DModel.parse(
        masks_grown, transformations=None, dims=("y", "x")
    )
    set_transformation(sdata.labels[segmentation_key], transformation)

    if len(sdata.tables.keys()) != 0:
        # after segmentation masks were grown, the obs features (e. g. centroids and areas) need to be updated
        # if anything other than the default obs were present, a warning is shown, as they will be removed
        adata = _process_adata(sdata, table_key=table_key)
        existing_features = list(adata.obs.columns)

        # getting all of the obs features
        if existing_features != [SDFeatures.ID, SDFeatures.REGION] and not suppress_warning:
            logger.warning(
                "Mask growing requires recalculation of the observations. All features will be removed and should be recalculated with pp.add_observations()."
            )

        # removing the old obs
        cell_idx = adata.obs[SDFeatures.ID].values.copy()
        adata.obs = pd.DataFrame(index=adata.obs_names)
        adata.obs[SDFeatures.ID] = cell_idx
        adata.obs[SDFeatures.REGION] = segmentation_key

    if copy:
        return sdata


def merge_channels(
    sdata,
    channels: List[str],
    key_added: str,
    normalize: bool = True,
    method: str = "sum",
    image_key: str = SDLayers.IMAGE,
    copy: bool = False,
):
    """
    Merges multiple channels into a single channel in the spatialdata object.

    Parameters
    ----------
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

    Raises
    ------
    AssertionError
        If key_added already exists in the object.
    KeyError
        If any of the specified channels do not exist in the image data.
    ValueError
        If less than two channels are provided to merge.
        If an unknown merging method is provided.

    Returns
    -------
    xr.Dataset
        The object with the merged channel added.
    """
    import spatialdata
    from spatialdata.transformations import get_transformation, set_transformation

    # check that the new channel name does not already exist
    assert (
        key_added not in sdata.images[image_key].coords["c"].values
    ), f"Channel {key_added} already exists in the object. Please choose a different name by setting the 'key_added' parameter."

    # check that the channels are a list, and that there are at least two channels to merge
    assert isinstance(channels, list), "Channels must be provided as a list."
    assert len(channels) >= 2, "At least two channels must be provided to merge."

    if copy:
        sdata = cp.deepcopy(sdata)

    image = _process_image(sdata, image_key=image_key, key_added=None, return_values=False)
    all_channels = list(sdata.images[image_key].coords["c"].values)

    merged_image = _merge_channels(
        image.sel(c=channels).values,
        normalize=normalize,
        method=method,
    )

    # adding the merged channel to the image data
    merged_image = np.concatenate([image, merged_image[np.newaxis, :, :]], axis=0)
    all_channels = all_channels + [key_added]

    # get transformations
    transformation = get_transformation(sdata.images[image_key])

    # add the image to the spatial data object
    sdata.images[image_key] = spatialdata.models.Image2DModel.parse(
        merged_image, c_coords=all_channels, transformations=None, dims=("c", "y", "x")
    )
    set_transformation(sdata.images[image_key], transformation)

    if copy:
        return sdata


# === SPATIALPROTEOMICS ACCESSOR ===
@xr.register_dataset_accessor("pp")
class PreprocessingAccessor:
    """The image accessor enables fast indexing and preprocessing of the spatialproteomics object."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, indices) -> xr.Dataset:
        """
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

        Parameters
        ----------
        indices : str, slice, list, tuple
            The indices to subset the image container.

        Returns
        -------
        xr.Dataset
            The subsetted image container.
        """
        # checking if the user provided dict_values or dict_keys and turns them into a list if that is the case
        if type(indices) is {}.keys().__class__ or type(indices) is {}.values().__class__:
            indices = list(indices)

        if type(indices) is str:
            c_slice = [indices]
            x_slice = slice(None)
            y_slice = slice(None)
        elif type(indices) is slice:
            c_slice = slice(None)
            x_slice = indices
            y_slice = slice(None)
        elif type(indices) is list:
            all_str = all([type(s) is str for s in indices])

            if all_str:
                c_slice = indices
                x_slice = slice(None)
                y_slice = slice(None)
            else:
                raise TypeError(f"Invalid input. Found non-string elements in the list. Input list: {indices}")

        elif type(indices) is tuple:
            all_str = all([type(s) is str for s in indices])

            if all_str:
                c_slice = [*indices]
                x_slice = slice(None)
                y_slice = slice(None)

            if len(indices) == 2:
                if (type(indices[0]) is slice) & (type(indices[1]) is slice):
                    c_slice = slice(None)
                    x_slice = indices[0]
                    y_slice = indices[1]
                elif (type(indices[0]) is str) & (type(indices[1]) is slice):
                    # Handles arguments in form of im['Hoechst', 500:1000]
                    c_slice = [indices[0]]
                    x_slice = indices[1]
                    y_slice = slice(None)
                elif (type(indices[0]) is list) & (type(indices[1]) is slice):
                    c_slice = indices[0]
                    x_slice = indices[1]
                    y_slice = slice(None)
                else:
                    raise AssertionError("Some error in handling the input arguments")

            elif len(indices) == 3:
                if type(indices[0]) is str:
                    c_slice = [indices[0]]
                elif type(indices[0]) is list:
                    c_slice = indices[0]
                else:
                    raise AssertionError("First index must index channel coordinates.")

                if (type(indices[1]) is slice) & (type(indices[2]) is slice):
                    x_slice = indices[1]
                    y_slice = indices[2]
        else:
            raise TypeError(
                f"Invalid input. To subselect, you can input a string, slice, list, or tuple. You provided {type(indices)}"
            )

        ds = self._obj.pp.get_channels(c_slice)

        return ds.pp.get_bbox(x_slice, y_slice)

    def get_bbox(self, x_slice: slice, y_slice: slice) -> xr.Dataset:
        """
        Returns the bounds of the image container.

        Parameters
        ----------
        x_slice : slice
            The slice representing the x-coordinates for the bounding box.
        y_slice : slice
            The slice representing the y-coordinates for the bounding box.

        Returns
        -------
        xr.Dataset
            The updated image container.
        """

        # get the dimensionality of the image
        xdim = self._obj.coords[Dims.X]
        ydim = self._obj.coords[Dims.Y]

        # set the start and stop indices
        x_start = xdim[0] if x_slice.start is None else x_slice.start
        y_start = ydim[0] if y_slice.start is None else y_slice.start
        x_stop = xdim[-1] if x_slice.stop is None else x_slice.stop
        y_stop = ydim[-1] if y_slice.stop is None else y_slice.stop

        # raise a warning or an error if the slices are out of bounds
        x_start_clamped, x_stop_clamped = _validate_and_clamp_slice(x_start, x_stop, xdim, "X_slice")
        y_start_clamped, y_stop_clamped = _validate_and_clamp_slice(y_start, y_stop, ydim, "Y_slice")

        x_slice = slice(x_start_clamped, x_stop_clamped)
        y_slice = slice(y_start_clamped, y_stop_clamped)

        # set up query
        query = {
            Dims.X: x_slice,
            Dims.Y: y_slice,
        }

        # handle case when there are cells in the image
        if Dims.CELLS in self._obj.sizes:
            coords = self._obj[Layers.OBS]
            cells = (
                (coords.loc[:, Features.X] >= x_start)
                & (coords.loc[:, Features.X] <= x_stop)
                & (coords.loc[:, Features.Y] >= y_start)
                & (coords.loc[:, Features.Y] <= y_stop)
            ).values

            # finalise query
            query[Dims.CELLS] = cells

            # ensuring that cells and cells_2 are synchronized
            if Dims.CELLS_2 in self._obj.coords:
                query[Dims.CELLS_2] = cells

            # if the centroids of a cell are outside the bounding box, the cell is removed
            # to ensure proper synchronization between the segmentation and the observations,
            # we also remove the cell from the segmentation
            # synchronizing the segmentation mask with the selected cells
            obj = self._obj.sel(query)
            cells = obj.coords[Dims.CELLS].values
            segmentation = obj[Layers.SEGMENTATION].values
            # setting all cells that are not in cells to 0
            segmentation = _remove_unlabeled_cells(segmentation, cells)

            # creating a data array with the segmentation mask, so that we can merge it to the original
            da = xr.DataArray(
                segmentation,
                coords=[obj.coords[Dims.Y], obj.coords[Dims.X]],
                dims=[Dims.Y, Dims.X],
                name=Layers.SEGMENTATION,
            )

            # removing the old segmentation
            obj = obj.drop_vars(Layers.SEGMENTATION)

            # adding the new filtered and relabeled segmentation
            return xr.merge([obj, da], join="outer", compat="no_conflicts")

        return self._obj.sel(query)

    def get_channels(self, channels: Union[List[str], str]) -> xr.Dataset:
        """
        Retrieve the specified channels from the dataset.

        Parameters
        ----------
        channels : Union[List[str], str]
            The channels to retrieve. Can be a single channel name or a list of channel names.

        Returns
        -------
        xr.Dataset
            The dataset containing the specified channels.
        """
        if isinstance(channels, str):
            channels = [channels]

        # build query
        query = {Dims.CHANNELS: channels}

        return self._obj.sel(query)

    def add_channel(self, channels: Union[str, list], array: np.ndarray, layer_key: str = Layers.IMAGE) -> xr.Dataset:
        """
        Adds channel(s) to an existing image container.

        Parameters
        ----------
        channels : Union[str, list]
            The name of the channel or a list of channel names to be added.
        array : np.ndarray
            The numpy array representing the channel(s) to be added.
        layer_key : str
            The layer key where the channel(s) should be added. Default is 'image'.

        Returns
        -------
        xr.Dataset
            The updated image container with added channel(s).
        """
        assert type(array) is np.ndarray, "Added channels must be numpy arrays."
        assert array.ndim in [2, 3], "Added channels must be 2D or 3D arrays."

        if array.ndim == 2:
            array = np.expand_dims(array, 0)

        if type(channels) is str:
            channels = [channels]

        assert (
            set(channels).intersection(set(self._obj.coords[Dims.CHANNELS].values)) == set()
        ), "Can't add a channel that already exists."

        self_channels, self_x_dim, self_y_dim = self._obj[Layers.IMAGE].shape
        other_channels, other_x_dim, other_y_dim = array.shape

        assert (
            len(channels) == other_channels
        ), "The length of channels must match the number of channels in array (DxMxN)."
        assert (self_x_dim == other_x_dim) & (
            self_y_dim == other_y_dim
        ), "Dimensions of the original image and the input array do not match."

        da = xr.DataArray(
            array,
            coords=[channels, range(other_y_dim), range(other_x_dim)],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=layer_key,
        )

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def add_segmentation(
        self,
        segmentation: Union[str, np.ndarray] = None,
        reindex: bool = True,
        keep_labels: bool = True,
        add_obs: bool = True,
    ) -> xr.Dataset:
        """
        Adds a segmentation mask field to the xarray dataset. This will be stored in the '_segmentation' layer.

        Parameters
        ----------
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

        Returns
        --------
        xr.Dataset
            The amended xarray.
        """
        assert (
            Layers.SEGMENTATION not in self._obj
        ), f'The key "{Layers.SEGMENTATION}" already exists in the object. If you want to make a new segmentation the default, drop the old one first using pp.drop_layers("{Layers.SEGMENTATION}").'

        # flag indicating if the segmentation mask is provided as a layer key or as a numpy array
        from_layer = None
        if isinstance(segmentation, str):
            if segmentation not in self._obj:
                raise KeyError(f'The key "{segmentation}" does not exist.')

            from_layer = segmentation
            segmentation = self._obj[segmentation].values.squeeze()

        assert segmentation.ndim == 2, "A segmentation mask must 2 dimensional."
        assert ~np.any(segmentation < 0), "A segmentation mask may not contain negative numbers."

        y_dim, x_dim = segmentation.shape

        assert (x_dim == self._obj.sizes[Dims.X]) & (
            y_dim == self._obj.sizes[Dims.Y]
        ), "The shape of segmentation mask does not match that of the image."

        segmentation = segmentation.copy()

        if reindex:
            segmentation, reindex_dict = _relabel_cells(segmentation)

        # crete a data array with the segmentation mask
        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )

        # add cell coordinates
        obj = self._obj.copy()
        obj.coords[Dims.CELLS] = np.unique(segmentation[segmentation > 0]).astype(int)

        if keep_labels and from_layer is not None:
            # checking that the segmentation has labels in the attrs
            if len(self._obj[from_layer].attrs) > 0:
                # this is a dict that maps from cell_id to a label (e. g. {1: 'CD68', 2: 'DAPI'})
                labels = self._obj[from_layer].attrs
                # if reindex was called, we first need to propagate the mapping to the labels before we can add them
                if reindex:
                    labels = {reindex_dict[k]: v for k, v in labels.items()}
                obj = obj.la.add_labels(labels)

        obj = xr.merge([obj, da], join="outer", compat="no_conflicts")
        if add_obs:
            return obj.pp.add_observations()
        return obj

    def add_layer(
        self,
        array: np.ndarray,
        key_added: str = Layers.MASK,
    ) -> xr.Dataset:
        """
        Adds a layer (such as a mask highlighting artifacts) to the xarray dataset.

        Parameters
        ----------
        array : np.ndarray
            The array representing the layer to be added. Can either be 2D or 3D (in this case, the first dimension should be the number of channels).
        key_added : str, optional
            The name of the added layer in the xarray dataset. Default is '_mask'.
        Returns
        -------
        xr.Dataset
            The updated dataset with the added layer.
        Raises
        ------
        AssertionError
            If the array is not 2-dimensional or its shape does not match the image shape.
        Notes
        -----
        This method adds a layer to the xarray dataset, where the layer has the same shape as the image field.
        The array should be a 2-dimensional numpy array representing the segmentation mask or layer to be added.
        The layer is created as a DataArray with the same coordinates and dimensions as the image field.
        The name of the added layer in the xarray dataset can be specified using the `key_added` parameter.
        The amended xarray dataset is returned after merging the original dataset with the new layer.
        """
        # checking that the layer does not exist yet
        assert key_added not in self._obj, f"Layer {key_added} already exists."
        assert array.ndim in [2, 3], "The array to add mask must 2 or 3-dimensional."

        if array.ndim == 2:
            # in the case of a 2D array
            y_dim, x_dim = array.shape
            assert (x_dim == self._obj.sizes[Dims.X]) & (
                y_dim == self._obj.sizes[Dims.Y]
            ), f"The shape of array does not match that of the image. Image has shape ({self._obj.sizes[Dims.Y]}, {self._obj.sizes[Dims.X]}), array has shape {array.shape}."

            # create a data array with the new layer
            da = xr.DataArray(
                array,
                coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
                dims=[Dims.Y, Dims.X],
                name=key_added,
            )
        else:
            # in the case of a 3D array
            channels, y_dim, x_dim = array.shape
            assert channels == len(
                self._obj.coords[Dims.CHANNELS]
            ), f"The number of channels in the array does not match the number of channels in the image. Image has {len(self._obj.coords[Dims.CHANNELS])} channels, array has {channels} channels."
            assert (x_dim == self._obj.sizes[Dims.X]) & (
                y_dim == self._obj.sizes[Dims.Y]
            ), f"The shape of array does not match that of the image. Image has shape ({self._obj.sizes[Dims.Y]}, {self._obj.sizes[Dims.X]}), array has shape {array.shape}."

            # create a data array with the new layer
            da = xr.DataArray(
                array,
                coords=[self._obj.coords[Dims.CHANNELS], self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
                dims=[Dims.CHANNELS, Dims.Y, Dims.X],
                name=key_added,
            )

        obj = self._obj.copy()
        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def add_layer_from_dataframe(self, df: pd.DataFrame, key_added: str = Layers.LA_LAYERS) -> xr.Dataset:
        """
        Adds a dataframe as a layer to the xarray object. This is similar to add_obs, with the only difference that it can be used to add any kind of data to the xarray object.
        Useful to add things like string-based labels or other metadata.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe with the observation values.

        Returns
        -------
        xr.Dataset
            The amended image container.
        """
        assert (
            Dims.CELLS in self._obj.coords
        ), "No cell coordinates found. Please add cells by running pp.add_observations() before calling this method."

        # pulls out the cell and feature coordinates from the image container
        cells = self._obj.coords[Dims.CELLS].values

        # ensuring that the shape of the data frame fits the number of cells in the segmentation
        assert len(cells) == len(
            df.index
        ), "Number of cells in the image container does not match the number of cells in the dataframe."

        # create a data array from the dataframe
        da = xr.DataArray(
            df,
            coords=[cells, df.columns],
            dims=[Dims.CELLS, Dims.LA_FEATURES],
            name=key_added,
        )

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def add_observations(
        self,
        properties: Union[str, list, tuple] = ("label", "centroid"),
        layer_key: str = Layers.SEGMENTATION,
        return_xarray: bool = False,
    ) -> xr.Dataset:
        """
        Adds properties derived from the segmentation mask to the image container.

        Parameters
        ----------
        properties : Union[str, list, tuple]
            A list of properties to be added to the image container. See
            skimage.measure.regionprops_table for a list of available properties.
        layer_key : str
            The key of the layer that contains the segmentation mask.
        return_xarray : bool
            If true, the function returns an xarray.DataArray with the properties
            instead of adding them to the image container.

        Returns
        -------
        xr.Dataset
            The amended image container.
        """
        if layer_key not in self._obj:
            raise ValueError(
                f"No segmentation mask found at layer {layer_key}. You can specify which layer to use with the layer_key parameter."
            )

        if type(properties) is str:
            properties = [properties]

        if "label" not in properties:
            properties = ["label", *properties]

        table = regionprops_table(self._obj[layer_key].values, properties=properties)

        label = table.pop("label")
        data = []
        cols = []

        for k, v in table.items():
            if Dims.FEATURES in self._obj.coords:
                if k in self._obj.coords[Dims.FEATURES] and not return_xarray:
                    continue
            # when looking at centroids, it could happen that the image has been cropped before
            # in this case, the x and y coordinates do not necessarily start at 0
            # to accommodate for this, we add the x and y coordinates to the centroids
            if k == Features.X:
                v += self._obj.coords[Dims.X].values[0]
            if k == Features.Y:
                v += self._obj.coords[Dims.Y].values[0]
            cols.append(k)
            data.append(v)

        if len(data) == 0:
            return self._obj

        da = xr.DataArray(
            np.stack(data, -1),
            coords=[label, cols],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        if return_xarray:
            return da

        obj = self._obj.copy()

        # if there are already observations, concatenate them
        if Layers.OBS in obj:
            # checking if the new number of cells matches with the old one
            # if it does not match, we need to update the cell dimension, i. e. remove all old _obs
            if len(label) != len(obj.coords[Dims.CELLS]):
                logger.warning(
                    "Found _obs with different number of cells in the image container. Removing all old _obs for consistency."
                )
                obj = obj.pp.drop_layers(Layers.OBS)
            else:
                da = xr.concat(
                    [obj[Layers.OBS].copy(), da],
                    dim=Dims.FEATURES,
                )

        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def drop_observations(
        self,
        properties: Union[str, list, tuple],
        key: str = Dims.FEATURES,
    ) -> xr.Dataset:
        assert (
            key in self._obj.coords
        ), f"Coordinate {key} not found in the object. Available coordinates: {list(self._obj.coords)}. Please adjust the key parameter accordingly."
        if type(properties) is str:
            properties = [properties]
        for prop in properties:
            assert (
                prop in self._obj.coords[key]
            ), f"Property {prop} not found in the object. Available properties: {self._obj.coords[key].values}. Please adjust the properties parameter accordingly."

        return self._obj.sel({Dims.FEATURES: ~self._obj.coords[Dims.FEATURES].isin(properties)})

    def add_feature(self, feature_name: str, feature_values: Union[list, np.ndarray]):
        """
        Adds a feature to the image container.

        Parameters
        ----------
        feature_name : str
            The name of the feature to be added.
        feature_values :
            The values of the feature to be added.

        Returns
        -------
        xr.Dataset
            The updated image container with the added feature.
        """
        # checking if the feature already exists
        assert feature_name not in self._obj.coords[Dims.FEATURES].values, f"Feature {feature_name} already exists."

        # checking if feature_values is a list or a numpy array
        assert type(feature_values) in [list, np.ndarray], "Feature values must be a list or a numpy array."

        # if feature_values is a list, we convert it to a numpy array
        if type(feature_values) is list:
            feature_values = np.array(feature_values)

        # collapsing the feature_values to a 1D array
        feature_values = feature_values.flatten()

        # checking if the length of the feature_values matches the number of cells
        assert len(feature_values) == len(
            self._obj.coords[Dims.CELLS]
        ), "Length of feature values must match the number of cells."

        # adding a new dimension to obtain a 2D array as required by xarray
        feature_values = np.expand_dims(feature_values, 1)

        # create a data array with the feature
        da = xr.DataArray(
            feature_values,
            coords=[self._obj.coords[Dims.CELLS], [feature_name]],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        da = xr.concat(
            [self._obj[Layers.OBS].copy(), da],
            dim=Dims.FEATURES,
        )

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def add_obs_from_dataframe(self, df: pd.DataFrame) -> xr.Dataset:
        """
        Adds an observation table to the image container. Columns of the
        dataframe have to match the feature coordinates of the image
        container, and the index of the dataframe has to match the cell coordinates
        of the image container.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe with the observation values.

        Returns
        -------
        xr.Dataset
            The amended image container.
        """
        if Dims.CELLS not in self._obj.coords:
            self._obj = self._obj.pp.add_observations()

        # pulls out the cell and feature coordinates from the image container
        cells = self._obj.coords[Dims.CELLS].values

        # ensuring that the shape of the data frame fits the number of cells in the segmentation
        assert len(cells) == len(
            df.index
        ), "Number of cells in the image container does not match the number of cells in the dataframe."

        # create a data array from the dataframe
        da = xr.DataArray(
            df,
            coords=[cells, df.columns],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def add_quantification(
        self,
        func: Union[str, Callable] = "intensity_mean",
        key_added: str = Layers.INTENSITY,
        layer_key: str = Layers.IMAGE,
        return_xarray=False,
    ) -> xr.Dataset:
        """
        Quantify channel intensities over the segmentation mask.

        Parameters
        ----------
        func : Callable or str, optional
            The function used for quantification. Can either be a string to specify a function from skimage.measure.regionprops_table or a custom function. Default is 'intensity_mean'.
        key_added : str, optional
            The key under which the quantification data will be stored in the image container. Default is '_intensity'.
        layer_key : str, optional
            The key of the layer to be quantified. Default is '_image'.
        return_xarray : bool, optional
            If True, the function returns an xarray.DataArray with the quantification data instead of adding it to the image container.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The updated image container with added quantification data or the quantification data as a separate xarray.DataArray.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found.")

        assert (
            key_added not in self._obj
        ), f"Found {key_added} in image container. Please add a different key or remove the previous quantification."

        assert layer_key in self._obj, f"Layer {layer_key} not found in image container."

        if Dims.CELLS not in self._obj.coords:
            logger.warning("No cell coordinates found. Adding _obs table.")
            self._obj = self._obj.pp.add_observations()

        measurements = []
        all_channels = self._obj.coords[Dims.CHANNELS].values.tolist()

        segmentation = self._obj[Layers.SEGMENTATION].values
        image = self._obj[layer_key].values

        measurements, cell_idx = _compute_quantification(image, segmentation, func)

        da = xr.DataArray(
            np.stack(measurements, -1),
            coords=[cell_idx, all_channels],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        if return_xarray:
            return da

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def add_quantification_from_dataframe(self, df: pd.DataFrame, key_added: str = Layers.INTENSITY) -> xr.Dataset:
        """
        Adds an observation table to the image container. Columns of the
        dataframe have to match the channel coordinates of the image
        container, and the index of the dataframe has to match the cell coordinates
        of the image container.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe with the quantification values.
        key_added : str, optional
            The key under which the quantification data will be added to the image container.

        Returns
        -------
        xr.Dataset
            The amended image container.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found. A segmentation mask is required to add quantification.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input must be a pandas DataFrame.")

        # pulls out the cell and channel coordinates from the image container
        cells = self._obj.coords[Dims.CELLS].values
        channels = self._obj.coords[Dims.CHANNELS].values

        # ensuring that all cells and channels are actually in the dataframe
        assert np.all([c in df.index for c in cells]), "Cells in the image container are not in the dataframe."
        assert np.all([c in df.columns for c in channels]), "Channels in the image container are not in the dataframe."

        # create a data array from the dataframe
        da = xr.DataArray(
            df.loc[cells, channels].values,
            coords=[cells, channels],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        return xr.merge([self._obj, da], join="outer", compat="no_conflicts")

    def drop_layers(
        self,
        layers: Optional[Union[str, list]] = None,
        keep: Optional[Union[str, list]] = None,
        drop_obs: bool = True,
        suppress_warnings: bool = False,
    ) -> xr.Dataset:
        """
        Drops layers from the image container. Can either drop all layers specified in layers or drop all layers but the ones specified in keep.

        Parameters
        ----------
        layers : Union[str, list]
            The name of the layer or a list of layer names to be dropped.
        keep : Union[str, list]
            The name of the layer or a list of layer names to be kept.
        drop_obs : bool
            If True, the observations are removed when the label or neighborhood properties are dropped. Default is True.
        suppress_warnings : bool
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        xr.Dataset
            The updated image container with dropped layers.
        """
        # checking that either layers or keep is provided
        assert layers is not None or keep is not None, "Please provide either layers or keep."
        assert not (layers is not None and keep is not None), "Please provide either layers or keep."

        if type(layers) is str:
            layers = [layers]

        if type(keep) is str:
            keep = [keep]

        # if keep is provided, we drop all layers that are not in keep
        if keep is not None:
            layers = [str(x) for x in self._obj.data_vars if str(x) not in keep]

            # if the user wants to keep obs but not segmentation or vice versa, we throw a warning that this is not possible
            if Layers.SEGMENTATION in keep and Layers.OBS not in keep:
                logger.warning("Cannot drop segmentation and keep observations. Removing both.")
            if Layers.OBS in keep and Layers.SEGMENTATION not in keep:
                logger.warning("Cannot drop observations and keep segmentation. Removing both.")

        # if the segmentation layer is dropped, we also need to drop the obs and vice versa
        # this helps to ensure that the segmentation and obs always stay in sync
        if Layers.SEGMENTATION in layers and Layers.OBS in self._obj.data_vars:
            layers.append(Layers.OBS)
        if Layers.OBS in layers and Layers.SEGMENTATION in self._obj.data_vars:
            layers.append(Layers.SEGMENTATION)

        assert all(
            [layer in self._obj.data_vars for layer in layers]
        ), f"Some layers that you are trying to remove are not in the image container. Available layers are: {', '.join(self._obj.data_vars)}. Layers requested to drop: {layers}."

        obj = self._obj.drop_vars(layers)

        # iterating through the remaining layers to get the dims that should be kept
        dims_to_keep = []
        for layer in obj.data_vars:
            dims_to_keep.extend(obj[layer].dims)

        # removing all dims that are not in dims_to_keep
        for dim in obj.dims:
            if dim not in dims_to_keep:
                obj = obj.drop_dims(dim)

        # if label props are dropped, we need to remove the labels from the obs as well
        if Layers.LA_PROPERTIES in layers and Dims.FEATURES in obj.coords:
            if Features.LABELS in obj.coords[Dims.FEATURES] and drop_obs:
                if not suppress_warnings:
                    logger.info(
                        "Removing labels from observations. If you want to keep the labels in the obs layer, set drop_obs=False."
                    )
                filtered_features = obj.coords[Dims.FEATURES].where(
                    obj.coords[Dims.FEATURES] != Features.LABELS, drop=True
                )
                obj = obj.sel(features=filtered_features)

        # if neighborhood props are dropped, we need to remove the neighborhoods from the obs as well
        if Layers.NH_PROPERTIES in layers and Dims.FEATURES in obj.coords:
            if Features.NEIGHBORHOODS in obj.coords[Dims.FEATURES] and drop_obs:
                if not suppress_warnings:
                    logger.info(
                        "Removing neighborhoods from observations. If you want to keep the neighborhoods in the obs layer, set drop_obs=False."
                    )
                filtered_features = obj.coords[Dims.FEATURES].where(
                    obj.coords[Dims.FEATURES] != Features.NEIGHBORHOODS, drop=True
                )
                obj = obj.sel(features=filtered_features)

        return obj

    def threshold(
        self,
        quantile: Union[float, list] = None,
        intensity: Union[int, list] = None,
        key_added: Optional[str] = None,
        channels: Optional[Union[str, list]] = None,
        shift: bool = True,
        **kwargs,
    ):
        """
        Apply thresholding to the image layer of the object.
        By default, shift is set to true. This means that the threshold value is subtracted from the image, and all negative values are set to 0.
        If you instead want to set all values below the threshold to 0 while retaining the rest of the image at the original values, set shift to False.

        Parameters
        ----------
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

        Returns
        -------
        xr.Dataset
            The object with the thresholding applied to the image layer.

        Raises
        ------
        ValueError
            If both quantile and intensity are None or if both quantile and intensity are provided.
        """
        if Layers.PLOT in self._obj:
            logger.warning(
                "Please only call plotting methods like pl.colorize() after any preprocessing. Otherwise, the image will not be displayed correctly."
            )

        # Pull out the image from its corresponding field (by default "_image")
        image_layer = self._obj[Layers.IMAGE]

        # performing the thresholding
        filtered = _threshold(
            image_layer, quantile=quantile, intensity=intensity, channels=channels, shift=shift, **kwargs
        )

        if key_added is None:
            # drop_vars returns a copy of the data array and should not perform any in-place operations
            obj = self._obj.drop_vars(Layers.IMAGE)
        else:
            # this is a reference, however xr.merge does not alter the original object, so it is safe to use it here
            obj = self._obj

        filtered = xr.DataArray(
            filtered,
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.IMAGE if key_added is None else key_added,
        )
        return xr.merge([obj, filtered], join="outer", compat="no_conflicts")

    def apply(self, func: Callable, key: str = Layers.IMAGE, key_added: str = Layers.IMAGE, **kwargs):
        """
        Apply a function to each channel independently.

        Parameters
        ----------
        func : Callable
            The function to apply to the layer.
        key : str
            The key of the layer to apply the function to. Default is '_image'.
        key_added : str
            The key under which the updated layer will be stored. Default is '_image' (i. e. the original image will be overwritten).
        **kwargs : dict, optional
            Additional keyword arguments to pass to the function.

        Returns
        -------
        xr.Dataset
            The updated image container with the applied function.
        """
        # checking if the key is in the object
        assert key in self._obj, f"Key {key} not found in the image container."

        obj = self._obj.copy()
        layer = obj[key].copy()

        processed_layer = _apply(layer, func, **kwargs)

        # adding the modified layer to the object
        obj[key_added] = xr.DataArray(
            processed_layer,
            coords=[self._obj.coords[Dims.CHANNELS], self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added,
        )

        return obj

    def normalize(self):
        """
        Performs a percentile normalization on each channel using the 3- and 99.8-percentile. Resulting values are in the range of 0 to 1.

        Returns
        -------
        xr.Dataset
            The image container with the normalized image stored in '_plot'.
        """
        image_layer = self._obj[Layers.IMAGE]
        normed = xr.DataArray(
            _normalize(image_layer.values),
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.PLOT,
        )

        return xr.merge([self._obj, normed], join="outer", compat="no_conflicts")

    def downsample(self, rate: int):
        """
        Downsamples the entire dataset by selecting every `rate`-th element along the x and y dimensions.

        Parameters
        ----------
        rate : int
            The downsampling rate. Only every `rate`-th pixel (or coordinate) is kept.

        Returns
        -------
        xr.Dataset
            The downsampled dataset with updated x and y coordinates.
        """
        # Make a copy of the original dataset
        new_obj = self._obj.copy()

        # Use isel to select every `rate`-th element along the x and y dimensions.
        # This applies to every variable that depends on these dimensions.
        new_obj = new_obj.isel({Dims.X: slice(None, None, rate), Dims.Y: slice(None, None, rate)})

        # Optionally, update the x and y coordinate arrays if needed.
        # In many cases, isel will automatically update the coordinate arrays,
        # but you can explicitly assign them if required.
        new_obj = new_obj.assign_coords({Dims.X: new_obj.coords[Dims.X].values, Dims.Y: new_obj.coords[Dims.Y].values})

        return new_obj

    def rescale(self, scale: int):
        """
        Rescales the image and segmentation mask in the object by a given scale.

        Parameters
        ----------
        scale :int
            The scale factor by which to rescale the image and segmentation mask.

        Returns
        -------
        xr.Dataset
            The rescaled object containing the updated image and segmentation mask.

        Raises
        ------
        - AssertionError: If no image layer is found in the object.
        - AssertionError: If no segmentation mask is found in the object.
        """
        # checking if the object contains an image layer
        assert Layers.IMAGE in self._obj, "No image layer found in the object."
        # checking if the object contains a segmentation mask
        assert Layers.SEGMENTATION in self._obj, "No segmentation mask found in the object."

        image_layer = self._obj[Layers.IMAGE]
        img = skimage.transform.rescale(image_layer.values, scale=scale, channel_axis=0)
        x = np.array(range(img.shape[1]))
        y = np.array(range(img.shape[2]))
        c = self._obj.channels.values
        new_img = xr.DataArray(img, coords=[c, y, x], dims=[Dims.CHANNELS, Dims.Y, Dims.X], name=Layers.IMAGE)
        obj = self._obj.drop(Layers.IMAGE)

        if Layers.SEGMENTATION in self._obj:
            seg_layer = self._obj[Layers.SEGMENTATION]
            seg = skimage.transform.rescale(seg_layer.values, scale=scale)
            new_seg = xr.DataArray(seg, coords=[y, x], dims=[Dims.Y, Dims.X], name=Layers.SEGMENTATION)
            obj = obj.drop(Layers.SEGMENTATION)

        obj = obj.drop_dims([Dims.Y, Dims.X])

        return xr.merge([obj, new_img, new_seg], join="outer", compat="no_conflicts")

    def filter_by_obs(self, col: str, func: Callable, segmentation_key: str = Layers.SEGMENTATION):
        """
        Filter the object by observations based on a given feature and filtering function.

        Parameters:
            col (str): The name of the feature to filter by.
            func (Callable): A filtering function that takes in the values of the feature and returns a boolean array.
            segmentation_key (str): The key of the segmentation mask in the object. Default is Layers.SEGMENTATION.

        Returns:
            xr.Dataset: The filtered object with the selected cells and updated segmentation mask.

        Raises:
            AssertionError: If the feature does not exist in the object's observations.

        Notes:
            - This method filters the object by selecting only the cells that satisfy the filtering condition.
            - It also updates the segmentation mask to remove cells that are not selected and relabels the remaining cells.

        Example:
            To filter the object by the feature "area" and keep only the cells with an area greater than 70px:
            `obj = obj.pp.add_observations('area').pp.filter_by_obs('area', lambda x: x > 70)`
        """
        # checking if the feature exists in obs
        assert (
            col in self._obj.coords[Dims.FEATURES].values
        ), f"Feature {col} not found in obs. You can add it with pp.add_observations()."

        assert (
            segmentation_key in self._obj
        ), f"Segmentation mask with key {segmentation_key} not found in the object. You can specify the key with the segmentation_key parameter."

        cells = self._obj[Layers.OBS].sel({Dims.FEATURES: col}).values.copy()
        cells_bool = func(cells)
        cells_sel = self._obj.coords[Dims.CELLS][cells_bool].values

        # making sure that if there is a cells_2 coordinate, this is also subset correctly
        query = {Dims.CELLS: cells_sel}
        if Dims.CELLS_2 in self._obj.coords:
            query[Dims.CELLS_2] = cells_sel
        obj = self._obj.sel(query)

        # synchronizing the segmentation mask with the selected cells
        segmentation = obj[segmentation_key].values
        # setting all cells that are not in cells to 0
        segmentation = _remove_unlabeled_cells(segmentation, cells_sel)
        # relabeling cells in the segmentation mask so the IDs go from 1 to n again
        segmentation, relabel_dict = _relabel_cells(segmentation)
        # updating the cell coords of the object
        obj.coords[Dims.CELLS] = [relabel_dict[cell] for cell in obj.coords[Dims.CELLS].values]

        # creating a data array with the segmentation mask, so that we can merge it to the original
        da = xr.DataArray(
            segmentation,
            coords=[obj.coords[Dims.Y], obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=segmentation_key,
        )

        # removing the old segmentation
        obj = obj.drop_vars(segmentation_key)

        # adding the new filtered and relabeled segmentation
        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def remove_outlying_cells(
        self, dilation_size: int = 25, threshold: int = 5, segmentation_key: str = Layers.SEGMENTATION
    ):
        """
        Removes outlying cells from the image container. It does so by dilating the segmentation mask and removing cells that belong to a connected component with less than 'threshold' cells.

        Parameters
        ----------
        dilation_size : int
            The size of the dilation kernel. Default is 25.
        threshold : int
            The minimum number of cells in a connected component required for the cells to be kept. Default is 5.
        segmentation_key : str
            The key of the segmentation mask in the object. Default is '_segmentation'.

        Returns
        -------
        xr.Dataset
            The updated image container with the outlying cells removed.

        Raises
        ------
        ValueError
            If the object does not contain a segmentation mask.
        """
        # Validate input parameters
        if dilation_size <= 0 or threshold <= 0:
            raise ValueError("Dilation size and threshold must be positive integers.")

        # Check if the segmentation mask exists
        if segmentation_key not in self._obj:
            raise ValueError(f"No segmentation mask found with key '{segmentation_key}' in the object.")

        # getting the segmentation mask
        segmentation = self._obj[segmentation_key].values

        # removing outlying cells
        cells_sel = _remove_outlying_cells(segmentation, dilation_size, threshold)
        # making sure that if there is a cells_2 coordinate, this is also subset correctly
        query = {Dims.CELLS: cells_sel}
        if Dims.CELLS_2 in self._obj.coords:
            query[Dims.CELLS_2] = cells_sel
        obj = self._obj.sel(query)
        # setting all cells that are not in cells to 0
        segmentation = _remove_unlabeled_cells(segmentation, cells_sel)
        # relabeling cells in the segmentation mask so the IDs go from 1 to n again
        segmentation, relabel_dict = _relabel_cells(segmentation)
        # updating the cell coords of the object
        obj.coords[Dims.CELLS] = [relabel_dict[cell] for cell in obj.coords[Dims.CELLS].values]

        # creating a data array with the segmentation mask, so that we can merge it to the original
        da = xr.DataArray(
            segmentation,
            coords=[obj.coords[Dims.Y], obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=segmentation_key,
        )

        # removing the old segmentation
        obj = obj.drop_vars(segmentation_key)
        # adding the new filtered and relabeled segmentation
        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def grow_cells(self, iterations: int = 2, suppress_warning: bool = False) -> xr.Dataset:
        """
        Grows the segmentation masks by expanding the labels in the object.

        Parameters
        ----------
        iterations : int
            The number of iterations to grow the segmentation masks. Default is 2.
        suppress_warning :bool
            Whether to suppress the warning about recalculating the observations. Used internally, default is False.

        Raises
        ------
        ValueError
            If the object does not contain a segmentation mask.

        Returns
        -------
        xr.Dataset
            The object with the grown segmentation masks and updated observations.
        """

        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("The object does not contain a segmentation mask.")

        # getting the segmentation mask
        segmentation = self._obj[Layers.SEGMENTATION].values

        # growing segmentation masks
        masks_grown = expand_labels(segmentation, iterations)

        # assigning the grown masks to the object
        da = xr.DataArray(
            masks_grown,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=Layers.SEGMENTATION,
        )

        # replacing the old segmentation mask with the new one
        obj = self._obj.drop_vars(Layers.SEGMENTATION)
        obj = xr.merge([obj, da], join="outer", compat="no_conflicts")

        # after segmentation masks were grown, the obs features (e. g. centroids and areas) need to be updated
        # if anything other than the default obs were present, a warning is shown, as they will be removed

        # getting all of the obs features
        obs_features = sorted(list(self._obj.coords[Dims.FEATURES].values))
        if obs_features != [Features.Y, Features.X] and not suppress_warning:
            logger.warning(
                "Mask growing requires recalculation of the observations. All features other than the centroids will be removed and should be recalculated with pp.add_observations()."
            )
        # removing the original obs and features from the object
        obj = obj.drop_vars(Layers.OBS)
        obj = obj.drop_dims(Dims.FEATURES)

        # adding the default obs back to the object
        return obj.pp.add_observations()

    def merge_segmentation(
        self,
        layer_key: str,
        key_added: str = "_merged_segmentation",
        labels: Optional[List[str]] = None,
        threshold: float = 0.8,
    ):
        """
        Merge segmentation masks.
        This can be done in two ways: either by merging a multi-dimensional array from the object directly, or by adding a numpy array.
        You can either just merge a multi-dimensional array, or merge to an existing 1D mask (e. g. a precomputed DAPI segmentation).

        Parameters
        ----------
        layer_key : Union[str, List[str]]
            The key(s) of the segmentation mask(s) to merge. Can be a single key (must be 3D) or a list of keys (each 2D).
        key_added : str
            The name of the new segmentation mask to be added to the xarray object. Default is "_merged_segmentation".
        labels : Optional[List[str]]
            Optional. Labels corresponding to each segmentation mask. If provided, must match number of arrays.
        threshold : float
            Optional. Threshold for merging cells. Default is 0.8.

        Returns
        -------
        xr.Dataset
            The xarray object with the merged segmentation mask.

        Raises
        ------
            AssertionError
                If specified keys are not found or other input inconsistencies exist.

        Notes
        -----
            - If the input array is 2D, it will be expanded to 3D.
            - If labels are provided, they need to match the number of arrays.
            - The merging process starts with merging the biggest cells first, then the smaller ones.
            - Disconnected cells in the input are handled based on the specified method.
        """
        # Make sure layer_key is a list internally
        if isinstance(layer_key, str):
            layer_keys = [layer_key]
        else:
            layer_keys = layer_key

        # Check: All layer keys must exist in the object
        for lk in layer_keys:
            assert lk in self._obj, f"The key '{lk}' does not exist in the object."

        # Check: The key_added must not already exist
        assert key_added not in self._obj, f"The key '{key_added}' already exists in the object."

        # merge big cells first, then small cells
        channels = self._obj.coords[Dims.CHANNELS].values.tolist()

        # checking that the number of labels matches the number of channels
        if labels is not None:
            if len(layer_keys) == 1:
                assert len(labels) == len(
                    channels
                ), f"The number of labels ({len(labels)}) must match the number of channels ({len(channels)})."
            else:
                assert len(labels) == len(
                    layer_keys
                ), f"The number of labels ({len(labels)}) must match the number of layer keys ({len(layer_keys)})."

        # Special check if only a single key is provided
        if len(layer_keys) == 1:
            first_array = self._obj[layer_keys[0]].values
            if first_array.ndim != 3:
                raise ValueError(
                    f"The segmentation mask '{layer_keys[0]}' must be 3D (channels, y, x). "
                    "If you have 2D arrays, provide multiple keys instead."
                )
            segmentation = first_array[0]  # Start with first channel
        else:
            # Check that all arrays are 2D
            for lk in layer_keys:
                arr = self._obj[lk].values
                if arr.ndim != 2:
                    raise ValueError(
                        f"Segmentation mask '{lk}' must be 2D. " "All masks must be 2D when using a list of keys."
                    )

            segmentation = self._obj[layer_keys[0]].values

        # Initialize an empty mapping
        mapping = {}

        if len(layer_keys) == 1:
            # single 3D stack → one merge per extra channel
            first_array = self._obj[layer_keys[0]].values
            merge_range = range(1, first_array.shape[0])
        else:
            # multiple 2D masks → one merge per extra key
            merge_range = range(1, len(layer_keys))

        # Iterate over each “next” layer to merge in
        for i in merge_range:
            current_layer_key = layer_keys[i] if i < len(layer_keys) else layer_keys[-1]
            if len(layer_keys) == 1:
                # single 3D stack → pull channel i
                next_segmentation = first_array[i]
            else:
                # multiple 2D masks → pull mask i
                next_segmentation = self._obj[current_layer_key].values

            if labels is not None:
                label_1, label_2 = labels[i - 1], labels[i]
            else:
                label_1, label_2 = channels[i - 1], channels[i]

            # Perform merging
            segmentation, final_mapping = _merge_segmentation(
                segmentation.squeeze(),
                next_segmentation.squeeze(),
                label1=label_1,
                label2=label_2,
                threshold=threshold,
            )

            # Update label mapping
            if i == 1:
                mapping = final_mapping
            else:
                mapping = {k: mapping.get(k, v) for k, v in final_mapping.items()}

        # Copy the object
        obj = self._obj.copy()

        # assigning the new segmentation to the object
        da = xr.DataArray(
            segmentation,
            coords=[self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=key_added,
            attrs=mapping,
        )

        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def get_layer_as_df(
        self,
        layer: str = Layers.OBS,
        celltypes_to_str: bool = True,
        neighborhoods_to_str: bool = True,
        idx_to_str: bool = False,
    ) -> pd.DataFrame:
        """
        Returns the specified layer as a pandas DataFrame.

        Parameters:
            layer (str): The name of the layer to retrieve. Defaults to Layers.OBS.
            celltypes_to_str (bool): Whether to convert celltype labels to strings. Defaults to True.
            neighborhoods_to_str (bool): Whether to convert neighborhood labels to strings. Defaults to True.
            idx_to_str (bool): Whether to convert the index to strings. Defaults to False.

        Returns:
            pandas.DataFrame: The layer data as a DataFrame.
        """
        assert layer in self._obj, f"Layer {layer} not found in the object."
        data_array = self._obj[layer]

        dims = data_array.dims
        coords = data_array.coords
        c1, c2 = coords[dims[0]].values, coords[dims[1]].values
        df = pd.DataFrame(data_array.values, index=c1, columns=c2)

        # special case: converting celltypes to strings
        if celltypes_to_str:
            # converting cts to strings in the obs df
            if layer == Layers.OBS and Features.LABELS in df.columns:
                label_dict = self._obj.la._label_to_dict(Props.NAME)
                # the conversion to int is necessary because when storing to zarr, all obs variables are converted to floats
                df[Features.LABELS] = df[Features.LABELS].apply(lambda x: label_dict[int(x)])
            # converting cts to strings in the neighborhood df
            if layer == Layers.NEIGHBORHOODS:
                label_dict = self._obj.la._label_to_dict(Props.NAME)
                df.columns = [label_dict[x] for x in df.columns.values]

        if neighborhoods_to_str:
            # converting neighborhoods to strings in the obs df
            if layer == Layers.OBS and Features.NEIGHBORHOODS in df.columns:
                label_dict = self._obj.nh._neighborhood_to_dict(Props.NAME)
                df[Features.NEIGHBORHOODS] = df[Features.NEIGHBORHOODS].apply(lambda x: label_dict[x])

        if idx_to_str:
            df.index = df.index.astype(str)

        return df

    def get_disconnected_cell(self) -> int:
        """
        Returns the first disconnected cell from the segmentation layer.

        Returns:
            np.ndarray: The first disconnected cell from the segmentation layer.
        """
        return _get_disconnected_cell(self._obj[Layers.SEGMENTATION])

    def transform_expression_matrix(
        self,
        method: str = "arcsinh",
        key: str = Layers.INTENSITY,
        key_added: str = Layers.INTENSITY,
        cofactor: float = 5.0,
        min_percentile: float = 1.0,
        max_percentile: float = 99.0,
        **kwargs,
    ):
        """
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
        """
        # checking if there is an expression matrix in the object
        assert key in self._obj, f"No expression matrix found at layer {key}."

        # getting the expression matrix from the object
        expression_matrix = self._obj[key].values

        transformed_matrix = _transform_expression_matrix(
            expression_matrix,
            method=method,
            cofactor=cofactor,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
            **kwargs,
        )

        # creating a new data array with the transformed matrix
        da = xr.DataArray(
            transformed_matrix,
            coords=[self._obj.coords[Dims.CELLS], self._obj.coords[Dims.CHANNELS]],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        obj = self._obj.copy()
        # removing the old expression matrix from the object
        if key == key_added:
            obj = obj.drop_vars(key)

        # adding the transformed matrix to the object
        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def mask_region(self, key: str = Layers.MASK, image_key=Layers.IMAGE, key_added=Layers.IMAGE) -> xr.Dataset:
        """
        Mask a region in the image.

        Parameters:
            key (str): The key of the region to mask.
            image_key (str): The key of the image layer in the object. Default is Layers.IMAGE.
            key_added (str): The key to assign to the masked image in the object. Default is Layers.IMAGE, which overwrites the original image.

        Returns:
            xr.Dataset: The object with the masked region in the image.
        """
        # checking if the keys exist
        assert key in self._obj, f"The key {key} does not exist in the object."
        assert image_key in self._obj, f"The key {image_key} does not exist in the object."

        # getting the region to mask
        mask = self._obj[key].values
        image = self._obj[image_key].values

        # checking that the mask only contains zeroes and ones
        assert np.all(np.isin(mask, [0, 1])), "The mask must only contain zeroes and ones."

        # masking the region in the image (so that only pixels with a one remain)
        masked_image = mask * image

        # removing the old image from the object
        if image_key == key_added:
            obj = self._obj.drop_vars(image_key)
        else:
            obj = self._obj.copy()

        # assigning the masked image to the object
        da = xr.DataArray(
            masked_image,
            coords=[self._obj.coords[Dims.CHANNELS], self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added,
        )

        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def mask_cells(self, mask_key: str = Layers.MASK, segmentation_key=Layers.SEGMENTATION) -> xr.Dataset:
        """
        Mask cells in the segmentation mask.

        Parameters:
            mask_key (str): The key of the mask to use for masking.
            segmentation_key (str): The key of the segmentation mask in the object. Default is Layers.SEGMENTATION.

        Returns:
            xr.Dataset: The object with the masked cells in the segmentation mask.
        """
        # checking if the keys exist
        assert mask_key in self._obj, f"The key {mask_key} does not exist in the object."
        assert segmentation_key in self._obj, f"The key {segmentation_key} does not exist in the object."

        # getting the mask and segmentation mask
        mask = self._obj[mask_key].values
        segmentation = self._obj[segmentation_key].values

        # checking that the mask only contains zeroes and ones
        assert np.all(np.isin(mask, [0, 1])), "The mask must only contain zeroes and ones."

        # getting all of the cells that overlap with the region where the mask is 0
        cells_to_remove = np.unique(segmentation[mask == 0])

        # removing the cells from the segmentation mask
        cells_sel = np.array(sorted(set(self._obj.coords[Dims.CELLS].values) - set(cells_to_remove)))

        # selecting only the cells that are in cells_sel
        obj = self._obj.sel({Dims.CELLS: cells_sel})

        # synchronizing the segmentation mask with the selected cells
        segmentation = obj[segmentation_key].values
        # setting all cells that are not in cells to 0
        segmentation = _remove_unlabeled_cells(segmentation, cells_sel)
        # relabeling cells in the segmentation mask so the IDs go from 1 to n again
        segmentation, relabel_dict = _relabel_cells(segmentation)
        # updating the cell coords of the object
        obj.coords[Dims.CELLS] = [relabel_dict[cell] for cell in obj.coords["cells"].values]

        # creating a data array with the segmentation mask, so that we can merge it to the original
        da = xr.DataArray(
            segmentation,
            coords=[obj.coords[Dims.Y], obj.coords[Dims.X]],
            dims=[Dims.Y, Dims.X],
            name=segmentation_key,
        )

        # removing the old segmentation
        obj = obj.drop_vars(segmentation_key)

        # adding the new filtered and relabeled segmentation
        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def convert_to_8bit(self, key: str = Layers.IMAGE, key_added: str = Layers.IMAGE):
        """
        Convert the image to 8-bit.

        Parameters:
            key (str): The key of the image layer in the object. Default is '_image'.
            key_added (str): The key to assign to the 8-bit image in the object. Default is '_image', which overwrites the original image.

        Returns:
            xr.Dataset: The object with the image converted to 8-bit.
        """
        # checking if the key exists
        assert key in self._obj, f"The key {key} does not exist in the object."

        # getting the image from the object
        image = self._obj[key].values

        # converting the image to 8-bit
        image_8bit = _convert_to_8bit(image)

        # removing the old image from the object
        if key == key_added:
            obj = self._obj.drop_vars(key)
        else:
            obj = self._obj.copy()

        # assigning the 8-bit image to the object
        # special case: if the image is 2D, we need to add a channel dimension
        if len(image_8bit.shape) == 2:
            image_8bit = np.expand_dims(image_8bit, axis=0)

        da = xr.DataArray(
            image_8bit,
            coords=[self._obj.coords[Dims.CHANNELS], self._obj.coords[Dims.Y], self._obj.coords[Dims.X]],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=key_added,
        )

        return xr.merge([obj, da], join="outer", compat="no_conflicts")

    def merge_channels(
        self,
        channels: List[str],
        key_added: str = "merged_channel",
        normalize: bool = True,
        method: Union[str, Callable] = "max",
        layer_key: str = Layers.IMAGE,
    ) -> xr.Dataset:
        """
        Merge specified channels into a single channel by summing their values.

        Parameters
        ----------
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

        Returns
        -------
        xr.Dataset
            The object with the merged channels in the image layer.
        """
        # check that the new channel name does not already exist
        assert (
            key_added not in self._obj.coords[Dims.CHANNELS].values
        ), f"Channel {key_added} already exists in the object. Please choose a different name by setting the 'key_added' parameter."

        # check that the channels are a list, and that there are at least two channels to merge
        assert isinstance(channels, list), "Channels must be provided as a list."
        assert len(channels) >= 2, "At least two channels must be provided to merge."

        # getting all relevant channels
        arr = self._obj.pp[channels][layer_key].values

        merged = _merge_channels(arr, method=method, normalize=normalize)

        # add the new channel to the object
        # if there are multiple layers with channels as coordinates, this will turn it into float32, since it will introduce NaNs for the new channel in other layers
        obj = self._obj.pp.add_channel(channels=key_added, array=merged)
        return obj
