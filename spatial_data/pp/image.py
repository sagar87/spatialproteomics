from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops_table

from ..base_logger import logger
from ..constants import COLORS, Attrs, Dims, Features, Layers, Props
from ..la.label import _format_labels
from .segmentation import _remove_unlabeled_cells, sum_intensity
from .transforms import _colorize, _normalize


@xr.register_dataset_accessor("pp")
class PreprocessingAccessor:
    """The image accessor enables fast indexing and preprocesses image.data"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, indices):
        """Fast subsetting the image container. The following examples show how
        the user can subset the image container:

        Subset the image container using x and y coordinates:
        >> ds.pp[0:50, 0:50]

        Subset the image container using x and y coordinates and channels:
        >> ds.pp['Hoechst', 0:50, 0:50]

        Subset the image container using channels:
        >> ds.pp['Hoechst']

        Multiple channels can be selected by passing a list of channels:
        >> ds.pp[['Hoechst', 'CD4']]

        Parameters:
        -----------
        indices: str, slice, list, tuple
            The indices to subset the image container.
        Returns:
        --------
        xarray.Dataset
            The subsetted image container.
        """

        # argument handling
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

        ds = self._obj.pp.get_channels(c_slice)
        return ds.pp.get_bbox(x_slice, y_slice)

    def get_bbox(self, x_slice: slice, y_slice: slice):
        """Returns the bounds of the image container."""

        # get the dimensionality of the image
        xdim = self._obj.coords[Dims.X]
        ydim = self._obj.coords[Dims.Y]

        # set the start and stop indices
        x_start = xdim[0] if x_slice.start is None else x_slice.start
        y_start = ydim[0] if y_slice.start is None else y_slice.start
        x_stop = xdim[-1] if x_slice.stop is None else x_slice.stop
        y_stop = ydim[-1] if y_slice.stop is None else y_slice.stop

        # set up query
        query = {
            Dims.X: x_slice,
            Dims.Y: y_slice,
        }

        # handle case when there are cells in the image
        if Dims.CELLS in self._obj.dims:
            num_cells = self._obj.dims[Dims.CELLS]

            coords = self._obj[Layers.OBS]
            cells = (
                (coords.loc[:, Features.X] >= x_start)
                & (coords.loc[:, Features.X] <= x_stop)
                & (coords.loc[:, Features.Y] >= y_start)
                & (coords.loc[:, Features.Y] <= y_stop)
            ).values
            # calculates the number of cells that were dropped due setting the bounding box
            lost_cells = num_cells - sum(cells)

            if lost_cells > 0:
                logger.warning(f"Dropped {lost_cells} cells.")

            # finalise query
            query[Dims.CELLS] = cells

        return self._obj.sel(query)

    def get_channels(self, channels: Union[List[str], str]):
        """Returns a single channel as a numpy array."""
        if isinstance(channels, str):
            channels = [channels]
        # build query
        query = {Dims.CHANNELS: channels}

        return self._obj.sel(query)

    def add_channel(self, channels: Union[str, list], array: np.ndarray):
        """
        Adds channel(s) to an existing image container.


        """
        assert type(array) is np.ndarray, "Added channel(s) must be numpy arrays"

        if array.ndim == 2:
            array = np.expand_dims(array, 0)

        if type(channels) is str:
            channels = [channels]

        self_channels, self_x_dim, self_y_dim = self._obj[Layers.IMAGE].shape
        other_channels, other_x_dim, other_y_dim = array.shape

        assert (
            len(channels) == other_channels
        ), "The length of channels must match the number of channels in array (DxMxN)."
        assert (self_x_dim == other_x_dim) & (self_y_dim == other_y_dim), "Dims do not match."

        da = xr.DataArray(
            array,
            coords=[channels, range(other_x_dim), range(other_y_dim)],
            dims=Dims.IMAGE,
            name=Layers.IMAGE,
        )
        # im = xr.concat([self._obj[Layers.IMAGE], da], dim=Dims.IMAGE[0])

        return xr.merge([self._obj, da])

    def add_segmentation(self, segmentation: np.ndarray, copy: bool = True) -> xr.Dataset:
        """Adds a segmentation mask (_segmentation) field to the xarray dataset.
        Expects an array of shape (x, y) that matches the shape of the image, ie.
        the x and y coordinates of the image container. Further, the mask is expected
        to mark the background with zeros and the cell instances with positive integers.


        Parameters:
        -----------
        segmentation: np.ndarray
            A segmentation mask, i.e. a np.ndarray with image.shape = (n, x, y),
            that indicates the location of each cell.
        copy: bool
            If true the segmentation mask is copied.

        Returns:
        --------
        xr.Dataset
            The amended xarray.
        """
        assert ~np.any(segmentation < 0), "A segmentation mask may not contain negative numbers."

        y_dim, x_dim = segmentation.shape

        assert (x_dim == self._obj.dims[Dims.X]) & (
            y_dim == self._obj.dims[Dims.Y]
        ), "The shape of segmentation mask does not match that of the image."

        if copy:
            segmentation = segmentation.copy()

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

        return xr.merge([obj, da])

    def add_observations(
        self,
        properties: Union[str, list, tuple] = ("label", "centroid"),
        return_xarray: bool = False,
    ) -> xr.Dataset:
        """Adds properties derived from the mask to the image container.

        Parameters:
        -----------
        properties: Union[str, list, tuple]
            A list of properties to be added to the image container. See
            skimage.measure.regionprops_table for a list of available properties.
        return_xarray: bool
            If true, the function returns an xarray.DataArray with the properties
            instead of adding them to the image container.
        key_added: str
            The key under which the properties are added to the image container,
            by default this is "_obs".

        Returns:
        --------
        xr.DataSet
            The amended image container.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found.")

        if type(properties) is str:
            properties = [properties]

        if "label" not in properties:
            properties = ["label", *properties]

        table = regionprops_table(self._obj[Layers.SEGMENTATION].values, properties=properties)

        label = table.pop("label")
        data = []
        cols = []

        for k, v in table.items():
            if Dims.FEATURES in self._obj.coords:
                if k in self._obj.coords[Dims.FEATURES] and not return_xarray:
                    logger.warning(f"Found {k} in _obs. Skipping.")
                    continue
            cols.append(k)
            data.append(v)

        if len(data) == 0:
            logger.warning("Warning: No properties were added.")
            return self._obj

        da = xr.DataArray(
            np.stack(data, -1),
            coords=[label, cols],
            dims=[Dims.CELLS, Dims.FEATURES],
            name=Layers.OBS,
        )

        if return_xarray:
            return da

        # if there are already observations, concatenate them
        if Layers.OBS in self._obj:
            logger.info("Found _obs in image container. Concatenating.")
            da = xr.concat(
                [self._obj[Layers.OBS].copy(), da],
                dim=Dims.FEATURES,
            )

        return xr.merge([self._obj, da])

    def add_quantification(
        self,
        channels: Union[str, list] = "all",
        func=sum_intensity,
        remove_unlabeled=True,
        key_added: str = Layers.INTENSITY,
        return_xarray=False,
    ):
        """
        Adds channel(s) to an existing image container.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found.")

        if key_added in self._obj:
            logger.warning(f"Found {key_added} in image container. Please add a different key.")
            return self._obj

        if Dims.CELLS not in self._obj.coords:
            logger.warning("No cell coordinates found. Adding _obs table.")
            self._obj = self._obj.pp.add_observations()

        measurements = []
        all_channels = self._obj.coords[Dims.CHANNELS].values.tolist()

        segmentation = self._obj[Layers.SEGMENTATION].values
        segmentation = _remove_unlabeled_cells(segmentation, self._obj.coords[Dims.CELLS].values)

        image = np.rollaxis(self._obj[Layers.IMAGE].values, 0, 3)
        props = regionprops_table(segmentation, intensity_image=image, extra_properties=(func,))
        cell_idx = props.pop("label")
        for k in sorted(props.keys(), key=lambda x: int(x.split("-")[-1])):
            if k.startswith(func.__name__):
                measurements.append(props[k])

        da = xr.DataArray(
            np.stack(measurements, -1),
            coords=[cell_idx, all_channels],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        if return_xarray:
            return da

        return xr.merge([self._obj, da])

    def add_quantification_from_dataframe(self, df: pd.DataFrame, key_added: str = Layers.INTENSITY) -> xr.Dataset:
        """Adds an observation table to the image container. Columns of the
        dataframe are added have to match the channel coords of the image
        container, and index of the dataframe has to match the cell coords
        of the image container.

        Parameters:
        -----------
        df: pd.DataFrame
            A dataframe with the quantification values.
        key_added: str
            The key under which the quantification is added to the image container,

        Returns:
        --------
        xr.DataSet
            The amended image container.
        """
        if Layers.SEGMENTATION not in self._obj:
            raise ValueError("No segmentation mask found. A segmentation mask is required to add quantification.")

        # pulls out the cell and channel coordinates from the image container
        cells = self._obj.coords[Dims.CELLS].values
        channels = self._obj.coords[Dims.CHANNELS].values

        # create a data array from the dataframe
        da = xr.DataArray(
            df.loc[cells, channels].values,
            coords=[cells, channels],
            dims=[Dims.CELLS, Dims.CHANNELS],
            name=key_added,
        )

        return xr.merge([self._obj, da])

    # def add_cell_type(self, num_cell_types: int):

    #     if Layers.LABELS not in self._obj:
    #         labels =  np.arange(1, num_cell_types + 1).reshape(-1, 1)
    #         props = np.full((num_cell_types, 2), -1)

    #         da = xr.DataArray(
    #             np.concatenate([labels, props], 1),
    #             coords=[np.arange(1, num_cell_types + 1), [Layers.LABELS, Props.NAME, Props.COLOR]],
    #             dims=[Dims.LABELS, Dims.PROPS],
    #             name=Layers.LABELS,
    #         )
    #     else:
    #         da = self.da[Layer.LABELS].coo

    #     return xr.merge([self._obj, da])

    def add_properties(self, array: Union[np.ndarray, list], prop: str = Features.LABELS, return_xarray: bool = False):
        unique_labels = np.unique(self._obj[Layers.OBS].sel({Dims.FEATURES: Features.LABELS}))

        if type(array) is list:
            array = np.array(array)

        if prop == Features.LABELS:
            unique_labels = np.unique(_format_labels(array))

        da = xr.DataArray(
            array.reshape(-1, 1),
            coords=[unique_labels.astype(int), [prop]],
            dims=[Dims.LABELS, Dims.PROPS],
            name=Layers.LABELS,
        )

        if return_xarray:
            return da

        if Layers.LABELS in self._obj:
            da = xr.concat(
                [self._obj[Layers.LABELS], da],
                dim=Dims.PROPS,
            )

        return xr.merge([da, self._obj])

    def add_labels(
        self,
        df: Union[pd.DataFrame, None] = None,
        cell_col: str = "cell",
        label_col: str = "label",
        colors: Union[list, None] = None,
        names: Union[list, None] = None,
    ):

        if df is None:
            cells = self._obj.coords[Dims.CELLS].values
            labels = np.ones(len(cells))
            formated_labels = np.ones(len(cells))
            unique_labels = np.unique(formated_labels)
        else:
            sub = df.loc[:, [cell_col, label_col]].dropna()
            cells = sub.loc[:, cell_col].values.squeeze()
            labels = sub.loc[:, label_col].values.squeeze()

            assert ~np.all(labels < 0), "Labels must be >= 0."

            formated_labels = _format_labels(labels)
            unique_labels = np.unique(formated_labels)

        if np.all(formated_labels == labels):
            da = xr.DataArray(
                formated_labels.reshape(-1, 1),
                coords=[cells, [Features.LABELS]],
                dims=[Dims.CELLS, Dims.FEATURES],
                name=Layers.OBS,
            )
        else:
            da = xr.DataArray(
                np.stack([formated_labels, labels], -1),
                coords=[
                    cells,
                    [
                        Features.LABELS,
                        Features.ORIGINAL_LABELS,
                    ],
                ],
                dims=[Dims.CELLS, Dims.FEATURES],
                name=Layers.OBS,
            )

        da = da.where(
            da.coords[Dims.CELLS].isin(
                self._obj.coords[Dims.CELLS],
            ),
            drop=True,
        )

        self._obj = xr.merge([self._obj.sel(cells=da.cells), da])

        if colors is not None:
            assert len(colors) == len(unique_labels), "Colors has the same."
        else:
            colors = np.random.choice(COLORS, size=len(unique_labels), replace=False)

        self._obj = self._obj.pp.add_properties(colors, Props.COLOR)

        if names is not None:
            assert len(names) == len(unique_labels), "Names has the same."
        else:
            names = [f"Cell type {i+1}" for i in range(len(unique_labels))]

        self._obj = self._obj.pp.add_properties(names, Props.NAME)
        self._obj[Layers.SEGMENTATION].values = _remove_unlabeled_cells(
            self._obj[Layers.SEGMENTATION].values, self._obj.coords[Dims.CELLS].values
        )

        return xr.merge([self._obj.sel(cells=da.cells), da])

    def normalize(self):
        """Performs a percentile normalisation on each channel.

        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in Layers.PLOT.
        """
        image_layer = self._obj[Layers.IMAGE]
        normed = xr.DataArray(
            _normalize(image_layer.values),
            coords=image_layer.coords,
            dims=[Dims.CHANNELS, Dims.Y, Dims.X],
            name=Layers.PLOT,
        )

        return xr.merge([self._obj, normed])

    def colorize(
        self,
        colors: List[str] = ["C0", "C1", "C2", "C3"],
        background: str = "black",
        normalize: bool = True,
        merge=True,
    ) -> xr.Dataset:
        """Colorizes a stack of images.

        Parameters
        ----------
        colors: List[str]
            A list of strings that denote the color of each channel.
        background: float
            Background color of the colorized image.
        normalize: bool
            Normalizes the image prior to colorizing it.
        merge: True
            Merge the channel dimension.


        Returns
        -------
        xr.Dataset
            The image container with the colorized image stored in Layers.PLOT.
        """

        image_layer = self._obj[Layers.IMAGE]
        colored = _colorize(
            image_layer.values,
            colors=colors,
            background=background,
            normalize=normalize,
        )
        da = xr.DataArray(
            colored,
            coords=[
                image_layer.coords[Dims.CHANNELS],
                image_layer.coords[Dims.Y],
                image_layer.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.CHANNELS, Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs={Attrs.IMAGE_COLORS: {k.item(): v for k, v in zip(image_layer.coords[Dims.CHANNELS], colors)}},
        )

        if merge:
            da = da.sum(Dims.CHANNELS, keep_attrs=True)
            da.values[da.values > 1] = 1.0

        return xr.merge([self._obj, da])
