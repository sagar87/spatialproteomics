from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
from skimage.segmentation import find_boundaries, relabel_sequential

from ..base_logger import logger
from ..constants import COLORS, Dims, Features, Layers, Props
from ..pl import _get_listed_colormap
from ..se.segmentation import _remove_unlabeled_cells

# from tqdm import tqdm


def _format_labels(labels):
    """Formats a label list."""
    formatted_labels = labels.copy()
    unique_labels = np.unique(labels)

    if 0 in unique_labels:
        logger.warning("Found 0 in labels. Reindexing ...")
        formatted_labels += 1

    if ~np.all(np.diff(unique_labels) == 1):
        logger.warning("Labels are non-consecutive. Relabeling ...")
        formatted_labels, _, _ = relabel_sequential(formatted_labels)

    return formatted_labels


def _label_segmentation_mask(segmentation: np.ndarray, annotations: dict) -> np.ndarray:
    """
    Relabels a segmentation according to the annotations df (contains the columns type, cell).
    """
    labeled_segmentation = segmentation.copy()
    all_cells = []

    for k, v in annotations.items():
        mask = np.isin(segmentation, v)
        labeled_segmentation[mask] = k
        all_cells.extend(v)

    # remove cells that are not indexed
    neg_mask = ~np.isin(segmentation, all_cells)
    labeled_segmentation[neg_mask] = 0

    return labeled_segmentation


def _render_label(mask, cmap_mask, img=None, alpha=0.2, alpha_boundary=0, mode="inner"):
    colored_mask = cmap_mask(mask)

    mask_bool = mask > 0
    mask_bound = np.bitwise_and(mask_bool, find_boundaries(mask, mode=mode))

    # blend
    if img is None:
        img = np.zeros(mask.shape + (4,), np.float32)
        img[..., -1] = 1

    im = img.copy()

    im[mask_bool] = alpha * colored_mask[mask_bool] + (1 - alpha) * img[mask_bool]
    im[mask_bound] = (
        alpha_boundary * colored_mask[mask_bound]
        + (1 - alpha_boundary) * img[mask_bound]
    )

    return im


@xr.register_dataset_accessor("la")
class LabelAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _relabel_dict(self, dictionary: dict):
        _, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values)
        return {fw[k]: v for k, v in dictionary.items()}

    def _label_to_dict(self, prop: str, relabel: bool = False):
        labels_layer = self._obj[Layers.LABELS]
        label_dict = {
            label.item(): labels_layer.loc[label, prop].item()
            for label in self._obj.coords[Dims.LABELS]
        }

        if relabel:
            return self._obj.la._relabel_dict(label_dict)

        return label_dict

    def _cells_to_label(self, relabel: bool = False):
        """Returns a dictionary that maps each label to a list of cells."""
        label_dict = {
            label.item(): self._obj.la._filter_cells_by_label(label.item())
            for label in self._obj.coords[Dims.LABELS]
        }

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

    def __getitem__(self, indices):
        """
        Sub selects labels.
        """
        # type checking
        # TODO: Write tests!
        if type(indices) is slice:
            l_start = indices.start if indices.start is not None else 1
            l_stop = (
                indices.stop
                if indices.stop is not None
                else self._obj.dims[Dims.LABELS]
            )
            sel = [i for i in range(l_start, l_stop)]
        elif type(indices) is list:
            all_int = all([type(i) is int for i in indices])
            assert all_int, "All label indices must be integers."
            sel = indices

        elif type(indices) is tuple:
            indices = list(indices)
            all_int = all([type(i) is int for i in indices])
            assert all_int, "All label indices must be integers."
            sel = indices
        else:
            assert (
                type(indices) is int
            ), "Label must be provided as slices, lists, tuple or int."

            sel = [indices]

        cells = self._obj.la._filter_cells_by_label(sel)
        return self._obj.sel({Dims.LABELS: sel, Dims.CELLS: cells})

    def add_labels(
        self,
        df: pd.DataFrame,
        cell_col: str = "cell",
        label_col: str = "label",
        colors: Union[list, None] = None,
        names: Union[list, None] = None,
    ):
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

        self._obj = self._obj.la.add_props(colors, Props.COLOR)

        if names is not None:
            assert len(names) == len(unique_labels), "Names has the same."
        else:
            names = [f"Cell type {i+1}" for i in range(len(unique_labels))]

        self._obj = self._obj.la.add_props(names, Props.NAME)
        self._obj[Layers.SEGMENTATION].values = _remove_unlabeled_cells(
            self._obj[Layers.SEGMENTATION].values, self._obj.coords[Dims.CELLS].values
        )

        return self._obj

    def add_props(self, array: Union[np.ndarray, list], prop: str):
        unique_labels = np.unique(
            self._obj[Layers.OBS].sel({Dims.FEATURES: Features.LABELS})
        )

        if type(array) is list:
            array = np.array(array)

        da = xr.DataArray(
            array.reshape(-1, 1),
            coords=[unique_labels.astype(int), [prop]],
            dims=[Dims.LABELS, Dims.PROPS],
            name=Layers.LABELS,
        )

        if Layers.LABELS in self._obj:
            da = xr.concat(
                [self._obj[Layers.LABELS], da],
                dim=Dims.PROPS,
            )

        return xr.merge([da, self._obj])

    def set_label_name(self, label, name):
        self._obj[Layers.LABELS].loc[label, Props.NAME] = name

    def set_label_color(self, label, color):
        self._obj[Layers.LABELS].loc[label, Props.COLOR] = color

    def render_label(self, alpha=0.2, alpha_boundary=0, mode="inner"):
        assert (
            Layers.LABELS in self._obj
        ), "Add labels via the add_labels function first."

        # TODO: Attribute class in constants.py
        color_dict = self._label_to_dict(Props.COLOR, relabel=True)
        cmap = _get_listed_colormap(color_dict)

        cells_dict = self._cells_to_label(relabel=True)
        segmentation = self._obj[Layers.SEGMENTATION].values
        mask = _label_segmentation_mask(segmentation, cells_dict)

        if Layers.PLOT in self._obj:
            attrs = self._obj[Layers.PLOT].attrs
            rendered = _render_label(
                mask,
                cmap,
                self._obj[Layers.PLOT].values,
                alpha=alpha,
                alpha_boundary=alpha_boundary,
                mode=mode,
            )
            self._obj = self._obj.drop_vars(Layers.PLOT)
        else:
            attrs = {}
            rendered = _render_label(
                mask, cmap, alpha=alpha, alpha_boundary=alpha_boundary, mode=mode
            )

        da = xr.DataArray(
            rendered,
            coords=[
                self._obj.coords[Dims.Y],
                self._obj.coords[Dims.X],
                ["r", "g", "b", "a"],
            ],
            dims=[Dims.Y, Dims.X, Dims.RGBA],
            name=Layers.PLOT,
            attrs=attrs,
        )

        return xr.merge([self._obj, da])

    # def add_labels(
    #     self,
    #     labels,
    #     cell_col: str = "cell",
    #     label_col: str = "label",
    #     color_dict: Union[None, dict] = None,
    #     names_dict: Union[None, dict] = None,
    # ):
    #     num_cells = self._obj.dims[Dims.CELLS]

    #     # select the cells indices which are consistent with the segmentation
    #     df = (
    #         pd.DataFrame(index=self._obj.coords[Dims.CELLS].values)
    #         .reset_index()
    #         .rename(columns={"index": cell_col})
    #     )

    #     df = df.merge(labels, on=cell_col, how="inner")

    #     array = df.loc[:, label_col].values

    #     if 0 in array:
    #         logger.warning(
    #             "Found '0' as cell type as label, reindexing. Please ensure that cell type labels are consecutive integers (1, 2, ..., k) starting from 1."
    #         )
    #         array += 1

    #     unique_labels = np.unique(array)
    #     attrs = {}

    #     # set up the meta data
    #     if color_dict is None:
    #         logger.warning("No label colors specified. Choosing random colors.")
    #         attrs[Attrs.LABEL_COLORS] = {
    #             k: v
    #             for k, v in zip(
    #                 unique_labels,
    #                 np.random.choice(COLORS, size=len(unique_labels), replace=False),
    #             )
    #         }
    #     else:
    #         attrs[Attrs.LABEL_COLORS] = color_dict

    #     if names_dict is None:
    #         attrs[Attrs.LABEL_NAMES] = {k: f"Cell type {k}" for k in unique_labels}
    #     else:
    #         attrs[Attrs.LABEL_NAMES] = names_dict

    #     da = xr.DataArray(
    #         array.reshape(-1, 1),
    #         coords=[df.loc[:, cell_col].values, ["label"]],
    #         dims=[Dims.CELLS, Dims.LABELS],
    #         name=Layers.LABELS,
    #         attrs=attrs,
    #     )

    #     # merge datasets
    #     ds = self._obj.merge(da, join="inner")
    #     ds.se._remove_unlabeled_cells()

    #     lost_cells = num_cells - ds.dims[Dims.CELLS]

    #     if lost_cells > 0:
    #         logger.warning(f"No cell label for {lost_cells} cells. Dropping cells.")

    #     return ds
