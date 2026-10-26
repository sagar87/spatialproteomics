import numpy as np
import pandas as pd

from spatialproteomics.constants import COLORS, Dims, Layers, Props
from spatialproteomics.pl.utils import _sample_colors


def test_sample_colors_from_palette():
    colors = _sample_colors(10)
    assert len(colors) == 10
    assert len(set(colors)) == 10
    assert set(colors).issubset(COLORS)


def test_sample_more_colors_than_palette():
    n = len(COLORS) + 50
    colors = _sample_colors(n)
    assert len(colors) == n
    assert len({c.upper() for c in colors}) == n
    assert set(COLORS).issubset(colors)


def _ds_with_many_cells(ds_image, n_cells):
    # a segmentation mask with one small square cell per 5x5 block
    segmentation = np.zeros((ds_image.sizes[Dims.Y], ds_image.sizes[Dims.X]), dtype=int)
    n_per_row = ds_image.sizes[Dims.X] // 5
    for i in range(n_cells):
        y, x = divmod(i, n_per_row)
        segmentation[5 * y : 5 * y + 3, 5 * x : 5 * x + 3] = i + 1
    return ds_image.pp.add_segmentation(segmentation)


def test_add_more_labels_than_colors(ds_image):
    n = len(COLORS) + 1
    ds = _ds_with_many_cells(ds_image, n)
    df = pd.DataFrame({"cell": np.arange(1, n + 1), "label": [f"type_{i}" for i in range(n)]})

    ds = ds.la.add_labels_from_dataframe(df)

    assert ds.sizes[Dims.LABELS] == n
    colors = ds[Layers.LA_PROPERTIES].sel({Dims.LA_PROPS: Props.COLOR}).values
    assert len(set(colors)) == n


def test_add_more_neighborhoods_than_colors(ds_image):
    n = len(COLORS) + 1
    ds = _ds_with_many_cells(ds_image, n)
    df = pd.DataFrame({"neighborhood": np.arange(n)})

    ds = ds.nh.add_neighborhoods_from_dataframe(df, neighborhood_col="neighborhood")

    assert ds.sizes[Dims.NEIGHBORHOODS] == n
    colors = ds[Layers.NH_PROPERTIES].sel({Dims.NH_PROPS: Props.COLOR}).values
    assert len(set(colors)) == n
