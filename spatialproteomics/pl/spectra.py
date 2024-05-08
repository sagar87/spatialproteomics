import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from ..constants import Dims


def format_annotation_df(annotation, da, index_col="idx", channel_col="name"):
    df = (
        da.coords[Dims.CHANNELS]
        .to_dataframe()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={Dims.CHANNELS: channel_col})
    )

    new_annot = (
        annotation.drop(columns=index_col).merge(df, on=channel_col, how="left").rename(columns={"index": index_col})
    )
    return new_annot


def plot_expression_spectra(
    array,
    annotations,
    titles=None,
    height=3,
    width=6,
    sorting=None,
    ylabel="",
    hspace=0.5,
    wspace=0.08,
    sharey=False,
    ncols=2,
    bottom=True,
    xticks=False,
    categories_col="category",
    index_col="idx",
    xlabel_col="name",
    sorting_col="sorting",
    color_col="color",
):
    if array.ndim == 3:
        spectra_dim = 1
    elif array.ndim == 2:
        spectra_dim = 0

    nrows = int(np.ceil(array.shape[spectra_dim] / ncols))
    fig = plt.figure(figsize=(ncols * width, nrows * height))
    outer_grid = fig.add_gridspec(nrows, ncols, hspace=hspace, wspace=wspace)
    all_axes = []

    if sorting is None:
        sorting = np.arange(array.shape[spectra_dim])

    for ii, i in enumerate(sorting):
        j, k = np.unravel_index(ii, (nrows, ncols))

        if spectra_dim == 1:
            arr = array[:, i]
        else:
            arr = array[i]

        if xticks is False:
            xticklabels = False if ii < (len(sorting) - ncols) else True
        else:
            xticklabels = True

        axes = plot_expression_spectrum(
            arr,
            annotations,
            outer_grid=outer_grid[j, k],
            fig=fig,
            xticks=xticklabels,
            ylabel=ylabel if k == 0 else "",
            title=f"Signature {i}" if titles is None else titles[ii],
            bottom=bottom,
            categories_col=categories_col,
            index_col=index_col,
            xlabel_col=xlabel_col,
            sorting_col=sorting_col,
            color_col=color_col,
        )
        all_axes += axes

    if sharey:
        axes_max = 0

        for ax in all_axes:
            if ax.get_ylim()[-1] > axes_max:
                axes_max = ax.get_ylim()[-1]

        for ax in all_axes:
            ax.set_ylim([0, axes_max])

    return all_axes


def plot_expression_spectrum(
    array,
    annotations,
    ylabel="Probability",
    title=None,
    xticks=True,
    outer_grid=None,
    fig=None,
    categories_col="category",
    index_col="idx",
    xlabel_col="name",
    sorting_col="sorting",
    color_col="color",
    bottom=True,
    figsize=(9, 2),
):
    error = False
    spectrum_dim = 0
    if array.ndim == 2:
        error = True
        spectrum_dim = 1

    if outer_grid is None:
        fig = plt.figure(figsize=figsize)
        grid = gs.GridSpec(1, array.shape[spectrum_dim], figure=fig)
    else:
        grid = outer_grid.subgridspec(1, array.shape[spectrum_dim])

    plot_frame = annotations.groupby([categories_col])[index_col].count().reset_index()
    grid_bounds = plot_frame[~(plot_frame[index_col] == 0)]

    axes = []
    axes_max = 0
    axes_min = 0
    last_bound = 0
    for i, grid_bound in grid_bounds.iterrows():
        sub_frame = annotations[(annotations[categories_col] == grid_bound[categories_col])]

        if sorting_col is not None:
            sub_frame = sub_frame.sort_values(by=sorting_col)

        ax1 = fig.add_subplot(grid[0, last_bound : last_bound + grid_bound[index_col]])
        last_bound += grid_bound[index_col]
        # print(sub_frame, sub_frame[index_col].values)
        ee = array[..., sub_frame[index_col].values]

        if color_col is not None:
            cc = sub_frame[color_col]
        else:
            cc = "C0"

        if error:
            bars = np.percentile(ee, [2.5, 97.5], axis=0)
            ee = ee.mean(0)
            bars = np.abs(bars - ee)
            ax1.bar(np.arange(ee.shape[0]), ee, color=cc, yerr=bars)
        else:
            ax1.bar(np.arange(ee.shape[0]), ee, color=cc)

        ax1.set_xticks(np.arange(ee.shape[0]))
        if xticks:
            ax1.set_xticklabels(sub_frame[xlabel_col].tolist(), rotation=90)
        else:
            ax1.set_xticklabels([])

        ax1.axes.spines["right"].set_visible(False)
        ax1.axes.spines["top"].set_visible(False)

        if i == 0:
            ax1.set_ylabel(ylabel)

        if title is not None and i == 2:
            ax1.set_title(title)

        if i > 0:
            ax1.margins(0.01)
            ax1.axes.spines["left"].set_visible(False)
            ax1.set_yticks([])

        if ax1.get_ylim()[-1] > axes_max:
            axes_max = ax1.get_ylim()[-1]

        if ax1.get_ylim()[0] < axes_min:
            axes_min = ax1.get_ylim()[0]

        axes.append(ax1)

    for ax in axes:
        if bottom:
            ax.set_ylim([0, axes_max])
        else:
            ax.set_ylim(bottom=axes_min, top=axes_max)

    return axes
