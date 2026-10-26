Core concepts and cheatsheet
============================

This page summarizes how spatialproteomics is organized and how its API is meant to be used.
It is a compact reference for both humans and AI coding assistants; the tutorials cover each topic in depth.

The data model
--------------

A spatialproteomics object is a plain ``xarray.Dataset``.
All data belonging to one sample lives in this single object as named data variables ("layers") that share dimensions:

=======================  =============================  ============================================================
Layer (``sp.Layers.*``)  Dimensions                     Content
=======================  =============================  ============================================================
``_image``               ``(channels, y, x)``           Multiplexed image; one channel per marker.
``_segmentation``        ``(y, x)``                     Integer cell mask; ``0`` is background, ``n`` is cell ``n``.
``_obs``                 ``(cells, features)``          Per-cell observations (centroids, area, cell type, neighborhood).
``_intensity``           ``(cells, channels)``          Expression matrix produced by ``pp.add_quantification``.
``_la_properties``       ``(labels, la_props)``         Cell type names and colors.
``_neighborhoods``       ``(cells, labels)``            Cell type composition of each cell's neighborhood.
``_nh_properties``       ``(neighborhoods, nh_props)``  Neighborhood names and colors.
=======================  =============================  ============================================================

You can add layers of your own with ``ds.pp.add_layer(array, key_added="_name")``.
A common one is ``_raw_image``: an untouched copy of ``_image`` kept before thresholding,
because ``pp.threshold`` overwrites ``_image`` (see the workflow and examples below).

Because these layers share coordinates, subsetting is synchronized:
cropping the image also crops the segmentation and drops cells outside the crop from ``_obs`` and ``_intensity``;
selecting a cell type removes the other cells from the mask and all tables.

Use the constants in ``sp.Layers``, ``sp.Dims``, ``sp.Features`` and ``sp.Props`` instead of hard-coding strings,
e.g. ``sp.Features.LABELS == "_labels"`` (the cell type column in ``_obs``), ``sp.Features.X == "centroid-1"``.

Accessors
---------

Importing spatialproteomics registers five accessors on every ``xarray.Dataset``:

======  =====================  =========================================================================================
Name    Class                  Responsibility
======  =====================  =========================================================================================
``pp``  PreprocessingAccessor  Subsetting, image processing, segmentation handling, quantification, tables.
``tl``  ToolAccessor           Wrappers around external tools (cellpose, stardist, instanseg, astir, spotiflow) and export.
``la``  LabelAccessor          Cell type labels: prediction, naming, colors, subtype hierarchies.
``nh``  NeighborhoodAccessor   Cellular neighborhoods and graph features.
``pl``  PlotAccessor           Plotting of images, masks, labels, neighborhoods and observations.
======  =====================  =========================================================================================

Every accessor method returns a **new** dataset; nothing is modified in place.
Always reassign (``ds = ds.pp.threshold(...)``) or chain calls:

.. code-block:: python

   ds = ds.pp.threshold(quantile=0.5).pp.add_quantification().pp.transform_expression_matrix()

Indexing
--------

.. code-block:: python

   ds.pp["DAPI"]                        # one channel
   ds.pp[["CD4", "CD8"]]                # several channels
   ds.pp[100:200, 300:400]              # spatial crop (x slice, y slice); cells outside are removed
   ds.pp[["CD4", "CD8"], 100:200, 300:400]
   ds.la["B cell"]                      # cells of one type (by name or integer label id)
   ds.la[["B cell", "T cell"]]
   ds.la.deselect(["B cell"])           # all cells except these types
   ds.nh["Neighborhood 1"]              # cells of one neighborhood

Canonical workflow (xarray backend)
-----------------------------------

.. code-block:: python

   import spatialproteomics as sp
   import xarray as xr
   from skimage.io import imread

   # 1. create the object from a (channels, y, x) array
   image = imread("input.tiff")
   ds = sp.load_image_data(image, channel_coords=["DAPI", "PAX5", "CD3", "CD4", "CD8"])

   # 2. segmentation (requires `pip install cellpose`); adds `_segmentation` and `_obs`
   ds = ds.tl.cellpose(channel="DAPI")
   ds = ds.pp.add_observations("area")
   ds = ds.pp.filter_by_obs("area", func=lambda x: (x > 20) & (x < 150))
   ds = ds.pp.grow_cells(iterations=2)

   # 3. image processing: per-channel thresholding (overwrites `_image`)
   # `quantile=` is relative to each channel's intensity distribution, `intensity=` is an absolute value;
   # pass one value per channel, or one value plus `channels=[...]` to threshold only some channels.
   # Thresholds require human inspection: plot the raw image, the thresholded image and the resulting
   # cell types (step 5) side by side, as in the "Cell type QC plot" example below.
   ds = ds.pp.add_layer(ds["_image"].values.copy(), key_added="_raw_image")  # keep the original for comparison
   ds = ds.pp.threshold(quantile=[0.2, 0.6, 0.6, 0.6, 0.6])

   # 4. quantification; adds `_intensity`
   ds = ds.pp.add_quantification(func="intensity_mean").pp.transform_expression_matrix(method="arcsinh")

   # 5. cell types; maps marker -> cell type
   ds = ds.la.predict_cell_types_argmax({"PAX5": "B cell", "CD3": "T cell"})
   ds = ds.la.set_label_colors(["B cell", "T cell"], ["red", "green"])

   # 6. binarization and subtypes; maps marker -> subtype
   # quantify the fraction of positive (non-zero) pixels per cell on the thresholded image, then threshold
   # that fraction; this adds a binary `{marker}_binarized` column to `_obs` (e.g. `CD4_binarized`).
   # See the "Binarization QC plot" example below for checking the result visually.
   ds = ds.pp.add_quantification(func=sp.percentage_positive, key_added="_percentage_positive")
   ds = ds.la.threshold_labels({"CD4": 0.5, "CD8": 0.5}, layer_key="_percentage_positive")
   ds = ds.la.predict_cell_subtypes({"T cell": {"CD4": "T_h", "CD8": "T_tox"}})
   ds = ds.la.set_label_colors(["B cell", "T cell", "T_h", "T_tox"], ["red", "green", "yellow", "blue"])

   # 7. plotting (all plotting functions accept `ax=`)
   # plotting an image
   ds.pp[["CD4", "CD8"]].pl.colorize(["red", "green"]).pl.show()
   # plotting the raw image instead of the thresholded one
   ds.pp[["CD4", "CD8"]].pl.colorize(["red", "green"], layer_key="_raw_image").pl.show()
   # plotting a segmentation mask and the labels
   ds.pl.show(render_image=False, render_segmentation=True, render_labels=True)

   # 8. tables and export
   obs = ds.pp.get_layer_as_df("_obs")               # cell types decoded to names
   expr = ds.pp.get_layer_as_df("_intensity")
   ds.drop_encoding().to_zarr("sample.zarr")         # reload with xr.open_zarr("sample.zarr")
   adata = ds.tl.convert_to_anndata()                # requires anndata
   sdata = ds.tl.convert_to_spatialdata()            # requires spatialdata

Other frequently used operations:

* Segmentation from elsewhere: ``ds.pp.add_segmentation(mask)``; alternatives to cellpose: ``ds.tl.stardist()`` and ``ds.tl.instanseg()`` (segments all selected channels jointly)
* Labels from elsewhere: ``ds.la.add_labels_from_dataframe(df, cell_col="cell", label_col="label")``.
* Marker binarization and subtypes: ``ds.la.threshold_labels({...})`` then ``ds.la.predict_cell_subtypes({...})``; switch levels with ``ds.la.set_label_level(...)``.
* Neighborhoods across samples: ``sp.ImageContainer({"s1": ds1, "s2": ds2}).compute_neighborhoods(k=5)`` computes compositions,
  clusters them with k-means and returns a dict of datasets with neighborhoods assigned (``ds.nh["Neighborhood 0"]``).
* Composable plotting: ``ds.pl.colorize([...]).pl.render_segmentation().pl.render_labels().pl.imshow()``; scatter plots with ``ds.pl.scatter_labels()`` / ``ds.pl.scatter(feature)``.
* Large images: ``ds.pp.downsample(rate=8)`` before plotting, ``xr.open_zarr`` for lazy loading.

The spatialdata backend
-----------------------

If you start from a ``spatialdata.SpatialData`` object (``pip install "spatialproteomics[spatialdata]"``),
use the module-level functions instead of the accessors.
They follow scverse conventions: they take the object as first argument and **modify it in place** unless ``copy=True``.

.. code-block:: python

   import spatialproteomics as sp

   sp.tl.cellpose(sdata, channel="DAPI")                    # writes sdata.labels["segmentation"]
   sp.pp.threshold(sdata, quantile=[0.2, 0.6, 0.6, 0.6, 0.6])
   sp.pp.add_quantification(sdata, func="intensity_mean")    # writes sdata.tables["table"]
   sp.pp.transform_expression_matrix(sdata, method="arcsinh")
   sp.la.predict_cell_types_argmax(sdata, {"CD4": "T_h", "CD8": "T_tox"})  # marker -> cell type; adds obs["celltype"]

Convert between the two backends with ``sp.read_from_spatialdata(sdata)`` and ``ds.tl.convert_to_spatialdata()``.
The xarray backend offers the full feature set; the spatialdata backend covers the core pipeline.

Common pitfalls
---------------

* The accessors only exist after ``import spatialproteomics``, including on objects loaded with ``xr.open_zarr``.
* Accessor methods return new objects. ``ds.pp.threshold(quantile=0.5)`` without reassignment has no effect.
* ``pp.threshold`` overwrites ``_image``. Keep a copy (``_raw_image``) first if you want to compare raw and thresholded images.
* ``la.predict_cell_types_argmax`` and ``la.predict_cell_subtypes`` take **marker -> cell type** mappings,
  e.g. ``{"CD4": "T_h", "CD8": "T_tox", "FOXP3": "T_reg", "BCL6": "T_fh"}``, not the reverse.
* ``ds.pp[a:b, c:d]`` slices **x first, then y**, while the image array is stored as ``(channels, y, x)``.
  Slices select by coordinate value (like ``xarray.Dataset.sel``), so the end point is included.
* Call ``drop_encoding()`` before ``to_zarr()`` to avoid xarray encoding errors.
* Optional tools must be installed separately: ``cellpose``, ``stardist``, ``instanseg-torch``, ``astir`` and ``spotiflow``
  (or all at once with ``pip install "spatialproteomics[all]"``).
* Many methods require earlier steps: e.g. ``la.predict_cell_types_argmax`` needs ``_intensity``, which needs ``_segmentation``.
  Inspect the object (``print(ds)``) to see which layers are present.

Examples
--------

Cell type QC plot
~~~~~~~~~~~~~~~~~

Thresholding, argmax cell type prediction and a side-by-side plot of raw image, thresholded image and
resulting cell types per marker, for a human to check the thresholds.

.. code-block:: python

   import matplotlib.pyplot as plt
   import xarray as xr
   import spatialproteomics  # noqa: F401  (registers the .pp / .la / .pl accessors)

   # --- Inputs (replace with your own) -----------------------------------------
   # marker -> display color
   marker_colors = {"CD3": "red", "CD20": "green", "CD68": "blue"}
   # marker -> cell type it identifies
   marker_to_celltype = {"CD3": "T cell", "CD20": "B cell", "CD68": "Macrophage"}
   # cell type -> display color
   celltype_colors = {"T cell": "red", "B cell": "green", "Macrophage": "blue"}
   # channel -> absolute intensity threshold
   thresholds = {"CD3": 0.3, "CD20": 0.25, "CD68": 0.4}


   def load_dataset(zarr_path, markers):
       """Open a segmented zarr and keep an untouched copy of the image."""
       ds = xr.open_zarr(zarr_path).pp[["DAPI"] + list(markers)]
       return ds.pp.add_layer(ds["_image"].values.copy(), key_added="_raw_image")


   def predict_cell_types(ds):
       """Threshold, quantify, transform and assign one cell type per cell."""
       ds = (
           ds.pp.threshold(intensity=list(thresholds.values()), channels=list(thresholds.keys()))
           .pp.add_quantification(func="intensity_mean")
           .pp.transform_expression_matrix(method="arcsinh")
           .la.predict_cell_types_argmax(marker_to_celltype)
       )
       return ds.la.set_label_colors(celltype_colors.keys(), celltype_colors.values()).pl.autocrop()


   def plot_markers_and_celltypes(ds, scaling=5):
       """Overview row, then one row per marker: raw + segmentation, raw, thresholded, cell type."""
       markers, colors = list(marker_colors.keys()), list(marker_colors.values())
       nrows, ncols = len(markers) + 1, 4
       fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * scaling, nrows * scaling))

       # Overview row
       ds.pp["DAPI"].pl.colorize("gold").pl.show(render_segmentation=True, ax=axes[0, 0])
       ds.pp[markers].pl.colorize(colors, layer_key="_raw_image").pl.show(ax=axes[0, 1])
       ds.pp[markers].pl.colorize(colors).pl.show(ax=axes[0, 2])
       ds.pl.show(render_image=False, render_labels=True, ax=axes[0, 3])
       for ax, title in zip(axes[0], ["Segmentation", "Raw", "Thresholded", "Cell types"]):
           ax.set_title(title)

       # One row per marker
       for row, (marker, color) in zip(axes[1:], marker_colors.items()):
           celltype = marker_to_celltype[marker]
           raw = ds.pp[marker].pl.colorize(color, layer_key="_raw_image")
           try:
               # outline only the cells of the matching type
               raw.la[celltype].pl.show(render_segmentation=True, ax=row[0])
           except ValueError:  # no cells of this type
               raw.pl.show(ax=row[0])
           raw.pl.show(ax=row[1])
           ds.pp[marker].pl.colorize(color).pl.show(ax=row[2])
           try:
               ds.la[celltype].pl.show(render_image=False, render_labels=True, ax=row[3])
           except ValueError:
               pass  # leave the panel empty
           for ax, title in zip(row, [marker, marker, marker, celltype]):
               ax.set_title(title)

       for ax in axes.flat:
           ax.axis("off")
       plt.tight_layout()
       plt.show()


   if __name__ == "__main__":
       ds = load_dataset("path/to/sample.zarr", marker_colors.keys())
       ds = predict_cell_types(ds)
       plot_markers_and_celltypes(ds)

Binarization QC plot
~~~~~~~~~~~~~~~~~~~~

Binarizing markers per cell and plotting each step (raw image, pixel-thresholded image, cells called positive)
for a human to check the two thresholds.

.. code-block:: python

   import matplotlib.pyplot as plt
   import xarray as xr
   import spatialproteomics as sp

   # --- Inputs (replace with your own) -----------------------------------------
   # marker -> (absolute pixel intensity threshold, minimum fraction of positive pixels per cell)
   # Plotting order follows the order of this dict.
   thresholds = {
       "Ki-67": (0.3, 0.1),
       "CD8": (0.25, 0.2),
       "PD-1": (0.4, 0.15),
   }


   def load_dataset(zarr_path):
       """Open a segmented zarr and keep an untouched copy of the image."""
       ds = xr.open_zarr(zarr_path)
       ds = ds.pp.add_layer(ds["_image"].values.copy(), key_added="_raw_image")
       return ds.pl.autocrop()


   def binarize_marker(ds, marker, intensity_threshold, perc_pos_threshold):
       """Threshold pixels, compute % positive pixels per cell, then binarize cells."""
       return (
           ds.pp[marker]
           .pp.threshold(intensity=intensity_threshold)
           .pp.add_quantification(func=sp.percentage_positive, key_added="_percentage_positive")
           .la.threshold_labels({marker: perc_pos_threshold}, layer_key="_percentage_positive")
       )


   def plot_binarization(ds, scaling=5):
       """One row per marker: raw image, thresholded image, positive cells outlined."""
       nrows, ncols = len(thresholds), 3
       fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * scaling, nrows * scaling), squeeze=False)

       for row, (marker, (intensity_thr, perc_pos_thr)) in zip(axes, thresholds.items()):
           ds.pp[marker].pl.colorize("white", layer_key="_raw_image").pl.show(ax=row[0])

           ds_bin = binarize_marker(ds, marker, intensity_thr, perc_pos_thr)
           ds_bin.pp[marker].pl.colorize("white").pl.show(ax=row[1])

           # keep only cells called positive (`{marker}_binarized == 1`) and outline them
           (
               ds_bin.pp[marker]
               .pp.filter_by_obs(f"{marker}_binarized", func=lambda x: x == 1)
               .pl.colorize("white")
               .pl.show(
                   render_segmentation=True,
                   segmentation_kwargs={"alpha": 0.5, "colors": ["gold"]},
                   ax=row[2],
               )
           )

           for ax, title in zip(row, ["Raw", "Thresholded", "Binarized"]):
               ax.set_title(f"{marker} – {title}")

       for ax in axes.flat:
           ax.axis("off")
       plt.tight_layout()
       plt.show()


   if __name__ == "__main__":
       ds = load_dataset("path/to/sample.zarr")
       plot_binarization(ds)
