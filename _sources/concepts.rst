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
``tl``  ToolAccessor           Wrappers around external tools (cellpose, stardist, mesmer, astir, spotiflow) and export.
``la``  LabelAccessor          Cell type labels: prediction, naming, colors, subtype hierarchies.
``nh``  NeighborhoodAccessor   Cellular neighborhoods and graph features.
``pl``  PlotAccessor           Plotting of images, masks, labels, neighborhoods and observations.
======  =====================  =========================================================================================

Every accessor method returns a **new** dataset; nothing is modified in place.
Always reassign (``ds = ds.pp.threshold(...)``) or chain calls:

.. code-block:: python

   ds = ds.pp.threshold(0.5).pp.add_quantification().pp.transform_expression_matrix()

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
   from scipy.signal import medfilt2d
   from skimage.io import imread

   # 1. create the object from a (channels, y, x) array
   image = imread("input.tiff")
   ds = sp.load_image_data(image, channel_coords=["DAPI", "CD4", "CD8", "FOXP3", "BCL6"])

   # 2. image processing: per-channel quantile thresholding, then any 2D function
   ds = ds.pp.threshold([0.2, 0.6, 0.6, 0.6, 0.6]).pp.apply(medfilt2d, kernel_size=3)

   # 3. segmentation (requires `pip install cellpose`); adds `_segmentation` and `_obs`
   ds = ds.tl.cellpose(channel="DAPI")
   ds = ds.pp.add_observations("area")
   ds = ds.pp.filter_by_obs("area", func=lambda x: (x > 20) & (x < 150))
   ds = ds.pp.grow_cells(iterations=2)

   # 4. quantification; adds `_intensity`
   ds = ds.pp.add_quantification(func="intensity_mean").pp.transform_expression_matrix(method="arcsinh")

   # 5. cell types; maps marker -> cell type
   ds = ds.la.predict_cell_types_argmax({"CD4": "T_h", "CD8": "T_tox", "FOXP3": "T_reg", "BCL6": "T_fh"})
   ds = ds.la.set_label_colors(["T_h", "T_tox", "T_reg", "T_fh"], ["red", "green", "yellow", "blue"])

   # 6. neighborhoods: cell type composition around each cell (adds `_neighborhoods` and `_adjacency_matrix`);
   #    to cluster these compositions into neighborhoods, use sp.ImageContainer (see below)
   ds = ds.nh.compute_neighborhoods_radius(radius=100)

   # 7. plotting (all plotting functions accept `ax=`)
   ds.pp[["CD4", "CD8"]].pl.colorize(["red", "green"]).pl.show(render_segmentation=True)
   ds.pl.show(render_image=False, render_labels=True)

   # 8. tables and export
   obs = ds.pp.get_layer_as_df("_obs")               # cell types decoded to names
   expr = ds.pp.get_layer_as_df("_intensity")
   ds.drop_encoding().to_zarr("sample.zarr")         # reload with xr.open_zarr("sample.zarr")
   adata = ds.tl.convert_to_anndata()                # requires anndata
   sdata = ds.tl.convert_to_spatialdata()            # requires spatialdata

Other frequently used operations:

* Segmentation from elsewhere: ``ds.pp.add_segmentation(mask)``; alternatives to cellpose: ``ds.tl.stardist()``, ``ds.tl.mesmer()``.
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

   sp.pp.threshold(sdata, quantile=[0.2, 0.6, 0.6, 0.6, 0.6])
   sp.tl.cellpose(sdata, channel="DAPI")                    # writes sdata.labels["segmentation"]
   sp.pp.add_quantification(sdata, func="intensity_mean")    # writes sdata.tables["table"]
   sp.pp.transform_expression_matrix(sdata, method="arcsinh")
   sp.la.predict_cell_types_argmax(sdata, {"CD4": "T_h", "CD8": "T_tox"})  # adds obs["celltype"]

Convert between the two backends with ``sp.read_from_spatialdata(sdata)`` and ``ds.tl.convert_to_spatialdata()``.
The xarray backend offers the full feature set; the spatialdata backend covers the core pipeline.

Common pitfalls
---------------

* The accessors only exist after ``import spatialproteomics``, including on objects loaded with ``xr.open_zarr``.
* Accessor methods return new objects. ``ds.pp.threshold(0.5)`` without reassignment has no effect.
* ``ds.pp[a:b, c:d]`` slices **x first, then y**, while the image array is stored as ``(channels, y, x)``.
  Slices select by coordinate value (like ``xarray.Dataset.sel``), so the end point is included.
* Call ``drop_encoding()`` before ``to_zarr()`` to avoid xarray encoding errors.
* Optional tools must be installed separately: ``cellpose``, ``stardist``, ``astir`` and ``spotiflow``
  (or all at once with ``pip install "spatialproteomics[all]"``); ``tl.mesmer`` additionally needs ``deepcell``.
* Many methods require earlier steps: e.g. ``la.predict_cell_types_argmax`` needs ``_intensity``, which needs ``_segmentation``.
  Inspect the object (``print(ds)``) to see which layers are present.
