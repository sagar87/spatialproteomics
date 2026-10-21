# Interoperability

This notebook shows some way that you can import and export data from `spatialproteomics`.
If you want to follow along with this tutorial, you can download the data [here](https://oc.embl.de/index.php/s/XzEa9po1tjiDzzJ).

```python
import spatialproteomics as sp
import pandas as pd
import xarray as xr
import os
import shutil
import anndata
```

## Exporting Data

Once you are happy with your analysis, you will likely want to export the results. The easiest way to do this is by using the `zarr` format, but `csv`, `anndata`, and `spatialdata` are also supported.

```python
# loading a test file which we will export later
# notice how easy it is to load the file from a zarr using xarray
ds = xr.open_zarr("../../data/LN_11_1.zarr")
```

## Exporting to Zarr
This is the easiest file format to work with. It allows you to store and load the xarray objects with a single line of code. It is highly recommended to call `drop_encoding()` before exporting to zarr. There are several open issues linked to encoding problems, and this is the easiest way to circumvent them. For more references, refer to these issues: [issue 1](https://github.com/pydata/xarray/issues/3476), [issue 2](https://github.com/pydata/xarray/issues/9037).

```python
zarr_path = "tmp.zarr"

# removing the zarr if it exists
if os.path.exists(zarr_path):
    shutil.rmtree(zarr_path)

# exporting as zarr
ds.drop_encoding().to_zarr("tmp.zarr")
```

Output:

```text
<xarray.backends.zarr.ZarrStore at 0x7f968ca71d40>
```

## Exporting Tables to CSV
Let's say you want to export some tables as csvs. This can be done with pandas.

```python
df = ds.pp.get_layer_as_df("_obs")
df.head()
```

Output:

```text
   BCL-2_binarized  BCL-6_binarized  CD163_binarized  CD45RA_binarized  \
1              0.0              0.0              0.0               0.0   
2              0.0              0.0              0.0               0.0   
3              0.0              0.0              0.0               0.0   
4              0.0              0.0              0.0               0.0   
5              0.0              0.0              0.0               1.0   

   CD45RO_binarized  CD8_binarized  FOXP3_binarized  Helios_binarized  \
1               0.0            0.0              0.0               0.0   
2               0.0            0.0              0.0               0.0   
3               0.0            0.0              0.0               0.0   
4               0.0            0.0              0.0               0.0   
5               0.0            0.0              0.0               0.0   

   ICOS_binarized  Ki-67_binarized  ...  TCF7-TCF1_binarized  Tim3_binarized  \
...
```

```python
# exporting as csv
df.to_csv("tmp.csv")
```

## Exporting to AnnData
AnnData is a format used by scanpy, which can be useful to create interesting plots and downstream analyses. For this reason, you can export the xarray object as an AnnData object. Note that this object will only store the tabular data, but not the image or the segmentation layer.

```python
# adding an additional quantification to demonstrate how those can be exported as well
ds = ds.pp.add_quantification(func=sp.percentage_positive, key_added="_percentage_positive")
```

```python
# putting the expression matrix into an anndata object
adata = ds.tl.convert_to_anndata(
    expression_matrix_key="_intensity",
    additional_layers={"percentage_positive": "_percentage_positive"},
    additional_uns={"label_colors": "_la_properties"},
)
adata
```

Output:

```text
AnnData object with n_obs × n_vars = 16871 × 56
    obs: 'BCL-2_binarized', 'BCL-6_binarized', 'CD163_binarized', 'CD45RA_binarized', 'CD45RO_binarized', 'CD8_binarized', 'FOXP3_binarized', 'Helios_binarized', 'ICOS_binarized', 'Ki-67_binarized', 'PD-1_binarized', 'TCF7-TCF1_binarized', 'Tim3_binarized', '_labels', '_neighborhoods', 'centroid-0', 'centroid-1', 'degree', 'diversity_index', 'homophily', 'inter_label_connectivity'
    uns: '_labels_colors', 'label_colors'
    obsm: 'spatial'
    layers: 'percentage_positive'
```

```python
# writing to disk as hdf5
adata.write("tmp.h5ad")
```

## Exporting to SpatialData
SpatialData is a data format which is commonly used for spatial omics analysis and combines the power of zarr with anndata. You can export to this data format as well.

```python
spatialdata_object = ds.tl.convert_to_spatialdata(expression_matrix_key="_intensity")
spatialdata_object
```

Output:

```text
[34mINFO    [0m Transposing `data` of type: [1m<[0m[1;95mclass[0m[39m [0m[32m'dask.array.core.Array'[0m[1m>[0m to [1m([0m[32m'c'[0m, [32m'y'[0m, [32m'x'[0m[1m)[0m.                           
[34mINFO    [0m Transposing `data` of type: [1m<[0m[1;95mclass[0m[39m [0m[32m'dask.array.core.Array'[0m[1m>[0m to [1m([0m[32m'y'[0m, [32m'x'[0m[1m)[0m.                                
SpatialData object
├── Images
│     └── 'image': DataArray[cyx] (5, 101, 101)
├── Labels
│     └── 'segmentation': DataArray[yx] (101, 101)
└── Tables
      └── 'table': AnnData (56, 5)
with coordinate systems:
    ▸ 'global', with elements:
        image (Images), segmentation (Labels)
```

```python
# storing as zarr file
spatialdata_object.write("tmp.zarr")
```

Output:

```text
[34mINFO    [0m The Zarr backing store has been changed from [3;35mNone[0m the new file path: tmp.zarr                             
```

## Importing from Spatialdata

In the example workflow, you have already seen how to read data from a tiff file. If you already have your data in `spatialdata` format, you can also read it in from there. Reading in the data like this will convert the data from `spatialdata` format to `xarray` format, so that you can use the `xarray` backend of `spatialproteomics`. The example data was obtained from [here](https://spatialdata.scverse.org/en/latest/api/datasets.html#spatialdata.datasets.raccoon).

```python
ds = sp.read_from_spatialdata("../../data/spatialdata_example.zarr", image_key="raccoon")
ds
```

Output:

```text
<xarray.Dataset> Size: 9MB
Dimensions:        (channels: 3, y: 768, x: 1024, cells: 70, features: 2)
Coordinates:
  * channels       (channels) int64 24B 0 1 2
  * y              (y) int64 6kB 0 1 2 3 4 5 6 7 ... 761 762 763 764 765 766 767
  * x              (x) int64 8kB 0 1 2 3 4 5 6 ... 1018 1019 1020 1021 1022 1023
  * cells          (cells) int64 560B 1 2 3 4 5 6 7 8 ... 64 65 66 67 68 69 70
  * features       (features) <U10 80B 'centroid-0' 'centroid-1'
Data variables:
    _image         (channels, y, x) uint8 2MB dask.array<chunksize=(3, 768, 1024), meta=np.ndarray>
    _segmentation  (y, x) int64 6MB 0 0 0 0 0 0 0 0 ... 69 69 69 69 69 69 69 69
    _obs           (cells, features) float64 1kB 44.79 402.5 ... 736.5 890.5
```

## Exporting to Seurat

`Seurat` is widely used in the R community to analyze spatial omics data. Here, we show how you can create a `Seurat` object from a `spatialproteomics` one. It is most sensible to store the data frame to disk first, and then read it in with R. For demonstration purposes, we run R directly in this notebook.

```python
# setup and data loading
import os
import sys

# setup for rpy2 to use the correct R binary
env_r_bin = "/g/huber/users/meyerben/.conda/envs/tmp_env_3/bin"
os.environ["PATH"] = env_r_bin + os.pathsep + os.environ.get("PATH", "")
if "R_HOME" in os.environ:
    del os.environ["R_HOME"]


import xarray as xr
import spatialproteomics as sp

celltype_colors = {
    "B cell": "#5799d1",
    "T cell": "#ebc850",
    "Myeloid cell": "#de6866",
    "Dendritic cell": "#4cbcbd",
    "Macrophage": "#bb7cb4",
    "Stromal cell": "#62b346",
    "Endothelial cell": "#bf997d",
}

# loading in a data set
ds = xr.open_zarr("../../data/LN_24_1.zarr")
# for clearer visualizations, we set the cell types to the broadest level (B cells, T cells, ...)
ds = ds.la.set_label_level("labels_0", ignore_neighborhoods=True).la.set_label_colors(
    celltype_colors.keys(), celltype_colors.values()
)
```

```python
# this gives us a data frame of cells by markers
intensity_df = ds.pp.get_layer_as_df("_intensity")

# getting the spatial information and cell types from the spatialproteomics object
spatial_df = ds.pp.get_layer_as_df()[["centroid-0", "centroid-1", "_labels"]]
# renaming the columns for better compatibility with R
spatial_df.columns = ["centroid.0", "centroid.1", "labels"]
```

```python
# send the data to R, so that we can use it directly
```

```python
library(Seurat)
library(dplyr)

# intensity_df: each row = cell, each col = marker
# transpose so markers become "genes"
mat <- t(as.matrix(intensity_df))

# Make sure row/col names align
colnames(mat) <- rownames(intensity_df)

# Create Seurat object
seu <- CreateSeuratObject(counts = mat)

# Add metadata (spatial info + cell labels)
seu <- AddMetaData(seu, metadata = spatial_df)
```

Output:

```text
    WARNING: The R package "reticulate" only fixed recently
    an issue that caused a segfault when used with rpy2:
    https://github.com/rstudio/reticulate/pull/1188
    Make sure that you use a version of that package that includes
    the fix.
    Loading required package: SeuratObject
Loading required package: sp
‘SeuratObject’ was built with package ‘Matrix’ 1.7.3 but the current
version is 1.7.4; it is recomended that you reinstall ‘SeuratObject’ as
the ABI for ‘Matrix’ may have changed

Attaching package: ‘SeuratObject’

The following objects are masked from ‘package:base’:

...
```

```python
# Make sure the rownames exactly match the Seurat cells
spatial_coords <- spatial_df[, c("centroid.0", "centroid.1")]

# The cell names in the Seurat object
cell_names <- Cells(seu)

# Assign rownames
rownames(spatial_coords) <- cell_names
colnames(spatial_coords) <- c(1, 2)

# Convert to matrix
embeddings_mat <- as.matrix(spatial_coords)

# Now create the spatial reduction
seu[["spatial"]] <- CreateDimReducObject(
  embeddings = embeddings_mat,
  key = "spatial_",
  assay = DefaultAssay(seu)
)
```

```python
library(ggplot2)

p <- DimPlot(
  seu,
  reduction = "spatial",
  group.by = "labels",
  pt.size = 1
) + coord_fixed()  # ensures x and y axes have equal scale

print(p)
```

Output:

```text
Learn more about the underlying theory at https://ggplot2-book.org/
```
