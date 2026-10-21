# Interactivity with napari

*Spatialproteomics* has a variety of functions to ensure interoperability with existing tools. For example, we can export the *spatialproteomics* object into *spatialdata*, and the use *napari* for interactive visualizations.

If you want to follow along with this tutorial, you can download the data [here](https://oc.embl.de/index.php/s/XzEa9po1tjiDzzJ).

```python
import spatialproteomics as sp
import xarray as xr
from napari_spatialdata import Interactive
import numpy as np
```

```python
# loading in a data set and performing some formatting for convenience
ds = xr.open_zarr("../../data/LN_11_1.zarr")
```

```python
# turning the spatialproteomics object into a spatialdata object
sd_obj = ds.tl.convert_to_spatialdata()
sd_obj
```

Output:

```text
[34mINFO    [0m Transposing `data` of type: [1m<[0m[1;95mclass[0m[39m [0m[32m'dask.array.core.Array'[0m[1m>[0m to [1m([0m[32m'c'[0m, [32m'y'[0m, [32m'x'[0m[1m)[0m.                           
[34mINFO    [0m Transposing `data` of type: [1m<[0m[1;95mclass[0m[39m [0m[32m'dask.array.core.Array'[0m[1m>[0m to [1m([0m[32m'y'[0m, [32m'x'[0m[1m)[0m.                                
SpatialData object
├── Images
│     └── 'image': DataArray[cyx] (56, 3000, 3000)
├── Labels
│     └── 'segmentation': DataArray[yx] (3000, 3000)
└── Tables
      └── 'table': AnnData (16871, 56)
with coordinate systems:
    ▸ 'global', with elements:
        image (Images), segmentation (Labels)
```

```python
interactive = Interactive(sd_obj)
interactive.run()
```

Running this code in a python script will open `napari`, which you can then use to interactively look at your data. You can look at different markers, visualize cell types, neighborhoods, and many more. Below are some examples of what you can visualize with `napari`.

#### Markers
![Visualization of a marker with napari](../_static/img/napari_marker.png)

#### Segmentation
![Visualization of a segmentation with napari](../_static/img/napari_segmentation.png)

#### Cell Type Labels
![Visualization of a cell type labels with napari](../_static/img/napari_labels.png)

#### Binarization
![Visualization of a marker binarization with napari](../_static/img/napari_binarization.png)

#### Neighborhoods
![Visualization of neighborhoods with napari](../_static/img/napari_neighborhoods.png)

#### Features
![Visualization of homophily with napari](../_static/img/napari_homophily.png)
