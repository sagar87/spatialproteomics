# Interactive Removal of Artifacts

Sometimes, it can happen that only a part of the tissue is usable for analysis, for example due to artifacts or poor tissue quality in a certain region. In such cases, it can make sense to mask out problematic regions. This way, the image can still be used for downstream analysis, but the artifact will not influence results in a negative way. `Spatialproteomics` allows users to upload a binary mask, which indicates which part of the image is suitable for further analysis and which part is not.

The example below shows how one could for example use [TissueTag](https://github.com/Teichlab/TissueTag/tree/main) to obtain such a mask, load it into the spatialproteomics and use the mask to remove the artifact. Note that TissueTag runs in jupyter notebooks or jupyter lab, but has no implementation for jupyterhub at the moment.

A link to this exact sample can be found in the spatialproteomics manuscript, if you want to follow along with a different sample you can download the data [here](https://oc.embl.de/index.php/s/XzEa9po1tjiDzzJ).

```python
import xarray as xr
import spatialproteomics as sp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
```

### Loading the image from a zarr file

```python
ds = xr.open_zarr("4_B4.zarr")

# quick plotting of some channels
plt.figure(figsize=(10, 10))
ds.pp[["PAX5", "CD3", "Podoplanin"]].pl.autocrop().pl.show()
```

Output:

```text
<xarray.Dataset>
Dimensions:                (cells: 8395, celltype_levels: 3, channels: 3,
                            y: 1811, x: 1721, features: 9, labels: 9, props: 2,
                            rgba: 4)
Coordinates:
  * cells                  (cells) int64 1 2 3 4 5 ... 8391 8392 8393 8394 8395
  * celltype_levels        (celltype_levels) <U8 'labels' 'labels_1' 'labels_2'
  * channels               (channels) <U11 'PAX5' 'CD3' 'Podoplanin'
  * features               (features) <U10 'BCL-2' 'CCR7' ... 'ki-67'
  * labels                 (labels) int64 1 2 3 4 5 6 7 8 9
  * props                  (props) <U6 '_color' '_name'
  * x                      (x) int64 70 71 72 73 74 ... 1786 1787 1788 1789 1790
  * y                      (y) int64 120 121 122 123 124 ... 1927 1928 1929 1930
  * rgba                   (rgba) <U1 'r' 'g' 'b' 'a'
Data variables:
...
```

### Drawing in the healthy regions using TissueTag

```python
# first, we need to get our image as an RGBA image

# here, we extract the Podoplanin channel and assign it to the red and green channel, while keeping the blue channel at 0 and setting alpha to 1 everywhere
podoplanin_img = ds.pp["Podoplanin"]["_image"].values[0]
# Create the RGBA image by stacking channels (R, G, B, A)
rgba_image = np.stack(
    (podoplanin_img, podoplanin_img, np.zeros_like(podoplanin_img), np.ones_like(podoplanin_img) * 255), axis=-1
)

plt.figure(figsize=(10, 10))
plt.imshow(rgba_image)
```

```python
# next, we set up the annotation layer, which will ultimately become the binary mask that tells us where the tissue is
annodict = {"Artifact": "yellow"}
labels = np.zeros((rgba_image.shape[0], rgba_image.shape[1]), dtype=np.uint8)
```

```python
# setup for TissueTag
import panel as pn
import socket
import os
import tissue_tag as tt

os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"
host = "8888"
```

```python
# use annotator to label tissue regions according to categories indicated above
annotator, render_dict = tt.annotator(rgba_image, labels, anno_dict=annodict, use_datashader=True)
pn.io.notebook.show_server(annotator, notebook_url=f"{socket.gethostname()}:{host}")
```

Output:

```text
<panel.io.server.Server at 0x16a01c0a0>
```

```python
labels = tt.update_annotator(imarray=rgba_image, labels=labels, anno_dict=annodict, render_dict=render_dict)
```

Output:

```text
Artifact
```

```python
# inverting the labels so that 0 becomes 1 and 1 becomes 0
# the region where the artifact is should be 0, everything else should be 1
labels = 1 - labels
```

### Putting the mask into the spatialproteomics object

```python
# by default, this method adds a layer with the key '_mask'
ds = ds.pp.add_layer(labels)
```

```python
# plotting the mask (by using the render_segmentation method, since the data structure is conceptually similar to the segmentation)
_ = ds.pl.render_segmentation(layer_key="_mask", alpha=0.7).pl.imshow()
```

### Using the mask to set every pixel intensity to 0 in that region

```python
ds_masked_intensities = ds.pp.mask_region()
_ = ds_masked_intensities.pp[["PAX5", "CD3", "Podoplanin"]].pl.show()
```

You can see that this set all of the marker intensities at 0 in the problematic region. However, if we plot the segmentation masks on top, you can see that we have not removed those.

```python
_ = ds_masked_intensities.pp[["PAX5", "CD3", "Podoplanin"]].pl.show(render_segmentation=True)
```

Most of the time, what we want to do instead is to remove all cells that were segmented in the area. This way we still stick to the original data, but remove the problematic region from analysis.

### Using the mask to remove cells in that region

```python
ds_masked_cells = ds.pp.mask_cells()
_ = ds_masked_cells.pp[["PAX5", "CD3", "Podoplanin"]].pl.show(render_segmentation=True)
```

You can of course also combine the two.

```python
_ = ds.pp.mask_region().pp.mask_cells().pp[["PAX5", "CD3", "Podoplanin"]].pl.show(render_segmentation=True)
```
