# Spatial-data

[![PyPI version](https://badge.fury.io/py/spat-data.svg)](https://badge.fury.io/py/spat-data)

Spatial data is a light weight wrapper around `xarray` with the intention to facilitate the processing, exploration and analysis of highly multiplexed immunohistochemistry data.

<p align="center" width="100%">
    <img width="50%" src="docs/preview2.png">
</p>

## Principles

Multiplexed imaging data comprises at least 3 dimensions (i.e. `channels`, `x`, and `y`) and has often additional data such as segmentation masks or cell type annotations associated with it. In `spatial-data`, we use `xarray` to create a data structure that keeps all of these data dimension in sync. This data structure can then be used to apply all sorts of operations to the data. Users can segment cells, perform different image processing steps, quantify protein expression, predict cell types, and perform a variety of plotting. By providing researchers with those tools, `spatial-data` can be used to quickly explort highly multiplexed spatial proteomics data directly within jupyter notebooks.


## Installation

To install `spatial-data` first create a python environment and install the package using 

```
pip install spat-data
```

Check the documentation for further information https://sagar87.github.io/spatial-data/.
