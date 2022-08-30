# Spatial-data

Spatial data is a light weight wrapper around `xarray` with the intention to factilitate the data exploration and analysis of highly multiplexed immunohistochemistry data.

## Principles

Multiplexed imaging data comprises at least 3 dimensions (i.e. `channels`, `x`, and `y`) and has often additional data associated such as a segmentation and cell type annotatons which implictly introduce new dimensions such as `cells` and `labels`. In `spatial-data`, we use `xarray` to create a data structure that keeps all of these data dimension in sync. This enables us to subset the data efficiently and perform reproducible exploratory data analyses in Juyter Notebooks as `spatial-data` comes with exteneded plotting functions.


## Installation

To install `spatial-data` first create `Python 3.8` environment and install the package using 

```
pip install spat-data
```






