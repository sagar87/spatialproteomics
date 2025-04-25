# spatialproteomics

[![PyPI version](https://badge.fury.io/py/spatialproteomics.svg)](https://badge.fury.io/py/spatialproteomics)

`Spatialproteomics` is an interoperable toolbox for analyzing highly multiplexed fluorescence image data. This analysis involves a sequence of steps, including segmentation, image processing, marker quantification, cell type classification, and neighborhood analysis. 

<p align="center" width="100%">
    <img src="docs/_static/img/figure_1.png" alt="Spatialproteomics orchestrates analysis workflows for highly multiplexed fluorescence images." style="width:70%;">
</p>

## Principles

Multiplexed imaging data comprises at least 3 dimensions (i.e. `channels`, `x`, and `y`) and has often additional data such as segmentation masks or cell type annotations associated with it. In `spatialproteomics`, we use `xarray` to create a data structure that keeps all of these data dimension in sync. This data structure can then be used to apply all sorts of operations to the data. Users can segment cells, perform different image processing steps, quantify protein expression, predict cell types, and plot their data in various ways. By providing researchers with those tools, `spatialproteomics` can be used to quickly explore highly multiplexed spatial proteomics data directly within jupyter notebooks.

<div style="text-align: center;">
    <img src="docs/_static/img/supplementary_figure_1.png" alt="The spatialproteomics data structure enables synchronized subsetting across shared dimensions." style="width:70%;"/>
</div>

## Getting Started

Please refer to the [documentation](https://sagar87.github.io/spatialproteomics). For a more interactive learning experience, you can also check out [this workshop](https://github.com/MeyerBender/spatialproteomics_workshop).

## Installation

To install `spatialproteomics` first create a python environment and install the package using 

```
pip install spatialproteomics
```

## Citation
> t.b.a.
