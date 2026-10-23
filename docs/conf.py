# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

# -- Project information -----------------------------------------------------

project = "spatialproteomics"
copyright = "2024, Matthias Meyer-Bender, Harald Vohringer"
author = "Matthias Meyer-Bender, Harald Vohringer"

# Dynamically set the release version for multi-version builds
release = os.getenv("SPHINX_MULTIVERSION_NAME", "0.8.6")

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_multiversion",
    "IPython.sphinxext.ipython_console_highlighting",
    "llms_txt",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_build/*", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = project
html_favicon = "_static/img/spatialproteomics_icon.png"
html_theme_options = {
    "logo_only": True,
    "home_page_in_toc": False,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/img/spatialproteomics_logo_light.png",
        "image_dark": "_static/img/spatialproteomics_logo_dark.png",
    },
}
pygments_style = "default"

# -- llms.txt (see _ext/llms_txt.py) -----------------------------------------

llms_txt_base_url = "https://sagar87.github.io/spatialproteomics/"
llms_txt_summary = (
    "spatialproteomics is a Python toolbox for analyzing highly multiplexed fluorescence images "
    "(e.g. CODEX, MIBI, IMC): image processing, cell segmentation, marker quantification, "
    "cell type prediction, neighborhood analysis and plotting. It stores each sample as an xarray.Dataset "
    "whose layers (image, segmentation, expression matrix, labels) stay synchronized when subsetting."
)
llms_txt_details = """
Key facts for using the library:

- Install with `pip install spatialproteomics`; optional extras: `[cellpose]`, `[stardist]`, `[astir]`, `[spotiflow]`, `[spatialdata]`, `[all]`.
- `import spatialproteomics as sp` registers five accessors on every `xarray.Dataset`: `ds.pp` (preprocessing, subsetting, quantification), `ds.tl` (segmentation and other external tools, export), `ds.la` (cell type labels), `ds.nh` (neighborhoods) and `ds.pl` (plotting).
- Accessor methods return a new dataset and can be chained: `ds.pp.threshold(0.5).tl.cellpose(channel="DAPI").pp.add_quantification()`.
- Create an object with `sp.load_image_data(image, channel_coords=[...])`, store it with `ds.drop_encoding().to_zarr(path)` and load it with `xr.open_zarr(path)`.
- A second, scverse-style API works on `spatialdata.SpatialData` objects in place: `sp.pp.add_quantification(sdata)`, `sp.tl.cellpose(sdata, channel="DAPI")`.
- Start with the "Core concepts and cheatsheet" page below; it covers the data model, indexing and a complete workflow.
"""
llms_txt_descriptions = {
    "concepts": "data model, accessors, indexing, a complete example workflow and common pitfalls (read this first)",
    "notebooks/Installation": "pip install and optional extras",
    "notebooks/ExampleWorkflow": "end-to-end example: load, threshold, segment with cellpose, quantify, predict cell types, plot",
    "notebooks/ExampleWorkflowSpatialdata": "the same workflow with the in-place spatialdata backend",
    "notebooks/Plotting": "pl.show, colorize, render_* + imshow, scatter plots, annotations",
    "notebooks/Segmentation": "cellpose and stardist segmentation, filtering and merging masks",
    "notebooks/ImageProcessing": "thresholding, merging channels and custom image filters",
    "notebooks/CellTypePrediction": "argmax prediction, marker binarization, subtypes and astir",
    "notebooks/Neighborhoods": "neighborhood computation (radius, knn, delaunay), ImageContainer, graph features",
    "notebooks/Interoperability": "export to zarr, csv, AnnData and spatialdata, import from spatialdata",
    "notebooks/DownstreamAnalysis": "neighborhood enrichment with squidpy and spatial statistics in R (spatstat)",
    "notebooks/Interactivity": "interactive viewing with napari",
    "notebooks/Customizability": "plugging in custom segmentation, image processing, quantification, cell type prediction and features; spot detection with spotiflow",
    "notebooks/Slicing": "subsetting by channels, coordinates, cell types and neighborhoods",
    "notebooks/ArtifactRemoval": "annotating artifact regions interactively and masking them out",
    "notebooks/MultiTechnologyIntegration": "comparing samples from different imaging technologies; reading OME-TIFF files",
    "notebooks/FAQ": "memory usage, parallelization with dask, xarray vs. spatialdata backend",
    "preprocessing": "API of ds.pp and the sp.pp functions",
    "label": "API of ds.la and the sp.la functions",
    "neighborhood": "API of ds.nh",
    "plot": "API of ds.pl",
    "tool": "API of ds.tl and the sp.tl functions",
    "image_container": "API of sp.ImageContainer for multi-sample neighborhood analysis",
}

# -- sphinx-multiversion configuration ---------------------------------------

smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"  # Semantic versioning tags (e.g., v1.0.0)
smv_branch_whitelist = r"^main$"  # Include only the main branch
smv_released_pattern = r"^refs/tags/v.*$"
smv_remote_whitelist = r"origin"
smv_latest_version = "latest"
