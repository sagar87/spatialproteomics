# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "spatialproteomics"
copyright = "2024, Matthias Meyer-Bender, Harald Vohringer"
author = "Matthias Meyer-Bender, Harald Vohringer"

# Dynamically set the release version for multi-version builds
release = os.getenv("SPHINX_MULTIVERSION_NAME", "0.7.3")

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_multiversion",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_build/*", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = project
html_logo = "_static/img/spatialproteomics_logo_light.png"
html_favicon = "_static/img/spatialproteomics_icon.png"
html_theme_options = {
    "logo_only": True,
    "home_page_in_toc": False,
    "navigation_with_keys": True,
    "logo_light": "_static/img/spatialproteomics_logo_light.png",
    "logo_dark": "_static/img/spatialproteomics_logo_dark.png",
}
pygments_style = "default"

# -- sphinx-multiversion configuration ---------------------------------------

smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"  # Semantic versioning tags (e.g., v1.0.0)
smv_branch_whitelist = r"^main$"  # Include only the main branch
smv_released_pattern = r"^refs/tags/v.*$"
smv_remote_whitelist = r"origin"
smv_latest_version = "latest"
