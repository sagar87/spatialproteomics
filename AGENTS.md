# AGENTS.md

Guidance for AI coding agents (and humans) working on the spatialproteomics code base.
For how to *use* the library, read `docs/concepts.rst` first; the published docs also serve
`llms.txt` and `llms-full.txt` at https://sagar87.github.io/spatialproteomics/.

## Setup

```bash
uv venv --python=python3.12      # python >= 3.11 for development (zarr 3)
source .venv/bin/activate
uv pip install -e ".[dev,all]"   # add ",docs" to build the documentation
```

`uv sync --extra dev --extra docs` installs the exact versions from `uv.lock`.

## Commands

| Task | Command |
|------|---------|
| Run all tests | `make test` (`uv run python -m pytest tests -x`) |
| Run one test file | `uv run python -m pytest tests/pp/test_threshold.py` |
| Format | `make format` (isort + black, line length 120) |
| Lint | `make lint` (flake8) |
| Everything | `make all` |
| Build docs | `cd docs && uv run sphinx-build -b html . _build` (needs `pandoc`) |

CI (`.github/workflows/branch.yaml`) runs pytest, `isort --check-only`, `black --check`, flake8 and bandit
on `spatialproteomics/`, and builds the docs. Run `make all` before pushing.

## Layout

```
spatialproteomics/
  __init__.py          public API (keep __all__ in sync; tests/test_public_api.py checks it)
  constants.py         Layers, Dims, Features, Props, SDLayers, ... (string keys of the data model)
  container.py         load_image_data(), read_from_spatialdata()
  pp/preprocessing.py  ds.pp accessor + sp.pp.* spatialdata functions
  la/label.py          ds.la accessor + sp.la.* functions
  nh/neighborhood.py   ds.nh accessor
  pl/plot.py           ds.pl accessor
  tl/tool.py           ds.tl accessor + sp.tl.* functions (cellpose, stardist, astir, spotiflow)
  sd/utils.py          conversion helpers for the spatialdata backend
  image_container/     ImageContainer (multi-sample neighborhoods)
  */utils.py           private helpers used by the accessors
tests/                 mirrors the package layout; fixtures in tests/conftest.py load tests/test_files/*.zarr
docs/                  Sphinx: *.rst API pages (automodule), notebooks/*.ipynb tutorials, concepts.rst
docs/_ext/llms_txt.py  Sphinx extension that writes llms.txt, llms-full.txt and per-page Markdown
```

## Conventions

- **Two backends.** Accessor methods (`ds.pp.foo(...)`) operate on `xarray.Dataset` objects and must return a new
  dataset without mutating the input. Module-level functions (`sp.pp.foo(sdata, ...)`) operate on
  `spatialdata.SpatialData` objects in place and accept `copy=False`. Put the actual computation into a private
  helper in the subpackage's `utils.py` and call it from both backends; `sd/utils.py` extracts images,
  segmentations and tables from spatialdata objects.
- **Use the constants** from `constants.py` (`Layers.SEGMENTATION`, `Dims.CELLS`, `Features.LABELS`, ...) instead of
  string literals.
- **Keep dimensions in sync.** Any operation that removes cells must update `_segmentation`, `_obs`, `_intensity`
  and all other layers with a `cells` dimension (see `pp.get_bbox` and `la.__getitem__` for examples).
- **Optional dependencies** (cellpose, stardist, deepcell, astir, spotiflow, spatialdata, anndata) are imported
  inside the functions that need them, never at module level.
- **Docstrings** are the API documentation (rendered by autodoc and by `docs/_ext/llms_txt.py`). Every public method
  needs a docstring with parameters and return value; numpydoc style is the most common in the code base.
- New public functions must be exported in the subpackage `__init__.py` and, if top-level, in `spatialproteomics/__init__.py`.
- Add a test under the matching `tests/<subpackage>/` directory for every new feature or bug fix.

## Documentation

- New tutorials go into `docs/notebooks/` and the toctree in `docs/index.rst`; add a one-line description to
  `llms_txt_descriptions` in `docs/conf.py` so the page is described in `llms.txt`.
- Notebooks are committed with outputs because the docs build does not execute them. Keep outputs small
  (downsample images before plotting) and do not commit notebooks just because they were re-run.
- When the API changes, update `docs/concepts.rst` if the cheatsheet or the workflow is affected.
