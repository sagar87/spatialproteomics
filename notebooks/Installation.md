# Installation

Install `spatialproteomics` with 
```
pip install spatialproteomics
```

## Extras
Some of the functionalities depend on external tools (e. g. `stardist`, `cellpose`,  `spatialdata`). Since having lots of external tools also leads to lots of dependencies, installing all of them into one environment is prone to cause dependency conflicts. 

Depending on your needs, you can choose which tools you want to install together with the `spatialproteomics` core like this:
```
pip install "spatialproteomics[stardist]"
pip install "spatialproteomics[cellpose]"
pip install "spatialproteomics[spatialdata]"
pip install "spatialproteomics[astir]"
pip install "spatialproteomics[spotiflow]"
pip install "spatialproteomics[all]"
```

The latter one will install all external packages automatically.

You can also combine dependencies like this:
```
pip install "spatialproteomics[cellpose,spatialdata]"
```

## Deprecated Packages
`Mesmer` only supported `python<=3.10`, which has reached its end of life in October 2026. It has hence been removed from `spatialproteomics`. If you want to run segmentation using `mesmer`, please downgrade to `python==3.10` and `spatialproteomics==0.8.6`.
