import xarray as xr
import pandas as pd
from ..constants import Layers, Dims


@xr.register_dataset_accessor("ext")
class ExternalAccessor:
    """The external accessor enables the application of external tools such as StarDist or Astir"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def astir(
        self,
        marker_dict: dict,
        key: str = Layers.INTENSITY,
        threshold: float = 0,
        seed: int = 42,
        learning_rate: float = 0.001,
        batch_size: float = 64,
        n_init: int = 5,
        n_init_epochs: int = 5,
        max_epochs: int = 500,
        cell_id_col: str = "cell_id",
        cell_type_col: str = "cell_type",
        **kwargs,
    ):
        """
        This method predicts cell types from an expression matrix using the Astir algorithm.

        Args:
            marker_dict (dict): dictionary mapping markers to cell types. Can also include cell states. Example: {"cell_type": {'B': ['PAX5'], 'T': ['CD3'], 'Myeloid': ['CD11b']}}
            key (str, optional): Layer to use as expression matrix.
            threshold (float, optional): Certainty threshold for astir to assign a cell type. Defaults to 0.
            seed (int, optional): Defaults to 42.
            learning_rate (float, optional): Defaults to 0.001.
            batch_size (float, optional): Defaults to 64.
            n_init (int, optional): Defaults to 5.
            n_init_epochs (int, optional): Defaults to 5.
            max_epochs (int, optional):  Defaults to 500.
            cell_id_col (str, optional): Defaults to "cell_id".
            cell_type_col (str, optional): Defaults to "cell_type".

        Raises:
            ValueError: if no expression matrix was present or the image is not of type uint8.

        Returns:
            DataArray: a DataArray with the assigned cell types.
        """
        import astir
        import torch

        # check if there is an expression matrix
        if key not in self._obj:
            raise ValueError(
                f"No expression matrix with key {key} found in the object. Make sure to call pp.quantify first."
            )

        # raise an error if the image is of anything other than uint8
        if self._obj[Layers.IMAGE].dtype != "uint8":
            raise ValueError(
                "The image is not of type uint8, which is required for astir to work properly. Use the dtype argument in add_quantification() to convert the image to uint8."
            )

        # converting the xarray to a pandas dataframe to keep track of channel names and indicies after running astir
        expression_df = pd.DataFrame(self._obj[key].values, columns=self._obj.coords[Dims.CHANNELS].values)
        expression_df.index = self._obj.coords[Dims.CELLS].values

        # running astir
        model = astir.Astir(expression_df, marker_dict, dtype=torch.float64, random_seed=seed)
        model.fit_type(
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_init=n_init,
            n_init_epochs=n_init_epochs,
            max_epochs=max_epochs,
            **kwargs,
        )

        # getting the predictions
        assigned_cell_types = model.get_celltypes(threshold=threshold)
        # assign the index to its own column (called cell)
        assigned_cell_types = assigned_cell_types.reset_index()
        # renaming the columns
        assigned_cell_types.columns = [cell_id_col, cell_type_col]
        # setting the cell dtype to int
        assigned_cell_types[cell_id_col] = assigned_cell_types[cell_id_col].astype(int)

        # adding the labels to the obs slot
        return self._obj.pp.add_labels(assigned_cell_types, cell_col=cell_id_col, label_col=cell_type_col)
