# import numpy as np

# from spatial_data.constants import Layers


def test_quantification(dataset):
    pass
    # cell_idx = np.unique(dataset[Layers.SEGMENTATION])
    # test_cell = np.random.choice(cell_idx[cell_idx > 0])

    # x = (
    #     dataset[Layers.IMAGE]
    #     .loc["Hoechst"]
    #     .values[dataset[Layers.SEGMENTATION] == test_cell]
    #     .sum()
    # )
    # y = dataset[Layers.DATA].loc[test_cell, "Hoechst"].values

    # assert x == y
