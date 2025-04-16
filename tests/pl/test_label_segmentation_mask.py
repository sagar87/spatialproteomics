import numpy as np
import pytest

from spatialproteomics.pl.utils import _label_segmentation_mask


@pytest.mark.parametrize(
    "relabel_dict, expected_values",
    [
        ({1: [1, 2, 3], 2: [5, 7], 3: [8, 9]}, {1: [1, 2, 3], 2: [5, 7], 3: [8, 9], 0: []}),
        ({1: [1, 2, 3], 2: [5], 3: [8]}, {1: [1, 2, 3], 2: [5], 3: [8], 0: [7, 9]}),
        ({1: [1, 2, 3]}, {1: [1, 2, 3], 0: [5, 7, 8, 9]}),
    ],
)
def test_label_segmentation_mask(relabel_dict, expected_values):
    seg_mask = np.zeros((10, 10), dtype=int)

    seg_mask[0:2, 0:2] = 1  # Object 1
    seg_mask[4:6, 2:4] = 2  # Object 2
    seg_mask[1:3, 5:7] = 3  # Object 3
    seg_mask[7:9, 1:3] = 5  # Object 5
    seg_mask[8:10, 3:5] = 7  # Object 7
    seg_mask[5:8, 5:8] = 8  # Object 8
    seg_mask[2, 9] = 9  # Object 9

    res = _label_segmentation_mask(seg_mask, relabel_dict)

    for new_label, old_labels in expected_values.items():
        for old in old_labels:
            assert np.all(res[seg_mask == old] == new_label)
