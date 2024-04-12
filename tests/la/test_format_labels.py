import numpy as np

from spatial_data.la.label import _format_labels


def test_format_labels_no_reformatting_necessary():
    lab = np.array(
        [
            1,
            2,
            3,
            4,
            4,
        ]
    )

    res = _format_labels(lab)
    assert np.all(res == lab)


def test_format_labels_simple():
    lab = np.array(
        [
            1,
            2,
            2,
            4,
            4,
        ]
    )

    res = _format_labels(lab)
    assert np.all(res == np.array([1, 2, 2, 3, 3]))


def test_format_labels_scrambled_reformatting():
    lab = np.array(
        [
            1,
            2,
            0,
            3,
            3,
        ]
    )

    res = _format_labels(lab)
    assert np.all(res == np.array([2, 3, 1, 4, 4]))


def test_format_labels_reformatting_with_zero():
    lab = np.array(
        [
            0,
            0,
            2,
            4,
            4,
        ]
    )

    res = _format_labels(lab)
    assert np.all(res == np.array([1, 1, 2, 3, 3]))
