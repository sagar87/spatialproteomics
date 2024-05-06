import numpy as np

from spatial_data.pp.utils import handle_disconnected_cells


def test_handle_disconnected_cells():
    # normal cases, should not throw an error
    arr = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]])
    handle_disconnected_cells(arr)


def test_handle_disconnected_cells_remove():
    arr = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]])
    arr_processed = handle_disconnected_cells(arr, mode="remove")
    arr_with_disconnect_removed = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]])

    # checking that the two arrays are the same
    assert np.all(arr_processed == arr_with_disconnect_removed)


def test_handle_disconnected_cells_keep_largest():
    arr = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]])

    arr_processed = handle_disconnected_cells(arr, mode="keep_largest")

    arr_target = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]])

    assert np.all(arr_processed == arr_target)

    # also testing that it works if we have more than two components
    arr = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 1], [0, 0, 0, 1, 1]])

    arr_target = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 2, 1], [0, 0, 0, 1, 1]])

    arr_processed = handle_disconnected_cells(arr, mode="keep_largest")

    assert np.all(arr_processed == arr_target)

    # testing that it works if the two largest components have the same size (in this case one should be chosen arbitrarily)
    arr = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 1], [0, 1, 0, 1, 1]])

    arr_processed = handle_disconnected_cells(arr, mode="keep_largest")

    arr_target = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 1, 0, 0, 0]])

    assert np.all(arr_processed == arr_target)


def test_handle_disconnected_cells_relabel():
    arr = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]])

    arr_processed = handle_disconnected_cells(arr, mode="relabel")

    arr_target = np.array([[0, 0, 0, 1, 0], [0, 3, 0, 2, 0], [0, 3, 0, 2, 0], [0, 0, 0, 0, 0]])

    assert np.all(arr_processed == arr_target), f"Processed array: {arr_processed}"
