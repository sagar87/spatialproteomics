import numpy as np
from skimage.segmentation import relabel_sequential

from ..base_logger import logger


def _format_labels(labels):
    """
    Format the labels array to ensure consecutive numbering.

    Parameters
    ----------
    labels : numpy.ndarray
        The input array of labels.

    Returns
    -------
    numpy.ndarray
        The formatted array of labels with consecutive numbering.

    Notes
    -----
    This function checks if the input labels array contains non-consecutive numbers. If it does, it relabels the array
    to ensure consecutive numbering. A warning message is logged if relabeling is performed.
    """

    formatted_labels = labels.copy()
    unique_labels = np.unique(labels)

    if ~np.all(np.diff(unique_labels) == 1):
        logger.warning("Labels are non-consecutive. Relabeling...")
        formatted_labels, _, _ = relabel_sequential(formatted_labels)

    return formatted_labels


def _get_markers_from_subtype_dict(subtype_dict):
    markers = []

    def extract_markers(subtypes):
        for subtype in subtypes:
            # Add markers from the current subtype
            markers.extend(subtype["markers"])
            # Recursively extract markers from nested subtypes
            if "subtypes" in subtype:
                extract_markers(subtype["subtypes"])

    for cell_type, details in subtype_dict.items():
        if "subtypes" in details:
            extract_markers(details["subtypes"])

    return markers


def _predict_cell_subtypes(df, subtype_dict):
    """
    Predicts cell subtypes based on a hierarchical dictionary of cell types and their markers.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cell data with binarized marker columns.
    subtype_dict : dict
        Dictionary defining the hierarchy of cell types and their markers.
        Each key is a cell type, and its value is a dictionary with keys:
        - 'markers': List of markers associated with the cell type.
        - 'subtypes': List of dictionaries defining subtypes, each with 'name' and 'markers'.
    Returns
    -------
    pandas.DataFrame
        DataFrame with new columns for each level of cell type annotations,
        named '_labels_0', '_labels_1', etc.
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()

    # Level-wise breadth-first search to iterate through the hierarchy
    level = 0
    queue = [(None, subtype_dict)]  # Queue holds (parent_type, current_dict)

    while queue:
        next_queue = []
        for parent_type, current_dict in queue:
            for cell_type, cell_info in current_dict.items():
                # Get subtypes if present
                subtypes = cell_info.get("subtypes", [])

                # If subtypes exist, add each to the queue
                for subtype in subtypes:
                    next_queue.append((cell_type, {subtype["name"]: subtype}))

                # Get markers for this cell type
                markers = cell_info.get("markers", [])

                # Check which cells match the markers and parent type
                for marker in markers:
                    if parent_type is None:
                        # Root level, assign to cells based on marker positivity
                        condition = df[f"{marker}_binarized"] == 1
                    else:
                        # Must match both parent type and marker
                        # Make this dependent on the previous level
                        previous_label_column = f"labels_{level - 1}" if level > 0 else "_labels"
                        condition = (df[f"{marker}_binarized"] == 1) & (df[previous_label_column] == parent_type)

                    # Update labels for current level (adding a new column)
                    new_label_column = f"labels_{level}"
                    df.loc[condition, new_label_column] = cell_type

        # Replace NaNs in the new label column with the annotations from the previous level
        if level > 0:
            previous_label_column = f"labels_{level - 1}"
            new_label_column = f"labels_{level}"
            df[new_label_column] = df[new_label_column].fillna(df[previous_label_column])
        else:
            # If level is 0, copy the root labels to the new label column
            df["labels_0"] = df["_labels"]

        # Update queue and increase level
        queue = next_queue
        level += 1

    # selecting only the new cell type annotations
    df = df[[x for x in df.columns if "labels_" in x]]
    return df
