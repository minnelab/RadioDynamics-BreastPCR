from os import PathLike
from typing import Union, Tuple, List

import numpy
import numpy as np
import pandas as pd

from Hive_ML.utilities.feature_utils import get_feature_set_details, get_4D_feature_stats, flatten_4D_features


def load_feature_set(
    feature_set_filename: Union[str, PathLike],
    get_4D_stats: bool = True,
    flatten_features: bool = False,
    select_T: int = None,
) -> Tuple[numpy.ndarray, List[str], List[str], List[str], numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Function to load a feature set from a filepath, including the Subject list, their corresponding labels and a list of feature names.
    If ``get_4D_stats`` is set to **True**, the 4D statistics of the feature set are returned (see :func:`utilities.feature_utils.get_4D_feature_stats` )
    If ``flatten_features`` is set to **True**, the 3D feature set is flattened into a 2D set (see :func:`utilities.feature_utils.flatten_4D_features`)
    If ``select_T`` is set to an integer value, the specific sequence is extracted and returned from the 3D feature set.

    Parameters
    ----------
    feature_set_filename    :
        Feature set file path.
    get_4D_stats    :
        Flag to compute and return sequence statistics.
    flatten_features    :
        Flag to flatten features along the sequence dimension.
    select_T    :
        Select and return only the specified sequence.

    Returns
    -------
        Feature set Array , Subject list, Subject labels, List of feature names, Mean Sequence Array,
        SD Sequence Array, Sum Sequence Array, Mean Delta Array.
    """
    if feature_set_filename.endswith(".xlsx"):
        feature_set = pd.read_excel(feature_set_filename, index_col=0)
    elif feature_set_filename.endswith(".csv"):
        feature_set = pd.read_csv(feature_set_filename, index_col=0)
    elif feature_set_filename.endswith(".pkl"):
        feature_set = pd.read_pickle(feature_set_filename, index_col=0)
    else:
        raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")

    feature_set = feature_set.sort_values(by=["Subject_Label", "Subject_ID", "Sequence_Number"])
    feature_list, subject_ids, subject_labels, feature_names = get_feature_set_details(feature_set)

    if get_4D_stats:
        mean_features, sum_features, std_features, mean_delta_features = get_4D_feature_stats(feature_list)
    else:
        mean_features, sum_features, std_features, mean_delta_features = None, None, None, None

    if flatten_features:
        feature_list, feature_names = flatten_4D_features(feature_list, feature_names)
    if select_T is not None:
        feature_list = np.array(feature_list).squeeze(axis=-2)[int(select_T), :, :]

    return (
        np.array(feature_list),
        subject_ids,
        subject_labels,
        feature_names,
        mean_features,
        sum_features,
        std_features,
        mean_delta_features,
    )
