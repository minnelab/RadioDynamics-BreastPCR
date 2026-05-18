import numpy
import numpy as np
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import Tuple, List, Any


def get_feature_set_details(feature_set: DataFrame) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    Function to extract details from a DataFrame, including the Subject IDs, the Subject Labels,and a list of Feature names.
    The 2-D Feature Set *[ (n_subjects x n_sequences) x n_features ]* is converted to a List of Lists, where the external list
    contains one element per sequence, and the internal one contains a list of features per each subject.

    Example:
        .. math::
            feature_{set}[sequence][subject][feature].

    Parameters
    ----------
    feature_set :
        Pandas DataFrame containing the feature set and the details.

    Returns
    -------
        3D Feature List [ n_sequences x n_subjects x n_features ], List of Subject IDS, List of Subject Labels and List of Feature names.

    """
    n_sequences = feature_set[["Subject_ID", "Sequence_Number"]].groupby("Subject_ID").count()
    n_sequences = max(n_sequences["Sequence_Number"])
    subjects_and_labels = feature_set[["Subject_ID", "Subject_Label"]].drop_duplicates()

    subject_ids = subjects_and_labels["Subject_ID"].values
    subject_labels = subjects_and_labels["Subject_Label"].values

    feature_list = []
    for sequence in range(n_sequences):
        sequence_feature_list = []
        for subject in subject_ids:
            feature_set_subject = feature_set[
                (feature_set["Subject_ID"] == subject) & (feature_set["Sequence_Number"] == sequence)]
            feature_set_subject = feature_set_subject.copy(deep=True)
            feature_set_subject.drop("Subject_ID", inplace=True, axis=1)
            feature_set_subject.drop("Subject_Label", inplace=True, axis=1)
            feature_set_subject.drop("Sequence_Number", inplace=True, axis=1)
            if feature_set_subject.values.shape[0] > 0:
                sequence_feature_list.append(feature_set_subject.values)
            else:
                nan_array = numpy.empty((1, feature_set_subject.values.shape[1]))
                nan_array[:] = numpy.nan
                sequence_feature_list.append(nan_array)
        feature_list.append(sequence_feature_list)

    feature_set_names = feature_set.copy(deep=True)
    feature_set_names.drop("Subject_ID", inplace=True, axis=1)
    feature_set_names.drop("Subject_Label", inplace=True, axis=1)
    feature_set_names.drop("Sequence_Number", inplace=True, axis=1)
    feature_names = feature_set_names.columns
    return feature_list, subject_ids, subject_labels, feature_names


def get_4D_feature_stats(feature_list: List[List[Any]]) -> Tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    r"""
    Function to accumulate the feature set (with size *[ n_sequences x n_subjects x n_features ]* ) along the sequence dimension.
    Returns 4 Numpy arrays (each with size *[ n_subjects x n_features ]*), including average, sum, standard deviation and mean
    delta values along the sequence dimension.

    .. math::
        Mean F_{s,f} = \frac{1}{T} \sum_{t=0}^{T} F_{t,s,f}

        Sum F_{s,f} = \sum_{t=0}^{T} F_{t,s,f}

        SD F_{s,f} = \sqrt{ \frac{1}{T} \sum_{t=0}^{T} (F_{t,s,f} - Mean F_{s,f})^2 }

        Mean Delta F_{s,f} = \frac{1}{T} \sum_{t=0}^{T} | F_{t,s,f} - Mean F_{s,f} |

    Parameters
    ----------
    feature_list    :
        Pandas DataFrame

    Returns
    -------
        Mean Sequence Array, SD Sequence Array, Sum Sequence Array, Mean Delta Array.
    """
    feature_arrays = np.array(feature_list).squeeze(axis=-2)

    mean_features = np.nanmean(feature_arrays, axis=0)
    sum_features = np.nansum(feature_arrays, axis=0)
    std_features = np.nanstd(feature_arrays, axis=0)

    delta_features = np.absolute(np.subtract(feature_arrays, mean_features))
    mean_delta_features = np.nanmean(delta_features, axis=0)

    return mean_features, sum_features, std_features, mean_delta_features


def flatten_4D_features(feature_list: List[List[Any]], feature_names: List[str]) -> Tuple[numpy.ndarray, List[str]]:
    """
    Function to flatten a 3D Feature set (with size *[ n_sequences x n_subjects x n_features ]* ) into a 2D Feature Set
    *[ n_subjects, n_flatten_features ]*, where the features are flattened along axis 1. The total number of flattened features
    is equal to *n_features x n sequences* .

    Parameters
    ----------
    feature_list    :
        3D Feature Set to flatten
    feature_names   :
        List of feature names

    Returns
    -------
        Flatten 2D Feature set, updated Feature name list (appending the corresponding sequence index at each feature name).
    """
    feature_arrays = np.array(feature_list).squeeze(axis=-2)
    flat_feature_array = np.zeros((feature_arrays.shape[1], 1))

    if feature_arrays.shape[0] > 1:
        T = feature_arrays.shape[0]
        n_features = feature_arrays.shape[2]
        for n in range(n_features):
            for t in range(T):
                flat_feature_array = np.hstack(
                    [flat_feature_array, feature_arrays[t, :, n].reshape(feature_arrays.shape[1], 1)])

        flatten_feature_names = []
        for feature_name in feature_names:
            for t in range(T):
                flatten_feature_names.append("{}_{}".format(t, feature_name))
        return flat_feature_array[:, 1:], flatten_feature_names

    else:
        return feature_list, feature_names


def data_shuffling(feature_set: numpy.ndarray, label_set: numpy.ndarray, seed_val: int, test_size: float = 0.2, split_file = None) -> \
Tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Function to randomly shuffle the feature set and the corresponding label set along the subject dimension.
    If a split file is provided, the feature set and label set will be split according to the split file.
    Parameters
    ----------
    feature_set :
        Feature set to shuffle.
    label_set   :
        Label set to shuffle.
    seed_val    :
        Random seed generator.
    test_size   :
        Proportion of dataset to include in the Test set. Defaults to 0.2.
    Returns
    -------
        Shuffled Feature set and Label Set
    """
    if float(test_size) == 0:
        X_train, y_train = shuffle(feature_set, label_set, random_state=seed_val)
        return X_train, y_train, None, None
    X_train, X_test, y_train, y_test = train_test_split(
        feature_set, label_set, test_size=float(test_size), random_state=seed_val, stratify=label_set)

    return X_train, y_train, X_test, y_test


def feature_normalization(x_train: numpy.ndarray, x_val: numpy.ndarray = None, x_test: numpy.ndarray = None) -> Tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Normalize each feature in the range 0 to 1

    Parameters
    ----------
    x_train :
        Feature matrix of training set.
    x_val :
        Feature matrix of validation set.
    x_test :
        Feature matrix of test set. The default is None.

    Returns
    -------
        normalized feature sets based on the statistics of training features.
    """
    min_max_norm = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_train = min_max_norm.fit_transform(x_train)
    if x_val is not None:
        x_val = min_max_norm.transform(x_val)

    if x_test is not None:
        x_test = min_max_norm.transform(x_test)

    return x_train, x_val, x_test


def prepare_features(feature_set: numpy.ndarray, label_set: numpy.ndarray, train_index: List[int], aggregation: str,
                     val_index: List[int] = None,
                     val_feature_set: numpy.ndarray = None,
                     val_label_set: numpy.ndarray = None) -> Tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Function to prepare a feature set into a train/test split, according to the provided indexes. If the
    ``feature_set.shape`` is 3D, performs a channel-wise (axis=1) normalization, optionally followed by a reduction (
    **mean**, **std** )  along the same axis.

    Parameters
    ----------
    feature_set :
        Original feature set to split.
    label_set   :
        Original label set to split.
    train_index :
        Train indexes to extract train split from the feature set. Ignored if ``val_label_set`` and ``val_feature_set`` are provided.
    aggregation :
        Aggregation type performed on the feature set. If ``Mean_Norm`` or ``SD_Norm``, perform reduction along axis 1.
    val_index   :
        Validation indexes to extract validation split from the feature set. Ignored if ``val_label_set`` and ``val_feature_set`` are provided.
    val_feature_set :
        Optional Validation Feature set, to directly provide the validation split data.
    val_label_set   :
        Optional Validation Label set, to directly provide the validation split data.

    Returns
    -------
        Train/Validation Feature and Label Data.

    """
    x_val = None
    y_val = None

    if val_feature_set is not None:
        x_train = feature_set
        x_val = val_feature_set
    else:
        x_train = feature_set[train_index, :]
        if val_index is not None:
            x_val = feature_set[val_index, :]

    if val_label_set is not None:
        y_train = label_set
        y_val = val_label_set
    else:
        y_train = label_set[train_index]
        if val_index is not None:
            y_val = label_set[val_index]

    if len(x_train.shape) > 2:
        for t in range(x_train.shape[1]):
            min_max_norm = preprocessing.MinMaxScaler(feature_range=(0, 1))
            x_train[:, t, :] = min_max_norm.fit_transform(x_train[:, t, :])
            if x_val is not None:
                x_val[:, t, :] = min_max_norm.transform(x_val[:, t, :])

    if aggregation == "Mean_Norm":
        x_train = np.nanmean(x_train, axis=1)
        if x_val is not None:
            x_val = np.nanmean(x_val, axis=1)
    elif aggregation == "SD_Norm":
        x_train = np.nanstd(x_train, axis=1)
        if x_val is not None:
            x_val = np.nanstd(x_val, axis=1)

    return x_train, y_train, x_val, y_val
