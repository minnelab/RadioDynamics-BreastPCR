import numpy as np
from sklearn.base import ClassifierMixin


def model_fit_and_predict(
    model: ClassifierMixin, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray
) -> np.ndarray:
    """
    Function to fit a SKLearn with the given training features and training labels, returning the predicted the labels for the validation features.


    Parameters
    ----------
    model :
        a SKlearn model.
    x_train   :
        training features.
    y_train   :
        training labels.
    x_val :
        validation features.

    Returns
    -------
        predicted validation labels.

    """
    model.fit(x_train, y_train)

    y_pred_prob = model.predict_proba(x_val)

    return y_pred_prob
