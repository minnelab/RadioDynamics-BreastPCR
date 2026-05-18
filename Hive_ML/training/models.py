from typing import Sequence, Dict, List, Union

from sklearn import naive_bayes
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def logistic_regression(random_state: int = None) -> LogisticRegression:
    """
    Function to create and return a ``Logistic Regression`` classifier.

    Parameters
    ----------
    random_state :
        Random state to initialize the classifier.

    Returns
    -------
        Logistic Regression classifier.
    """
    clf = LogisticRegression(random_state=random_state)
    return clf


def mlp(
    hidden_layer_sizes: Sequence[int] = (10, 10, 10),
    solver: str = "adam",
    activation: str = "relu",
    random_state: int = None,
) -> MLPClassifier:
    """
    Function to create and return a ``Multy-Layer Perceptron`` classifier.

    Parameters
    ----------
        hidden_layer_sizes  :
            List of Hidden layers dimensions in the MLP.
        solver  :
            Optimizer. Default: ``Adam``.
        activation :
            Activation function. Default: ``ReLU``.
        random_state :
            Random state to initialize the classifier.
    Returns
    -------
        MLP classifier.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, solver=solver, activation=activation, random_state=random_state
    )
    return clf


def decicion_tree(
    criterion: str = "gini", max_depth: int = 10, class_weight: Union[Dict, List] = None, random_state: int = None
) -> DecisionTreeClassifier:
    """
    Function to create and return a ``Decision Tree`` classifier.

    Parameters
    ----------
        criterion   :
            selected from either ``gini`` or ``entropy``. Default: ``gini``.
        max_depth   :
            the maximum depth of the trees. Default: ``10``.
        class_weight    :
            Assigning weights to  class labels e.g., ``{0:1, 1:2}``.
        random_state :
            Random state to initialize the classifier.

    Returns
    -------
        Decision Tree classifier.
    """

    clf = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, class_weight=class_weight, random_state=random_state
    )

    return clf


def random_forest(
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: int = 10,
    class_weight: Union[Dict, List] = None,
    max_samples: Union[int, float] = None,
    random_state: int = None,
) -> RandomForestClassifier:
    """
    Function  to create and return a ``Random Forest`` classifier.

    Parameters
    ----------
    max_samples :
        The number of samples to draw to train each base estimator.
    n_estimators :
        set the number of trees in the forest. Default: ``100``.
    criterion :
        selected from either 'gini' or 'entropy'. Default: ``gini``.
    max_depth :
        the maximum depth of the trees. Default: ``10``.
    class_weight :
         Assigning weights to  class labels e.g. : ``{0:1, 1:2}``.
    random_state :
        Random state to initialize the classifier.
    Returns
    -------
    clf : class
        Random Forest classifier.
    """

    clf = RandomForestClassifier(
        criterion=criterion,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        max_samples=max_samples,
        n_jobs=-1,
        random_state=random_state,
    )
    return clf


def svm_kernel(
    kernel: str = "linear",
    poly_degree: int = 3,
    c_val: int = 1,
    class_weight: Union[Dict, List] = None,
    random_state: int = None,
) -> svm.SVC:
    """
    Function  to create and return a ``Support Vector Machine`` classifier.

    Parameters
    ----------
    kernel :
        selected from ``linear``, ``rbf``, ``gaussian``, or ``poly``. Default : ``linear``.
    poly_degree :
        specify the degree of polynomial if poly kernel used.
    c_val :
        regularization parameter.
    class_weight :
         Assigning weights to  class labels e.g. : ``{0:1, 1:2}``.
    random_state :
        Random state to initialize the classifier.
    Returns
    -------
        SVM Classifier.
    """

    clf = svm.SVC(
        kernel=kernel,
        degree=poly_degree,
        gamma="scale",
        C=c_val,
        tol=1e-1,
        class_weight=class_weight,
        probability=True,
        random_state=random_state,
        max_iter=-1,
    )

    return clf


def knn(neighbors=5, random_state: int = None) -> KNeighborsClassifier:
    """
    Function  to create and return a ``k-NN`` classifier.

    Parameters
    ----------
    neighbors   :
        Nearest neighbours to consider for calculation.
    random_state :
        Random state to initialize the classifier.
    Returns
    -------
        k-NN Classifier.
    """
    clf = KNeighborsClassifier(n_neighbors=neighbors, weights="distance", n_jobs=1)

    return clf


def adab_tree(
    max_depth: int = 10,
    criterion: str = "gini",
    class_weight: Union[Dict, List] = None,
    n_estimators: int = 100,
    random_state: int = None,
) -> AdaBoostClassifier:
    """
    Function  to create and return a ``AdaBoost`` classifier.

    Parameters
    ----------
    n_estimators :
        set the number of trees in the forest. Default: ``100``.
    criterion :
        selected from either 'gini' or 'entropy'. Default: ``gini``.
    max_depth :
        the maximum depth of the trees. Default: ``10``.
    class_weight :
         Assigning weights to  class labels e.g. : ``{0:1, 1:2}``.
    random_state :
        Random state to initialize the classifier.
    Returns
    -------
        AdaBoost classifier.
    """

    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, class_weight=class_weight),
        n_estimators=n_estimators,
        random_state=random_state,
    )
    return clf


def lda(random_state: int = None) -> LinearDiscriminantAnalysis:
    """
    Function  to create and return a ``Linear Discriminant Analysis`` classifier.

    Parameters
    ----------
    random_state :
        Random state to initialize the classifier.

    Returns
    -------
        LDA Classifier
    """
    clf = LinearDiscriminantAnalysis()

    return clf


def qda(random_state: int = None) -> QuadraticDiscriminantAnalysis:
    """
    Function  to create and return a ``Quadratic Discriminant Analysis`` classifier.

    Parameters
    ----------
    random_state :
        Random state to initialize the classifier.

    Returns
    -------
        QDA Classifier
    """
    clf = QuadraticDiscriminantAnalysis()

    return clf


def naive(random_state: int = None) -> naive_bayes.GaussianNB:
    """
    Function  to create and return a ``Gaussian Naive Bayes`` classifier.

    Parameters
    ----------
    random_state :
        Random state to initialize the classifier.

    Returns
    -------
        Gaussian Naive Bayes Classifier
    """
    clf = naive_bayes.GaussianNB()

    return clf


def ridge(random_state: int = None):
    """
    Function  to create and return a ``Ridge`` classifier.

    Parameters
    ----------
    random_state :
        Random state to initialize the classifier.

    Returns
    -------
        Ridge Classifier
    """
    clf = RidgeClassifier(random_state=random_state)

    return clf
