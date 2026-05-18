import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.base import Visualizer
from yellowbrick.classifier import (
    ClassificationReport,
    ROCAUC,
    PrecisionRecallCurve,
    ClassPredictionError,
    DiscriminationThreshold,
)
from yellowbrick.style import set_palette

from Hive_ML.training.models import (
    adab_tree,
    random_forest,
    knn,
    decicion_tree,
    lda,
    qda,
    naive,
    svm_kernel,
    logistic_regression,
    ridge,
    mlp,
)

from sklearn.decomposition import PCA
from os import PathLike
from typing import List, Dict, Union
from pandas import DataFrame
from typing import Tuple
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from Hive_ML.training.model_trainer import model_fit_and_predict
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from Hive_ML.utilities.feature_utils import feature_normalization, prepare_features
from sklearn.base import ClassifierMixin

set_palette("sns_pastel")

MODELS = {
    "rf": random_forest,
    "adab": adab_tree,
    "lda": lda,
    "qda": qda,
    "logistic_regression": logistic_regression,
    "knn": knn,
    "naive": naive,
    "decision_tree": decicion_tree,
    "svm": svm_kernel,
    "ridge": ridge,
    "mlp": mlp,
}

AGGR_NUMPY = {"median": np.median, "mean": np.mean}

YB_VISUALIZERS = {
    "Report": ClassificationReport,
    "ROCAUC": ROCAUC,
    "PR": PrecisionRecallCurve,
    "CPE": ClassPredictionError,
    "DT": DiscriminationThreshold,
}


def select_best_classifiers(
    df_summary: DataFrame, metric: str, reduction: str, k: int = 1
) -> Tuple[List[Tuple[str, str]], List[float]]:
    """
    Given a DataFrame containing Validation scores for different Classifiers and Number of Selected Features,
    returns the k-best combinations and their respective reduced score (mean or median over the validation splits).

    Parameters
    ----------
    df_summary  :
        Validation DataFrame Summary.
    metric  :
        Metric to consider to select the best performance.
    reduction   :
        Reduction to apply to the validation splits to select the best performance.
    k   :
        Number of the best combinations to select.

    Returns
    -------
        Selected best combinations [(N_Features, Classifier), (N_Features, Classifier), ... ] and corresponding reduced validation scores.
    """
    aggr = df_summary[df_summary["Metric"] == metric][["Value", "Classifier"]].groupby(["Classifier"]).agg(reduction)

    aggr = aggr.loc[aggr["Value"].nlargest(k).index]
    classifiers = aggr.index.values

    n_features = []
    best_val_scores = []
    for classifier in classifiers:
        aggr = (
            df_summary[(df_summary["Metric"] == metric) & (df_summary["Classifier"] == classifier)][
                ["Value", "N_Features"]
            ]
            .groupby(["N_Features"])
            .agg(reduction)
        )
        aggr = aggr.loc[aggr["Value"].nlargest(1).index]
        n_features.append(aggr.index.values[0])
        best_val_scores.append(aggr.values[0][0])

    n_features_selected_classifier = [(n_features[i], classifiers[i]) for i in range(len(classifiers))]

    n_features, selected_classifier = n_features_selected_classifier[0]
    print(f"Best Configuration: {selected_classifier}-{n_features}, {metric}: {best_val_scores[0]}")

    return n_features_selected_classifier, best_val_scores


def evaluate_classifiers(
    ensemble_configuration_df: DataFrame,
    classifier_kwargs_list: List[Dict],
    train_feature_set: np.ndarray,
    train_label_set: np.ndarray,
    test_feature_set: np.ndarray,
    test_label_set: np.ndarray,
    aggregation: str,
    feature_selection: str,
    visualizers: List[Dict] = None,
    output_file: Union[str, PathLike] = None,
    plot_title: str = "",
    random_state=None,
) -> Dict:
    """
    Evaluate ensemble Classification performance of provided classifiers, weighting and combining the single classifier predictions.
    If a list of YellowBrick Visualizers is provided, generates a single multi-plot report file.

    Parameters
    ----------
    ensemble_configuration_df:
        Dataframe containing the ensemble configuration. Each row should include `Classifier` , `N_Features` ( Number of
        Features to select), and `weight` ( weighting of the classifier prediction in the ensemble).
    classifier_kwargs_list  :
        List of classifiers kwargs Dict, used to configure the classifiers.
    train_feature_set   :
        Train Feature set used for the classifiers fitting.
    train_label_set :
        Train Label set used for the classifiers fitting.
    test_feature_set    :
        Test Feature set used for the classifiers evaluations.
    test_label_set  :
        Test Label set used for the classifiers evaluations.
    feature_selection   :
        Type of Feature Selection to perform ( ``SFFS`` or ``PCA``).
    aggregation :
        Type of Feature Aggregation.
    visualizers :
        List of YellowBrick Visualizers to use in the report plot generation.
    output_file :
        File location where to save the YellowBrick Plot Report.
    plot_title  :
        String used in the YellowBrick plots as title.

    Returns
    -------
        Dictionary with the ensemble classifier report ( including the classification metrics ).
    """
    fig, axs = plt.subplots(
        int(len(visualizers)),
        int(ensemble_configuration_df.shape[0]),
        figsize=(int(ensemble_configuration_df.shape[0]) * 10 * 1.5, int(len(visualizers)) * 10 * 1),
        squeeze=False,
    )

    visualgrid = []
    x_train, y_train, x_test, y_test = prepare_features(
        train_feature_set, train_label_set, None, aggregation, None, test_feature_set, test_label_set
    )

    x_train, x_test, _ = feature_normalization(x_train, x_test)

    ensemble_y_test_pred = np.zeros((x_test.shape[0], 2))

    ensemble_weights = ensemble_configuration_df["weight"].values

    weight_sum = np.sum(ensemble_weights)

    ensemble_weights = ensemble_weights / weight_sum

    for ensemble_idx, (classifier_configuration, classifier_kwargs, weight) in enumerate(
        zip(ensemble_configuration_df.iterrows(), classifier_kwargs_list, ensemble_weights)
    ):
        classifier, n_features = classifier_configuration[1]["Classifier"], classifier_configuration[1]["N_Features"]

        clf = MODELS[classifier](**classifier_kwargs, random_state=random_state)

        x_train, y_train, x_test, y_test = prepare_features(
            train_feature_set, train_label_set, None, aggregation, None, test_feature_set, test_label_set
        )

        x_train, x_test, _ = feature_normalization(x_train, x_test)

        if n_features != "All" and feature_selection == "SFFS":
            sffs_model = SFS(
                clf,
                k_features=int(n_features),
                forward=True,
                floating=True,
                scoring="roc_auc",
                verbose=0,
                n_jobs=-1,
                cv=5,
            )

            sffs = sffs_model.fit(x_train, y_train)
            sffs_features = sffs.subsets_

            feature_idx = sffs_features[n_features]["feature_idx"]

            x_train = x_train[:, feature_idx]
            x_test = x_test[:, feature_idx]

        if n_features != "All" and feature_selection == "PCA":
            pca = PCA(n_components=n_features)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)

        clf = MODELS[classifier](**classifier_kwargs)

        y_test_pred = model_fit_and_predict(clf, x_train, y_train, x_test)

        for idx_visualizer, visualizer in enumerate(visualizers):
            visualizers[visualizer]["ax"] = axs[idx_visualizer, ensemble_idx]
            visualizers[visualizer]["title"] = f"{plot_title} {visualizer}, {classifier}-{n_features}"
            visualgrid.append(YB_Visualizer(clf, visualizer, x_train, y_train, x_test, y_test, visualizers[visualizer]))

        ensemble_y_test_pred += y_test_pred * weight

    roc_auc_val = roc_auc_score(y_test, ensemble_y_test_pred[:, 1])
    report = classification_report(y_test, np.where(ensemble_y_test_pred[:, 1] > 0.5, 1, 0), output_dict=True)
    report["roc_auc"] = roc_auc_val

    if output_file is not None:
        plt.savefig(output_file)

    return report


def YB_Visualizer(
    clf: ClassifierMixin,
    visualizer: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kwargs: Dict,
) -> Visualizer:
    """
    Creates and Finalize a YellowBrick visualizer, given the classifier and the train/test features and corresponding labels
    to use for fitting and scoring.

    Parameters
    ----------
    clf :
        Classifier used by the Visualizer.
    visualizer  :
        visualizer name to create. Must match a value in YB_VISUALIZERS.
    x_train :
        Train Feature set used for the classifiers fitting.
    y_train :
        Train Label set used for the classifiers fitting.
    x_test  :
        Test Feature set used for the classifiers scoring.
    y_test  :
        Test Label set used for the classifiers scoring.
    kwargs  :
        Dictionary of kwargs for the YellowBrick Visualizer.

    Returns
    -------
        YellowBrick Visualizer finalized.
    """
    visualizer = YB_VISUALIZERS[visualizer](clf, **kwargs)

    if visualizer != "DT":
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
    else:
        x = np.vstack((x_train, x_test))
        y = np.vstack((y_train, y_test))
        visualizer.fit(x, y)
    if visualizer == "Report":
        visualizer.draw()
    visualizer.finalize()

    return visualizer
