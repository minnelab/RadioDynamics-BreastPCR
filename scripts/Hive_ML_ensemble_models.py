#!/usr/bin/env python

import datetime
import importlib.resources
import json
import numpy as np
import os
import pandas as pd
import plotly.express as px
import warnings
from Hive.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
)
from argparse import ArgumentParser, RawTextHelpFormatter
from joblib import parallel_backend
from pathlib import Path
from textwrap import dedent


from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.decomposition import PCA
import Hive_ML.configs
from Hive_ML.data_loader.feature_loader import load_feature_set
from Hive_ML.training.model_trainer import model_fit_and_predict
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
from Hive_ML.utilities.feature_utils import data_shuffling, feature_normalization, prepare_features
from Hive_ML.evaluation.model_evaluation import select_best_classifiers, evaluate_classifiers

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

COMPOSED_METRICS = {"sensitivity": lambda x: x["1"]["recall"], "specificity": lambda x: x["0"]["recall"]}

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

DESC = dedent("""
    Script to run 5-CV Model Ensembling (after performing Feature Selection and Model Fitting) on a Feature Set. The Metrics evaluation
    summary (in Excel format) is saved in the experiment folder, defined by the ``experiment_name`` argument.
    The models to ensemble are provided as an input DataFrame
    """)  # noqa: E501
EPILOG = dedent("""
    Example call:
    ::
        {filename} -feature-file /path/to/feature_table.csv --config-file config_file.json --experiment-name Radiomics --ensemble-config <ENSEMBLE.csv>
    """.format(filename=Path(__file__).name))  # noqa: E501


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--feature-file",
        type=str,
        required=True,
        help="Input Dataset folder",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration JSON file with experiment and dataset parameters.",
    )

    pars.add_argument(
        "--ensemble-config",
        type=str,
        required=True,
        help="Configuration DataFrame containing the model information ( Classifier + N-Features + Ensembling Weights) to run ensembling.",
    )

    pars.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name used to save the model fitting metrics evaluation summary.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Hive_ML.configs, arguments["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    models = config_dict["models"]

    aggregation = "Flat"
    stats_4D = False
    flatten_features = True

    if "feature_aggregator" in config_dict:
        aggregation = config_dict["feature_aggregator"]
        if aggregation != "Flat":
            stats_4D = True
            flatten_features = False
        elif aggregation.endswith("Norm"):
            stats_4D = False
            flatten_features = False

    (
        feature_set,
        subject_ids,
        subject_labels,
        feature_names,
        mean_features,
        sum_features,
        std_features,
        mean_delta_features,
    ) = load_feature_set(arguments["feature_file"], get_4D_stats=stats_4D, flatten_features=flatten_features)

    if aggregation == "Flat":
        features = feature_set
    elif aggregation == "Mean":
        features = mean_features
    elif aggregation == "SD":
        features = std_features
    elif aggregation == "Sum":
        features = sum_features
    elif aggregation == "Delta":
        features = mean_delta_features

    label_set = np.array(subject_labels)

    if "test_size" not in config_dict:
        config_dict["test_size"] = 0.2

    if aggregation.endswith("Norm"):
        features = feature_set

        feature_set_3D = np.array(features).squeeze(-2)

        train_feature_set, train_label_set, test_feature_set, test_label_set = data_shuffling(
            np.swapaxes(feature_set_3D, 0, 1), label_set, config_dict["random_seed"], test_size=config_dict["test_size"]
        )

    else:

        n_features = features.shape[1]
        n_subjects = features.shape[0]

        filtered_feature_set = []
        filtered_feature_names = []
        features = np.nan_to_num(features)
        for feature in range(n_features):
            exclude = False
            for feature_val in np.unique(features[:, feature]):
                if (np.count_nonzero(features[:, feature] == feature_val) / n_subjects) > 0.5:
                    exclude = True
                    print("Excluding:", feature_names[feature])
                    break

            if not exclude:
                filtered_feature_set.append(list(features[:, feature]))
                filtered_feature_names.append(feature_names[feature])

        feature_set = np.vstack(filtered_feature_set).T
        feature_names = filtered_feature_names

        print("# Features: {}".format(feature_set.shape[1]))
        print("# Labels: {}".format(label_set.shape))

        train_feature_set, train_label_set, test_feature_set, test_label_set = data_shuffling(
            feature_set, label_set, config_dict["random_seed"], test_size=config_dict["test_size"]
        )

    experiment_name = arguments["experiment_name"]

    experiment_dir = Path(os.environ["ROOT_FOLDER"]).joinpath(
        experiment_name, config_dict["feature_selection"], aggregation, "FS"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    n_features = config_dict["n_features"]
    if n_features > train_feature_set.shape[1]:
        n_features = train_feature_set.shape[1]

    n_iterations = 0
    for classifier in models:
        if classifier in ["rf", "adab"]:
            n_iterations += config_dict["n_folds"]
        else:
            n_iterations += config_dict["n_folds"] * n_features

    pbar = tqdm(total=n_iterations)

    df_summary = []

    visualizers = {
        "Report": {"support": True, "classes": [config_dict["label_dict"][key] for key in config_dict["label_dict"]]},
        "ROCAUC": {
            "micro": False,
            "macro": False,
            "per_class": False,
            "classes": [config_dict["label_dict"][key] for key in config_dict["label_dict"]],
        },
        "PR": {},
        "CPE": {"classes": [config_dict["label_dict"][key] for key in config_dict["label_dict"]]},
        "DT": {},
    }

    ensemble_configuration = pd.read_csv(arguments["ensemble_config"])
    feature_selection_method = config_dict["feature_selection"]
    reduction = config_dict["reduction_best_model"]
    metric = config_dict["metric_best_model"]
    plot_title = f"{experiment_name} {feature_selection_method} {aggregation}"

    classifiers = ensemble_configuration["Classifiers"].values

    classifier_kwargs_list = [models[classifier] for classifier in classifiers]

    with parallel_backend("loky", n_jobs=-1):
        kf = StratifiedKFold(n_splits=config_dict["n_folds"], random_state=config_dict["random_seed"], shuffle=True)
        for fold, (train_index, val_index) in enumerate(kf.split(train_feature_set, train_label_set)):
            output_file = str(
                Path(os.environ["ROOT_FOLDER"]).joinpath(
                    experiment_name,
                    f"{experiment_name} {feature_selection_method} {aggregation} {reduction}_{fold}.png",
                )
            )
            report = evaluate_classifiers(
                ensemble_configuration,
                classifier_kwargs_list,
                train_feature_set[train_index, :],
                train_label_set[train_index],
                test_feature_set[val_index, :],
                test_label_set[val_index],
                aggregation,
                feature_selection_method,
                visualizers,
                output_file,
                plot_title,
                config_dict["random_seed"],
            )

            roc_auc_val = report[metric]

            df_summary.append(
                {
                    "Value": roc_auc_val,
                    "Classifier": "Ensemble",
                    "Metric": metric,
                    "Fold": str(fold),
                    "N_Features": "All",
                    "Experiment": experiment_name + "_" + config_dict["feature_selection"] + "_" + aggregation,
                }
            )
            pbar.update(1)

    df_summary.to_excel(
        Path(os.environ["ROOT_FOLDER"]).joinpath(
            experiment_name, experiment_name + "_" + feature_selection_method + f"_{aggregation}.xlsx"
        )
    )


if __name__ == "__main__":
    main()
