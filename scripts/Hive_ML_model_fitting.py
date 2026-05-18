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

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

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
    Script to run 5-CV Forward Model Fitting (after performing Feature Selection) on a Feature Set. The Metrics evaluation
    summary (in Excel format) is saved in the experiment folder, defined by the ``experiment_name`` argument.
    """)  # noqa: E501
EPILOG = dedent("""
    Example call:
    ::
        {filename} -feature-file /path/to/feature_table.csv --config-file config_file.json --experiment-name Radiomics
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
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name used to save the model fitting metrics evaluation summary.",
    )

    pars.add_argument(
        "--split-file",
        type=str,
        required=False,
        help="Split file used to split the data into train and test sets.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )

    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Hive_ML.configs, arguments["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    models = config_dict["models"]
    metrics = ["accuracy", "roc_auc", "specificity", "sensitivity"]

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

    split_file = arguments["split_file"] if arguments["split_file"] else None
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
        if not Path(split_file).is_file():
            train_feature_set, train_label_set, test_feature_set, test_label_set = data_shuffling(
                np.swapaxes(feature_set_3D, 0, 1),
                label_set,
                config_dict["random_seed"],
                test_size=config_dict["test_size"],
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

        if not Path(split_file).is_file():
            train_feature_set, train_label_set, test_feature_set, test_label_set = data_shuffling(
                feature_set, label_set, config_dict["random_seed"], test_size=config_dict["test_size"]
            )

    if Path(split_file).is_file():
        logger.info(f"Using split file: {split_file}")
        split_data = pd.read_csv(split_file)
        train_ids = split_data["train_split"].tolist()
        test_ids = split_data["test_split"].tolist()

    train_feature_set = []
    test_feature_set = []
    train_label_set = []
    test_label_set = []

    for feature, label, subject_id in zip(feature_set, subject_labels, subject_ids):
        if subject_id in train_ids:
            train_feature_set.append(feature)
            train_label_set.append(label)
        elif subject_id in test_ids:
            test_feature_set.append(feature)
            test_label_set.append(label)

    train_feature_set = np.array(train_feature_set)
    test_feature_set = np.array(test_feature_set)
    train_label_set = np.array(train_label_set)
    test_label_set = np.array(test_label_set)

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

    if config_dict["feature_selection"] == "SFFS":
        with open(
            Path(os.environ["ROOT_FOLDER"]).joinpath(
                experiment_name,
                config_dict["feature_selection"],
                aggregation,
                "FS",
                f"{experiment_name}_FS_summary.json",
            ),
            "rb",
        ) as fp:
            feature_selection = json.load(fp)

    pbar = tqdm(total=n_iterations)

    df_summary = []
    feature_selection_method = config_dict["feature_selection"]
    if (
        Path(os.environ["ROOT_FOLDER"])
        .joinpath(experiment_name, f"{experiment_name}_{feature_selection_method}_{aggregation}.xlsx")
        .exists()
    ):
        df_summary = pd.read_excel(
            Path(os.environ["ROOT_FOLDER"]).joinpath(
                experiment_name, f"{experiment_name}_{feature_selection_method}_{aggregation}.xlsx"
            )
        )
    else:
        with parallel_backend("loky", n_jobs=-1):
            for classifier in models:
                pbar.set_description(f"{classifier}, Model Fitting")
                if classifier in ["rf", "adab"]:
                    n_features = "All"
                elif classifier == "knn":
                    n_features = 10
                else:
                    n_features = config_dict["n_features"]
                    if n_features > train_feature_set.shape[-1]:
                        n_features = train_feature_set.shape[-1]
                if n_features != "All":
                    for n_feature in range(1, n_features + 1):
                        kf = StratifiedKFold(
                            n_splits=config_dict["n_folds"], random_state=config_dict["random_seed"], shuffle=True
                        )
                        for fold, (train_index, val_index) in enumerate(kf.split(train_feature_set, train_label_set)):
                            if fold != 0:
                                continue
                            # x_train, y_train, x_val, y_val = prepare_features(train_feature_set, train_label_set,
                            #                                                train_index, aggregation, val_index)
                            x_train = train_feature_set
                            y_train = train_label_set
                            x_val = test_feature_set
                            y_val = test_label_set

                            if config_dict["feature_selection"] == "SFFS":

                                selected_features = feature_selection[classifier][str(fold)][str(n_feature)][
                                    "feature_names"
                                ]
                                train_feature_name_list = list(feature_names)

                                feature_idx = []

                                for selected_feature in selected_features:
                                    feature_idx.append(train_feature_name_list.index(selected_feature))

                                x_train = x_train[:, feature_idx]

                                x_val = x_val[:, feature_idx]

                            elif config_dict["feature_selection"] == "PCA":
                                pca = PCA(n_components=n_features)
                                x_train = pca.fit_transform(x_train)
                                x_val = pca.transform(x_val)

                            x_train, x_val, _ = feature_normalization(x_train, x_val)
                            clf = MODELS[classifier](**models[classifier], random_state=config_dict["random_seed"])

                            y_val_pred = model_fit_and_predict(clf, x_train, y_train, x_val)

                            roc_auc_val = roc_auc_score(y_val, y_val_pred[:, 1])

                            report = classification_report(
                                y_val, np.where(y_val_pred[:, 1] > 0.5, 1, 0), output_dict=True
                            )
                            report["roc_auc"] = roc_auc_val

                            for metric in metrics:
                                if metric not in report:
                                    report[metric] = COMPOSED_METRICS[metric](report)
                                df_summary.append(
                                    {
                                        "Value": report[metric],
                                        "Classifier": classifier,
                                        "Metric": metric,
                                        "Fold": str(fold),
                                        "N_Features": n_feature,
                                        "Experiment": experiment_name
                                        + "_"
                                        + config_dict["feature_selection"]
                                        + "_"
                                        + aggregation,
                                    }
                                )
                            pbar.update(1)
                else:
                    kf = StratifiedKFold(
                        n_splits=config_dict["n_folds"], random_state=config_dict["random_seed"], shuffle=True
                    )
                    for fold, (train_index, val_index) in enumerate(kf.split(train_feature_set, train_label_set)):
                        if fold != 0:
                            continue
                        # x_train, y_train, x_val, y_val = prepare_features(train_feature_set, train_label_set, train_index,
                        #                                                aggregation, val_index)
                        x_train = train_feature_set
                        y_train = train_label_set
                        x_val = test_feature_set
                        y_val = test_label_set

                        x_train, x_val, _ = feature_normalization(x_train, x_val)
                        clf = MODELS[classifier](**models[classifier], random_state=config_dict["random_seed"])

                        y_val_pred = model_fit_and_predict(clf, x_train, y_train, x_val)

                        roc_auc_val = roc_auc_score(y_val, y_val_pred[:, 1])

                        report = classification_report(y_val, np.where(y_val_pred[:, 1] > 0.5, 1, 0), output_dict=True)
                        report["roc_auc"] = roc_auc_val

                        for metric in metrics:
                            if metric not in report:
                                report[metric] = COMPOSED_METRICS[metric](report)
                            df_summary.append(
                                {
                                    "Value": report[metric],
                                    "Classifier": classifier,
                                    "Metric": metric,
                                    "Fold": str(fold),
                                    "N_Features": "All",
                                    "Experiment": experiment_name
                                    + "_"
                                    + config_dict["feature_selection"]
                                    + "_"
                                    + aggregation,
                                }
                            )
                        pbar.update(1)

        df_summary = pd.DataFrame.from_records(df_summary)

        df_summary.to_excel(
            Path(os.environ["ROOT_FOLDER"]).joinpath(
                experiment_name, experiment_name + "_" + feature_selection_method + f"_{aggregation}.xlsx"
            )
        )

        df_summary_all = df_summary[df_summary["N_Features"] == "All"]
        df_summary_all = df_summary_all.drop(["Fold"], axis=1)
        df_summary = df_summary[df_summary["N_Features"] != "All"]

        df_summary = df_summary[df_summary["N_Features"] <= 20]
        df_summary = df_summary.drop(["Fold"], axis=1)
        df_summary = pd.concat([df_summary, df_summary_all])

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

    metric = config_dict["metric_best_model"]
    reduction = config_dict["reduction_best_model"]
    plot_title = f"{experiment_name} {feature_selection_method} {aggregation}"
    val_scores = []

    features_classifiers, scores = select_best_classifiers(df_summary, metric, reduction, 1)

    val_scores.append(
        {
            "Metric": metric,
            "Experiment": experiment_name,
            "Score": scores[0],
            "Section": f"Validation Set [5-CV {reduction.capitalize()}]",
        },
    )

    for k in config_dict["k_ensemble"]:
        features_classifiers, scores = select_best_classifiers(df_summary, metric, reduction, k)

        classifiers = [classifier for n_features, classifier in features_classifiers]
        n_feature_list = [n_features for n_features, classifier in features_classifiers]
        classifier_kwargs_list = [models[classifier] for classifier in classifiers]

        ensemble_weights = scores

        ensemble_configuration_df = []

        for classifier, n_features, weight in zip(classifiers, n_feature_list, ensemble_weights):
            ensemble_configuration_df.append({"Classifier": classifier, "N_Features": n_features, "weight": weight})

        ensemble_configuration = pd.DataFrame.from_records(ensemble_configuration_df)

        print(ensemble_configuration)
        ensemble_configuration.to_excel(
            Path(os.environ["ROOT_FOLDER"]).joinpath(experiment_name, f"{plot_title}_TOP_{k}.xlsx")
        )
        output_file = str(
            Path(os.environ["ROOT_FOLDER"]).joinpath(
                experiment_name, f"{experiment_name} {feature_selection_method} {aggregation} {reduction}_{k}.png"
            )
        )

        if test_feature_set is not None:
            report = evaluate_classifiers(
                ensemble_configuration,
                classifier_kwargs_list,
                train_feature_set,
                train_label_set,
                test_feature_set,
                test_label_set,
                aggregation,
                feature_selection_method,
                visualizers,
                output_file,
                plot_title,
                config_dict["random_seed"],
            )

            roc_auc_val = report[metric]

            val_scores.append(
                {"Metric": metric, "Experiment": experiment_name, "Score": roc_auc_val, "Section": f"Test Set [k={k}]"}
            )

    val_scores = pd.DataFrame.from_records(val_scores)
    val_scores.to_excel(Path(os.environ["ROOT_FOLDER"]).joinpath(experiment_name, f"{plot_title}.xlsx"))

    fig = px.bar(
        val_scores, x="Section", y="Score", color="Experiment", text_auto=True, title=plot_title, barmode="group"
    )
    fig.write_image(Path(os.environ["ROOT_FOLDER"]).joinpath(experiment_name, f"{plot_title}.svg"))


if __name__ == "__main__":
    main()
