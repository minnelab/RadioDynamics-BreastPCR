import pandas as pd
import os
from pathlib import Path

experiment_name = "MAMAMIA_Radiodynamics-TrainTestSplit-1Fold"
feature_selection_method = "SFFS"
aggregation = "Flat"
metric = "roc_auc"
reduction = "mean"
k = 5
ROOT_FOLDER = "<PATH_TO_EXPERIMENTS_FOLDER>"
df_summary = pd.read_excel(Path(ROOT_FOLDER).joinpath(experiment_name, f"{experiment_name}_{feature_selection_method}_{aggregation}.xlsx"))

aggr = df_summary[df_summary["Metric"] == metric][["Value", "Classifier"]].groupby(["Classifier"]).agg(reduction)

print(aggr)
aggr = aggr.loc[aggr["Value"].nlargest(k).index]
classifiers = aggr.index.values

n_features = []
best_val_scores = []
for classifier in classifiers:
    aggr = df_summary[(df_summary["Metric"] == metric) & (df_summary["Classifier"] == classifier)][
        ["Value", "N_Features"]].groupby(["N_Features"]).agg(
        reduction)
    aggr = aggr.loc[aggr["Value"].nlargest(1).index]
    n_features.append(aggr.index.values[0])
    best_val_scores.append(aggr.values[0][0])

n_features_selected_classifier = [(n_features[i], classifiers[i]) for i in range(len(classifiers))]

for i in range(k):
    n_features, selected_classifier = n_features_selected_classifier[i]
    print(f"Best Configuration: {selected_classifier}-{n_features}, {metric}: {best_val_scores[i]}")