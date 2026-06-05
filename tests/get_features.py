import json

features = "Radiodynamics"
features_type = "Flat"
feature_selection = "SFFS"

with open(f"{feature_selection}/Model_Selection/{features}_{features_type}/SFFS/{features_type}/FS/{features}_{features_type}_FS_summary.json", "r") as f:
    summary = json.load(f)
    
models = ["LIST","OF","MODELS"]


feature_idx_list = []
feature_filtering_list = []
feature_class_list = []
feature_name_list = []
for model in models:
    classifier, n_features = model.split('-')[0], model.split('-')[1]
    for fold in range(5):
        features = summary[classifier][str(fold)][n_features]['feature_names']

        for feature in features:
            feature_properties = feature.split('_')
            feature_idx = feature_properties[0]
            feature_filtering = feature_properties[1]
            feature_class = feature_properties[2]
            feature_name = feature_properties[3]
            feature_idx_list.append(feature_idx)
            feature_filtering_list.append(feature_filtering)
            feature_class_list.append(feature_class)
            feature_name_list.append(feature_name)

from collections import Counter

print("Total number of features:", len(feature_idx_list))
print("feature_idx_list occurencies:", Counter(feature_idx_list))
print("feature_filtering_list occurencies:", Counter(feature_filtering_list))
print("feature_class_list occurencies:", Counter(feature_class_list))
print("feature_name_list occurencies:", Counter(feature_name_list))