import json
import pandas as pd

feature_selection_file = "<PATH_TO_FEATURE_SELECTION_FILE>"

with open(feature_selection_file, 'r') as f:
    feature_selection_data = json.load(f)

classifier_best_features = []
for classifier in feature_selection_data:
    for fold in feature_selection_data[classifier]:
        for feature in feature_selection_data[classifier][fold]:
            classifier_best_features.append({"model": f"{classifier}-{feature}", "score": feature_selection_data[classifier][fold][feature]['avg_score']})

df_best_features = pd.DataFrame(classifier_best_features)
df_best_features = df_best_features.sort_values(by="score", ascending=False)

df_best_features.to_csv("best_features.csv", index=False)
top_k = 5

best_features = []
added_models = set()
k = 0
i = 0
while i < top_k and k < len(df_best_features):
    k_best_feature = df_best_features.iloc[k]
    model = k_best_feature["model"].split("-")[0]
    feature = k_best_feature["model"].split("-")[1]
    if model not in added_models:
        best_features.append(model+"-"+feature+"-"+str(k_best_feature["score"]))
        added_models.add(model)
        i += 1
    k += 1
   

print(best_features)