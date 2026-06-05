import json
import numpy as np
from pathlib import Path

DATA_FOLDER = "<PATH_TO_MAMA-MIA_DATA_FOLDER>"
with open(Path(DATA_FOLDER).joinpath("Classes", "extracted_histograms.json"), "r") as f:
    extracted_histograms = json.load(f)

with open(Path(DATA_FOLDER).joinpath("Classes", "subjects_to_skip.json"), "r") as f:
    subjects_to_skip = json.load(f)



histograms = {"min_ttp": [], "max_ttp": [], "min_cbv": [], "max_cbv": [], "min_cbf": [], "max_cbf": [], "min_mtt": [], "max_mtt": []}

for key, value in extracted_histograms.items():
    min_value, max_value = value
    if key.endswith("_ttp_map.nii.gz"):
        histograms["min_ttp"].append(min_value)
        histograms["max_ttp"].append(max_value)
    elif key.endswith("_cbv_map.nii.gz"):
        histograms["min_cbv"].append(min_value)
        histograms["max_cbv"].append(max_value)
    elif key.endswith("_cbf_map.nii.gz"):
        histograms["min_cbf"].append(min_value)
        histograms["max_cbf"].append(max_value)
    elif key.endswith("_mtt_map.nii.gz"):
        histograms["min_mtt"].append(min_value)
        histograms["max_mtt"].append(max_value)

for key, value in histograms.items():
    print(key)
    print(np.percentile(value, 95))