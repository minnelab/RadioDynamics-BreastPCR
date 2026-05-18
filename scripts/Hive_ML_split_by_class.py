#!/usr/bin/env python

import os
import pandas as pd
import ast
import shutil
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent
import importlib
import json
import Hive_ML.configs

DESC = dedent("""
    Script to split the dataset by the given class label.
    """)
EPILOG = dedent("""
    Example call:
    ::
        {filename} --root-dir /path/to/MAMA-MIA
    """.format(filename=Path(__file__).name))


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing MAMA-MIA dataset (should contain 'images', 'segmentations', and clinical_and_imaging_info.xlsx)",
    )
    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Config file containing the configuration for the dataset.",
    )

    return pars


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    root_dir = args.root_dir
    data_dir = os.path.join(root_dir, "images")
    segmentation_dir = os.path.join(root_dir, "segmentations", "expert")
    clinical_data = pd.read_excel(os.path.join(root_dir, "clinical_and_imaging_info.xlsx"))

    try:
        with open(args.config_file) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Hive_ML.configs, args.config_file) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    # Build data dictionary
    data_dict = {}
    for sub in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, sub)):
            data_dict[sub] = {"images": [], "timepoints": []}
            for img in os.listdir(os.path.join(data_dir, sub)):
                if img.endswith(".nii.gz"):
                    data_dict[sub]["images"].append(os.path.join(data_dir, sub, img))

            data_dict[sub]["images"].sort()

            acq_times = clinical_data[clinical_data["patient_id"] == sub]["acquisition_times"].values[0]
            class_label = clinical_data[clinical_data["patient_id"] == sub]["pcr"].values[0]
            if isinstance(acq_times, str):
                data_dict[sub]["timepoints"] = ast.literal_eval(acq_times)
            else:
                data_dict[sub]["timepoints"] = acq_times
            data_dict[sub]["segmentation"] = os.path.join(segmentation_dir, f"{sub}.nii.gz")
            data_dict[sub]["label"] = class_label

    for label in config_dict["label_dict"].keys():
        Path(root_dir).joinpath(config_dict["label_dict"][label]).mkdir(parents=True, exist_ok=True)
    for sub in data_dict.keys():
        label = data_dict[sub]["label"]
        try:
            output_dir = Path(root_dir).joinpath(config_dict["label_dict"][str(int(label))], sub)
        except ValueError:
            continue
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(data_dict[sub]["segmentation"]), output_dir.joinpath(f"{sub}_mask.nii.gz"))
        shutil.copy(
            Path(root_dir).joinpath("4D_images", sub, f"{sub}.nii.gz"), output_dir.joinpath(f"{sub}_image.nii.gz")
        )


if __name__ == "__main__":
    main()
