#!/usr/bin/env python

import os
import pandas as pd
import ast
import SimpleITK as sitk
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

DESC = dedent(
    """
    Script to generate 4D volumes from 3D images for MAMA-MIA dataset.
    Combines multiple 3D images (timepoints) into a single 4D volume for each subject.
    """
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --root-dir /path/to/MAMA-MIA
    """.format(
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing MAMA-MIA dataset (should contain 'images', 'segmentations', and clinical_and_imaging_info.xlsx)",
    )

    return pars


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    root_dir = args.root_dir
    data_dir = os.path.join(root_dir, "images")
    output_dir = os.path.join(root_dir, "4D_images")
    segmentation_dir = os.path.join(root_dir, "segmentations", "expert")
    clinical_data = pd.read_excel(os.path.join(root_dir, "clinical_and_imaging_info.xlsx"))

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
            if isinstance(acq_times, str):
                data_dict[sub]["timepoints"] = ast.literal_eval(acq_times)
            else:
                data_dict[sub]["timepoints"] = acq_times
            data_dict[sub]["segmentation"] = os.path.join(segmentation_dir, sub, f"{sub}.nii.gz")

    # Generate 4D volumes
    for sub in data_dict.keys():
        output_path = os.path.join(output_dir, sub, f"{sub}.nii.gz")
        if not os.path.exists(output_path):
            image_list = []
            for img in data_dict[sub]["images"]:
                sitk_img = sitk.ReadImage(img)
                image_list.append(sitk_img)
            sitk_4D_image = sitk.JoinSeries(image_list)
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
            sitk.WriteImage(sitk_4D_image, output_path)
            print(f"Created 4D volume for {sub}: {output_path}")
        else:
            print(f"4D volume already exists for {sub}: {output_path}")


if __name__ == "__main__":
    main()
