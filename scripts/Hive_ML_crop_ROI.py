#!/usr/bin/env python


import datetime
import json

from argparse import ArgumentParser, RawTextHelpFormatter

from importlib.resources import as_file, files
from pathlib import Path
from textwrap import dedent

from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args

import SimpleITK as sitk

import Hive_ML.configs
from tqdm import tqdm
from logging import WARNING

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent("""
    Script to crop ROI for a specified dataset. The images and masks used to crop the ROI are specified in the
    ``config-file``.
    """)  # noqa: E501
EPILOG = dedent("""
    Example call:
    ::
        {filename} -i /path/to/data_folder --config-file config_file.json
    """.format(filename=Path(__file__).name))  # noqa: E501


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "-i",
        "--data-folder",
        type=str,
        required=True,
        help="Input Dataset folder",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration file path, containing training and processing parameters.",
    )

    add_verbosity_options_to_argparser(pars)

    return pars


def crop_ROI(image_path, mask_path, output_path, output_mask_path, logger):
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
    except Exception as e:
        logger.error(f"Error reading image or mask: {e}")
        return
    # FIX: Cast the mask to an integer type
    # LabelShapeStatisticsImageFilter does not support float32
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    logger.info(f"Image size: {image.GetSize()}")

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)

    roi_label = 1
    if not label_shape_filter.HasLabel(roi_label):
        logger.error("ROI label not found in the mask.")
        return

    bbox = label_shape_filter.GetBoundingBox(roi_label)

    # Extract the ROI
    # bbox is (start_x, start_y, start_z, size_x, size_y, size_z)
    start_index = bbox[:3]
    size = bbox[3:]

    cropped_image = sitk.RegionOfInterest(image, size, start_index)

    if output_path:
        sitk.WriteImage(cropped_image, output_path)

    cropped_mask = sitk.RegionOfInterest(mask, size, start_index)
    if output_mask_path:
        sitk.WriteImage(cropped_mask, output_mask_path)

    logger.info(f"Cropped image size: {cropped_image.GetSize()}")


def extract_histogram(image_path, logger):
    try:
        image = sitk.ReadImage(image_path)
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return None
    max_value = float(sitk.GetArrayFromImage(image).max())
    min_value = float(sitk.GetArrayFromImage(image).min())

    return min_value, max_value


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )
    logger.setLevel(WARNING)
    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with as_file(files(Hive_ML.configs).joinpath(arguments["config_file"])) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    image_suffix = config_dict["image_suffix"]
    mask_suffix = config_dict["mask_suffix"]

    subjects_to_skip = []
    labels = subfolders(arguments["data_folder"], join=False)
    extracted_histograms = {}
    subjects_to_skip = []
    for label in labels:
        subjects = subfolders(Path(arguments["data_folder"]).joinpath(label), join=False)
        for subject in tqdm(subjects):
            if subject in subjects_to_skip:
                continue
            if type(image_suffix) is list:
                image_list = [
                    str(Path(arguments["data_folder"]).joinpath(label, subject, subject + single_image_suffix))
                    for single_image_suffix in image_suffix
                ]
                for image in image_list:
                    logger.info(f"Cropping {image}")
                    output_path = image[: -len(".nii.gz")] + "_cropped.nii.gz"
                    logger.info(f"Output path: {output_path}")
                    output_mask_path = (
                        str(Path(arguments["data_folder"]).joinpath(label, subject, subject + mask_suffix))[
                            : -len(".nii.gz")
                        ]
                        + "_cropped_mask.nii.gz"
                    )
                    extracted_histogram = extract_histogram(image, logger)
                    if extracted_histogram is None:
                        subjects_to_skip.append(subject)
                        continue
                    extracted_histograms[f"{label}_{subject}_{image}"] = extracted_histogram

                    if Path(output_path).is_file() and Path(output_mask_path).is_file():
                        continue
                    crop_ROI(
                        image,
                        str(Path(arguments["data_folder"]).joinpath(label, subject, subject + mask_suffix)),
                        output_path,
                        output_mask_path,
                        logger,
                    )
                logger.info(f"Cropped {len(image_list)} images for {label} {subject}")
            else:
                image_path = str(Path(arguments["data_folder"]).joinpath(label, subject, subject + image_suffix))
                mask_path = str(Path(arguments["data_folder"]).joinpath(label, subject, subject + mask_suffix))
                logger.info(f"Cropping {image_path}")
                output_path = image_path[: -len(".nii.gz")] + "_cropped.nii.gz"
                logger.info(f"Output path: {output_path}")
                logger.info(f"Cropping {mask_path}")
                output_mask_path = mask_path[: -len(".nii.gz")] + "_cropped_mask.nii.gz"
                extracted_histogram = extract_histogram(image_path, logger)
                if extracted_histogram is None:
                    subjects_to_skip.append(subject)
                    continue
                extracted_histograms[f"{label}_{subject}_{image_path}"] = extracted_histogram
                if Path(output_path).is_file() and Path(output_mask_path).is_file():
                    continue
                crop_ROI(image_path, mask_path, output_path, output_mask_path, logger)
                logger.info(f"Cropped {label} {subject}")
    with open(Path(arguments["data_folder"]).joinpath("extracted_histograms.json"), "w") as f:
        json.dump(extracted_histograms, f)
    with open(Path(arguments["data_folder"]).joinpath("subjects_to_skip.json"), "w") as f:
        json.dump(subjects_to_skip, f)


if __name__ == "__main__":
    main()
