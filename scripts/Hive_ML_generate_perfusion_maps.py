#!/usr/bin/env python

import importlib.resources
import json
import os
import pandas as pd
import ast
from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, DEBUG
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent
from tqdm import tqdm

import Hive_ML.configs
from Hive_ML.feature_generation.perfusion_features import PERFUSION_FUNCTIONS

DESC = dedent("""
    Script to generate Perfusion Maps for a given dataset. The Perfusion Maps to create, and their correpsonding suffix files, are specified in the
    ``config-file``
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
        help="Dataset folder",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration file path, containing training and processing parameters.",
    )

    pars.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=None,
        help="Number of parallel threads to use when generating the Perfusion Maps (Default: ``N_THREADS``).",
    )
    pars.add_argument(
        "--clinical-data-file",
        type=str,
        required=False,
        default=None,
        help="Clinical data file path, containing the acquisition times for each subject.",
    )
    pars.add_argument(
        "--timepoints-column",
        type=str,
        required=False,
        default="acquisition_times",
        help="Column name in the clinical data file containing the acquisition times for each subject.",
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

    perfusion_maps_dict = config_dict["perfusion_maps"]
    labels = subfolders(arguments["data_folder"], join=False)

    n_workers = "1"
    if arguments["n_workers"] is None:
        if "N_THREADS" in os.environ is not None:
            n_workers = str(os.environ["N_THREADS"])
    else:
        n_workers = str(arguments["n_workers"])

    if arguments["clinical_data_file"] is not None:
        df = pd.read_excel(arguments["clinical_data_file"])
    else:
        df = None

    pool = Pool(int(n_workers))
    perfusion_maps = []
    for label in labels:
        subjects = subfolders(Path(arguments["data_folder"]).joinpath(label), join=False)
        for subject in subjects:
            logger.log(DEBUG, "Processing {}".format(subject))
            for perfusion_map in perfusion_maps_dict:
                if type(perfusion_maps_dict[perfusion_map]) is dict:
                    map_suffix = perfusion_maps_dict[perfusion_map]["suffix"]
                    kwargs = perfusion_maps_dict[perfusion_map]["kwargs"]
                else:
                    map_suffix = perfusion_maps_dict[perfusion_map]
                    kwargs = {}
                if df is not None:
                    timepoints = df[df["patient_id"] == subject][arguments["timepoints_column"]].values[0]
                    if isinstance(timepoints, str):
                        timepoints = ast.literal_eval(timepoints)
                    kwargs = [timepoints]
                if Path(arguments["data_folder"]).joinpath(label, subject, subject + map_suffix).exists():
                    continue
                # Check if timepoints are a list of ints before continuing
                elif isinstance(kwargs[0], list) and all(isinstance(x, int) for x in kwargs[0]):
                    logger.log(DEBUG, "Creating Perfusion Map {} for {}".format(map_suffix, subject))
                    perfusion_maps.append(
                        pool.starmap_async(
                            PERFUSION_FUNCTIONS[perfusion_map],
                            (
                                (
                                    str(
                                        Path(arguments["data_folder"]).joinpath(
                                            label, subject, subject + config_dict["image_suffix"]
                                        )
                                    ),
                                    str(
                                        Path(arguments["data_folder"]).joinpath(
                                            label, subject, subject + config_dict["mask_suffix"]
                                        )
                                    ),
                                    str(Path(arguments["data_folder"]).joinpath(label, subject, subject + map_suffix)),
                                    *kwargs,
                                ),
                            ),
                        )
                    )

    for res in tqdm(perfusion_maps, desc="Perfusion Maps Creation"):
        _ = res.get()


if __name__ == "__main__":
    main()
