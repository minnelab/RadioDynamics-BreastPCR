import SimpleITK as sitk
import copy
import importlib
import numpy
import numpy as np
from loguru import logger
from pathlib import Path
from scipy.ndimage.morphology import distance_transform_edt
from typing import Union


def generate_distance_map(
    image_filename: Union[str, Path], mask_filename: Union[str, Path], output_filename: Union[str, Path]
):
    """
    Generate a Distance map for a given binary mask.

    Parameters
    ----------
    image_filename :
        Image filename, not used.
    mask_filename :
        Mask filename to compute the distance map.
    output_filename :
        Filename where to save the distance map.
    """
    mask = sitk.ReadImage(mask_filename)
    mask_array = sitk.GetArrayFromImage(mask)
    distance_map = distance_transform_edt(
        mask_array, sampling=(mask.GetSpacing()[0], mask.GetSpacing()[1], mask.GetSpacing()[2])
    )

    distance_map = np.where(mask_array == 0, mask_array, distance_map)
    distance_map_image = sitk.GetImageFromArray(distance_map)
    distance_map_image.CopyInformation(mask)
    sitk.WriteImage(distance_map_image, output_filename)


def generate_distance_map_depth(
    image_filename: Union[str, Path],
    mask_filename: Union[str, Path],
    output_filename: Union[str, Path],
    n_bins: int = 5,
):
    """
    Generate a discrete distance map, equally distributing the distance map values in ``n_bins``, and labelling the voxels
    with discrete values, according to the corresponding bin, from 1 to ``n_bins``.

    Parameters
    ----------
    image_filename :
        Image filename, not used.
    mask_filename :
        Mask filename to compute the discrete distance map.
    output_filename :
        Filename where to save the discrete distance map.
    n_bins :
        Number of splits for the distance map histogram.
    """
    mask = sitk.ReadImage(mask_filename)
    mask_array = sitk.GetArrayFromImage(mask)
    distance_map = distance_transform_edt(
        mask_array, sampling=(mask.GetSpacing()[0], mask.GetSpacing()[1], mask.GetSpacing()[2])
    )

    distance_map = np.where(mask_array == 0, mask_array, distance_map)
    distance_map_copy = distance_map.copy()
    max_depth = np.max(distance_map)

    for idx, depth_range in enumerate(np.arange(0, max_depth, max_depth / n_bins)):
        depth_interval = [depth_range, depth_range + max_depth / n_bins]
        distance_map_copy = np.where(
            (distance_map >= depth_interval[0]) & (distance_map <= depth_interval[1]) & (mask_array != 0),
            idx + 1,
            distance_map_copy,
        )
    distance_map_image = sitk.GetImageFromArray(distance_map_copy)
    distance_map_image.CopyInformation(mask)
    sitk.WriteImage(distance_map_image, output_filename)


def extract_cbf(array: numpy.ndarray, acc: str = "max", timepoints: Union[list, np.ndarray] = None):
    r"""
    Extract 3D CBF map for a given 4D array.

    .. math::
        CBF_{i,j,k} = acc(\frac{\nabla I_{i,j,k,t}}{ \nabla t})

    Parameters
    ----------
    array :
        Input Array
    acc :
        Accumulator to use for extracting gradient. Default: "max" (maximum gradient value).
    timepoints :
        Timepoints array, used to map the CBF values to the timepoints.
    Returns
    -------
        3-D CBF Map
    """
    temporal_axis = np.argmin(array.shape)

    if temporal_axis != 0:
        array = np.transpose(array, (temporal_axis, 0, 1, 2))
    gradient_image = copy.deepcopy(array)

    t_size = gradient_image.shape[0]

    for t in range(1, t_size):
        gradient_image[t, :] = array[t, :] - array[t - 1, :]
        if timepoints is not None:
            gradient_image = gradient_image.astype(float)
            gradient_image[t, :] = gradient_image[t, :] / float(timepoints[t] - timepoints[t - 1])
    gradient_image[0, :] = 0

    m = importlib.import_module("numpy")
    acc_fn = getattr(m, acc)

    cbf_map = acc_fn(gradient_image, axis=0)

    return cbf_map


def generate_cbf_image(
    image_filename: Union[str, Path],
    mask_filename: Union[str, Path],
    output_filename: Union[str, Path],
    timepoints: Union[list, np.ndarray] = None,
):
    r"""
    Generate a 3D CBF map (maximum gradient) from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        CBF_{i,j,k} = \max(\frac{\nabla I_{i,j,k,t}}{ \nabla t})

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel CBF values in the mask region.
    output_filename :
        Filename where to save the CBF map.
    timepoints :
        Timepoints array, used to map the CBF values to the timepoints.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    cbf_map = extract_cbf(image_array, timepoints=timepoints)
    cbf_map = cbf_map * mask_array
    cbf_image = sitk.GetImageFromArray(cbf_map)
    cbf_image.CopyInformation(mask)
    sitk.WriteImage(cbf_image, output_filename)


def generate_ncbf_image(
    image_filename: Union[str, Path], mask_filename: Union[str, Path], output_filename: Union[str, Path]
):
    r"""
    Generate a 3D Negative CBF map (minimum gradient) from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        NCBF_{i,j,k} = \min(\frac{\nabla I_{i,j,k,t}}{ \nabla t})

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel NCBF values in the mask region.
    output_filename :
        Filename where to save the NCBF map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    ncbf_map = extract_cbf(image_array, acc="min")
    ncbf_map = ncbf_map * mask_array
    ncbf_image = sitk.GetImageFromArray(ncbf_map)
    ncbf_image.CopyInformation(mask)
    sitk.WriteImage(ncbf_image, output_filename)


def generate_acbf_image(
    image_filename: Union[str, Path], mask_filename: Union[str, Path], output_filename: Union[str, Path]
):
    r"""
    Generate a 3D Average CBF map (average gradient) from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        ACBF_{i,j,k} = \mu(\frac{\nabla I_{i,j,k,t}}{ \nabla t})

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel ACBF values in the mask region.
    output_filename :
        Filename where to save the ACBF map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    acbf_map = extract_cbf(image_array, acc="mean")

    acbf_map = acbf_map * mask_array
    acbf_image = sitk.GetImageFromArray(acbf_map)
    acbf_image.CopyInformation(mask)
    sitk.WriteImage(acbf_image, output_filename)


def generate_sdcbf_image(
    image_filename: Union[str, Path], mask_filename: Union[str, Path], output_filename: Union[str, Path]
):
    r"""
    Generate a 3D SD CBF map (Standard Deviation gradient) from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        ACBF_{i,j,k} = \sigma(\frac{\nabla I_{i,j,k,t}}{ \nabla t})

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel SDCBF values in the mask region.
    output_filename :
        Filename where to save the SDCBF map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    sdcbf_map = extract_cbf(image_array, acc="std")

    sdcbf_map = sdcbf_map * mask_array
    sdcbf_image = sitk.GetImageFromArray(sdcbf_map)
    sdcbf_image.CopyInformation(mask)
    sitk.WriteImage(sdcbf_image, output_filename)


def extract_mtt(array: numpy.ndarray, acc: str = "max", timepoints: Union[list, np.ndarray] = None):
    r"""
    Extract 3D MTT map for a given 4D array.

    .. math::
        MTT_{i,j,k} = \frac{CBV_{i,j,k}}{CBF_{i,j,k}}

    Parameters
    ----------
    array :
        Input Array
    acc :
        Accumulator to use for extracting gradient for CBF computation. Default: "max" (maximum gradient value).
    timepoints :
        Timepoints array, used to map the MTT values to the timepoints.

    Returns
    -------
        3-D MTT Map
    """
    temporal_axis = np.argmin(array.shape)

    if temporal_axis != 0:
        array = np.transpose(array, (temporal_axis, 0, 1, 2))

    gradient_image = copy.deepcopy(array)
    t_size = gradient_image.shape[0]
    for t in range(1, t_size):
        gradient_image[t, :] = array[t, :] - array[t - 1, :]
        if timepoints is not None:
            gradient_image = gradient_image.astype(float)
            gradient_image[t, :] = gradient_image[t, :] / float(timepoints[t] - timepoints[t - 1])
    gradient_image[0, :] = 0
    m = importlib.import_module("numpy")
    acc_fn = getattr(m, acc)

    cbf_map = acc_fn(gradient_image, axis=0).astype(float)
    cbv_map = np.sum(array, axis=0).astype(float)

    mtt_map = np.divide(cbv_map, cbf_map, out=np.zeros_like(cbv_map), where=cbf_map != 0)

    return mtt_map


def generate_mtt_image(
    image_filename: Union[str, Path],
    mask_filename: Union[str, Path],
    output_filename: Union[str, Path],
    timepoints: Union[list, np.ndarray] = None,
):
    r"""
    Generate a 3D MTT map (Mean Transit Time) from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        MTT_{i,j,k} = \frac{CBV_{i,j,k}}{CBF_{i,j,k}}

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel MTT values in the mask region.
    output_filename :
        Filename where to save the MTT map.
    timepoints :
        Timepoints array, used to map the MTT values to the timepoints.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    mtt_map = extract_mtt(image_array, timepoints=timepoints)
    mtt_map = mtt_map * mask_array
    mtt_image = sitk.GetImageFromArray(mtt_map)
    mtt_image.CopyInformation(mask)
    sitk.WriteImage(mtt_image, output_filename)


def generate_nmtt_image(
    image_filename: Union[str, Path], mask_filename: Union[str, Path], output_filename: Union[str, Path]
):
    r"""
    Generate a 3D NMTT map (Negative Mean Transit Time) from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        NMTT_{i,j,k} = \frac{CBV_{i,j,k}}{NCBF_{i,j,k}}

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel NMTT values in the mask region.
    output_filename :
        Filename where to save the NMTT map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    nmtt_map = extract_mtt(image_array, acc="min")

    nmtt_map = nmtt_map * mask_array
    nmtt_image = sitk.GetImageFromArray(nmtt_map)
    nmtt_image.CopyInformation(mask)
    sitk.WriteImage(nmtt_image, output_filename)


def generate_amtt_image(
    image_filename: Union[str, Path], mask_filename: Union[str, Path], output_filename: Union[str, Path]
):
    r"""
    Generate a 3D AMTT map (Average Mean Transit Time) from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        AMTT_{i,j,k} = \frac{CBV_{i,j,k}}{ACBF_{i,j,k}}

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel AMTT values in the mask region.
    output_filename :
        Filename where to save the AMTT map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    amtt_map = extract_mtt(image_array, acc="mean")

    amtt_map = amtt_map * mask_array
    amtt_image = sitk.GetImageFromArray(amtt_map)
    amtt_image.CopyInformation(mask)
    sitk.WriteImage(amtt_image, output_filename)


def generate_sdmtt_image(
    image_filename: Union[str, Path], mask_filename: Union[str, Path], output_filename: Union[str, Path]
):
    r"""
    Generate a 3D SDMTT map (Standard Deviation Mean Transit Time) from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        SDMTT_{i,j,k} = \frac{CBV_{i,j,k}}{SDCBF_{i,j,k}}

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel SDMTT values in the mask region.
    output_filename :
        Filename where to save the SDMTT map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    sdmtt_map = extract_mtt(image_array, acc="std")

    sdmtt_map = sdmtt_map * mask_array
    sdmtt_image = sitk.GetImageFromArray(sdmtt_map)
    sdmtt_image.CopyInformation(mask)
    sitk.WriteImage(sdmtt_image, output_filename)


def extract_cbv(array: numpy.ndarray):
    r"""
    Extract 3D CBV map for a given 4D array.

    .. math::
        CBV_{i,j,k} = \sum_{t=0}^{N} I_{i,j,k,t}

    Parameters
    ----------
    array :
        Input Array

    Returns
    -------
        3-D CBV Map
    """
    temporal_axis = np.argmin(array.shape)
    cbv_map = np.sum(array, axis=temporal_axis)

    return cbv_map


def generate_cbv_image(
    image_filename: Union[str, Path],
    mask_filename: Union[str, Path],
    output_filename: Union[str, Path],
    timepoints: Union[list, np.ndarray] = None,
):
    r"""
    Generate a 3D CBV map from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        CBV_{i,j,k} = \sum_{t=0}^{N} I_{i,j,k,t}

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel CBV values in the mask region.
    output_filename :
        Filename where to save the CBV map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    cbv_map = extract_cbv(image_array)
    cbv_map = cbv_map * mask_array
    cbv_image = sitk.GetImageFromArray(cbv_map)
    cbv_image.CopyInformation(mask)
    sitk.WriteImage(cbv_image, output_filename)


def extract_ttp(array: numpy.ndarray, timepoints: Union[list, np.ndarray] = None):
    r"""
    Extract 3D TTP (Time-to-Peak) map for a given 4D array.

    .. math::
        TTP_{i,j,k} = argmax(I_{i,j,k,t})

    Parameters
    ----------
    array :
        Input Array
    timepoints :
        Timepoints array, used to map the TTP values to the timepoints.

    Returns
    -------
        3-D TTP Map
    """
    temporal_axis = np.argmin(array.shape)

    m = importlib.import_module("numpy")
    acc_fn = getattr(m, "argmax")

    ttp_array = acc_fn(array, axis=temporal_axis).astype(np.uint8)
    if timepoints is not None:
        ttp_array = np.array(timepoints)[ttp_array]
    ttp_array = ttp_array + 1

    if timepoints is None:
        ttp_array = np.where(ttp_array > array.shape[temporal_axis], array.shape[temporal_axis], ttp_array)

    return ttp_array


def generate_ttp_image(
    image_filename: Union[str, Path],
    mask_filename: Union[str, Path],
    output_filename: Union[str, Path],
    timepoints: Union[list, np.ndarray] = None,
):
    r"""
    Generate a 3D TTP map from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        TTP_{i,j,k} = argmax(I_{i,j,k,t})

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel TTP values in the mask region.
    output_filename :
        Filename where to save the TTP map.
    timepoints :
        Timepoints array, used to map the TTP values to the timepoints.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)

    mask_array = sitk.GetArrayFromImage(mask)

    ttp_array = extract_ttp(sitk.GetArrayFromImage(image), timepoints=timepoints)
    ttp_array = np.where(mask_array == 0, mask_array, ttp_array)

    ttp_image = sitk.GetImageFromArray(ttp_array)
    ttp_image.CopyInformation(mask)
    sitk.WriteImage(ttp_image, output_filename)


def extract_ttcbf(array: numpy.ndarray, acc: str = "argmax"):
    r"""
    Extract 3D TTCBF (Time-to-CBF) map for a given 4D array.

    .. math::
        TTCBF_{i,j,k} = acc(\frac{\nabla I_{i,j,k,t}}{ \nabla t})

    Parameters
    ----------
    array :
        Input Array
    acc :
        Accumulator to use for extracting gradient for CBF computation. Default: "argmax" (maximum gradient index).

    Returns
    -------
        3-D TTCBF Map
    """
    temporal_axis = np.argmin(array.shape)

    if temporal_axis != 0:
        array = np.transpose(array, (temporal_axis, 0, 1, 2))
    gradient_image = copy.deepcopy(array)

    t_size = gradient_image.shape[0]

    for t in range(1, t_size):
        gradient_image[t, :] = array[t, :] - array[t - 1, :]
    gradient_image[0, :] = 0

    m = importlib.import_module("numpy")
    acc_fn = getattr(m, acc)

    ttcbf_array = acc_fn(gradient_image, axis=0).astype(np.uint8)

    ttcbf_array = ttcbf_array + 1
    ttcbf_array = np.where(ttcbf_array > array.shape[temporal_axis], array.shape[temporal_axis], ttcbf_array)

    return ttcbf_array


def generate_ttcbf_image(
    image_filename: Union[str, Path], mask_filename: Union[str, Path], output_filename: Union[str, Path]
):
    r"""
    Generate a 3D TTCBF map from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        TTCBF_{i,j,k} = argmax(\frac{\nabla I_{i,j,k,t}}{ \nabla t})

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel TTCBF values in the mask region.
    output_filename :
        Filename where to save the TTCBF map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)

    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    ttcbf_array = extract_ttcbf(image_array)
    ttcbf_array = np.where(mask_array == 0, mask_array, ttcbf_array)

    ttcbf_image = sitk.GetImageFromArray(ttcbf_array)
    ttcbf_image.CopyInformation(mask)
    sitk.WriteImage(ttcbf_image, output_filename)


def generate_ttncbf_image(image_filename, mask_filename, output_filename):
    r"""
    Generate a 3D TTNCBF (Time-to-NCBF) map from a given 4D Volume, preserving only the values in the mask region.

    .. math::
        TTNCBF_{i,j,k} = argmin(\frac{\nabla I_{i,j,k,t}}{ \nabla t})

    Parameters
    ----------
    image_filename :
        Image filename, containing the 4D array.
    mask_filename :
        Mask filename, used to filter the voxel TTNCBF values in the mask region.
    output_filename :
        Filename where to save the TTNCBF map.
    """
    image = sitk.ReadImage(image_filename)
    mask = sitk.ReadImage(mask_filename)

    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    ttncbf_array = extract_ttcbf(image_array, acc="argmin")

    ttncbf_array = np.where(mask_array == 0, mask_array, ttncbf_array)

    ttncbf_image = sitk.GetImageFromArray(ttncbf_array)
    ttncbf_image.CopyInformation(mask)
    sitk.WriteImage(ttncbf_image, output_filename)


PERFUSION_FUNCTIONS = {
    "distance_map": generate_distance_map,
    "distance_map_depth": generate_distance_map_depth,
    "ttp": generate_ttp_image,
    "cbv": generate_cbv_image,
    "cbf": generate_cbf_image,
    "ncbf": generate_ncbf_image,
    "acbf": generate_acbf_image,
    "sdcbf": generate_sdcbf_image,
    "mtt": generate_mtt_image,
    "nmtt": generate_nmtt_image,
    "amtt": generate_amtt_image,
    "sdmtt": generate_sdmtt_image,
    "ttcbf": generate_ttcbf_image,
    "ttncbf": generate_ttncbf_image,
    "ttp_array": extract_ttp,
    "cbv_array": extract_cbv,
    "cbf_array": extract_cbf,
    "mtt_array": extract_mtt,
}
