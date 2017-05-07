"""
Create isotropic volumes from directory of DICOM studies
"""

import os
import pydicom
import pickle
import numpy as np
import scipy.ndimage
import multiprocessing
from joblib import Parallel, delayed
import argparse
import json


def get_files(root):
    """Yields all file paths recursively from root filepath.
    """
    for item in os.scandir(root):
        if item.is_file():
            yield item.path
        elif item.is_dir():
            yield from get_files(item.path)


def load_study(instance_filepaths):
    """Loads a study with pydicom and sorts slices in z-axis.
    Calculates slice thickness and writes it in the read dicom file.
    """
    slices = [pydicom.read_file(fp) for fp in instance_filepaths]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    if slice_thickness == 0:
        for i in range(1, len(slices) - 2):
            try:
                slice_thickness = np.abs(slices[i].ImagePositionPatient[2] - slices[i+1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(slices[i].SliceLocation - slices[i+1].SliceLocation)
            if slice_thickness > 0:
                break

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def to_volume(slices):
    """Creates ndarray volume in Hounsfield units (HU) from array of pydicom slices.
    """
    volume = np.stack([s.pixel_array for s in slices])
    volume = volume.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    volume[volume == -2000] = 0

    # Convert to Hounsfield units (HU)
    for n in range(len(slices)):
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope
        if slope != 1:
            volume[n] = slope * volume[n].astype(np.float64)
            volume[n] = volume[n].astype(np.int16)
        volume[n] += np.int16(intercept)

    volume = np.array(volume, dtype=np.int16)
    spacing = tuple(map(float, ([slices[0].SliceThickness] + slices[0].PixelSpacing)))
    return volume, spacing


def isotropic_resampling(volume, slices, new_spacing=1.0):
    """Resamples volume (z,y,x) with isotropic spacing.
    """
    spacing = tuple(map(float, ([slices[0].SliceThickness] + slices[0].PixelSpacing)))

    resize_factor = np.array(spacing) / ([new_spacing] * 3)
    new_real_shape = volume.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / volume.shape

    volume = scipy.ndimage.interpolation.zoom(volume, real_resize_factor, mode='nearest')

    new_spacing_actual = tuple(np.array(spacing) / real_resize_factor)
    return volume, new_spacing_actual


def process_study(study_id, data_dir, out_dir, new_spacing=1):
    study_root_path = os.path.join(data_dir, study_id)
    instance_filepaths = sorted(list(get_files(study_root_path)))
    slices = load_study(instance_filepaths)
    volume, spacing = to_volume(slices)
    volume_resampled, spacing_resampled = isotropic_resampling(volume, slices, new_spacing)

    out_filepath = os.path.join(out_dir, '{}.npy'.format(study_id))
    np.save(out_filepath, volume_resampled)

    print(study_id)
    return volume.shape, spacing, volume_resampled.shape, spacing_resampled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['stage1', 'stage2', 'sample'], default='sample')
    args = parser.parse_args()

    SETTINGS_FILE_PATH = '../SETTINGS.json'
    BASEPATH = os.path.dirname(os.path.abspath(SETTINGS_FILE_PATH))

    with open(SETTINGS_FILE_PATH, 'r') as f:
        SETTINGS = json.load(f)

    if args.dataset == 'stage1':
        data_dir = os.path.join(BASEPATH, SETTINGS['STAGE1_DATA_PATH'])
    elif args.dataset == 'stage2':
        data_dir = os.path.join(BASEPATH, SETTINGS['STAGE2_DATA_PATH'])
    elif args.dataset == 'sample':
        data_dir = os.path.join(BASEPATH, SETTINGS['SAMPLE_DATA_PATH'])

    study_ids = sorted(os.listdir(data_dir))
    print('# study ids:', len(study_ids))

    out_dir = os.path.join(BASEPATH, 'data_proc/{}/isotropic_volumes_1mm'.format(args.dataset))
    os.makedirs(out_dir, exist_ok=True)
    metadata_filepath = os.path.join(BASEPATH, 'data_proc/{}/isotropic_volumes_1mm.pkl'.format(args.dataset))

    n_jobs = multiprocessing.cpu_count() - 1
    print('# jobs processing in parallel:', n_jobs)
    print('')
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_study)(study_id, data_dir, out_dir) for study_id in study_ids
    )
    print('')
    print('# processed:', len(results))

    metadata = {}
    for i, (volume_shape, spacing, volume_resampled_shape, spacing_resampled) in enumerate(results):
        metadata[study_ids[i]] = {
            'volume_shape': volume_shape,
            'spacing': spacing,
            'volume_resampled_shape': volume_resampled_shape,
            'spacing_resampled': spacing_resampled
        }

    print('saving metadata file to:', metadata_filepath)
    with open(metadata_filepath, 'wb') as f:
        pickle.dump(metadata, f)
    print('done.')
