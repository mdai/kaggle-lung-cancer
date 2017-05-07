"""
Predicts sex of patient in study
"""

import os
import pickle
import numpy as np
import scipy.ndimage
import multiprocessing
from joblib import Parallel, delayed
import argparse
import json

from models.sexdet.sd01a import define_model as define_sd01a
sexdet_model = define_sd01a()


def process_study(study_id, isotropic_volumes_path, out_dir_data):
    isometric_volume = np.load(os.path.join(isotropic_volumes_path, '{}.npy'.format(study_id)))
    mean = np.mean(isometric_volume).astype(np.float32)
    std = np.std(isometric_volume).astype(np.float32)
    volume_resized = scipy.ndimage.interpolation.zoom(isometric_volume,
                                                      np.divide(64, isometric_volume.shape),
                                                      mode='nearest')
    volume_resized = (volume_resized.astype(np.float32) - mean) / (std + 1e-7)
    z0, z1 = volume_resized.shape[0]//2, volume_resized.shape[0]
    y0, y1 = 0, volume_resized.shape[1]//2
    volume_resized = volume_resized[z0:z1, y0:y1, :]
    np.save(os.path.join(out_dir_data, '{}.npy'.format(study_id)), volume_resized)
    return


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

    # load NN weights
    sexdet_model.load_weights(os.path.join(BASEPATH, SETTINGS['MODEL_WEIGHTS_PATH'], 'sd01a.hdf5'))

    isotropic_volumes_path = os.path.join(BASEPATH, 'data_proc/{}/isotropic_volumes_1mm'.format(args.dataset))
    out_dir_data = os.path.join(BASEPATH, 'data_proc/{}/study_sex_det_volumes'.format(args.dataset))
    os.makedirs(out_dir_data, exist_ok=True)

    out_dir_pred = os.path.join(BASEPATH, SETTINGS['PREDICTIONS_PATH'])
    os.makedirs(out_dir_pred, exist_ok=True)
    out_filepath_pred = os.path.join(out_dir_pred, 'study_sex_det.csv')

    with open(out_filepath_pred, 'w') as f:
        f.write('id,sex\n')

    n_jobs = multiprocessing.cpu_count() - 1
    print('# jobs processing in parallel:', n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_study)(study_id, isotropic_volumes_path, out_dir_data) for study_id in study_ids
    )
    print('# processed:', len(results))

    print('running sex det model...')
    volumes_all = np.zeros((len(study_ids), 32, 32, 64, 1), dtype=np.float32)
    for i, study_id in enumerate(study_ids):
        volume = np.load(os.path.join(out_dir_data, '{}.npy'.format(study_id)))
        volumes_all[i, :, :, :, 0] = volume
    sexdet_preds = sexdet_model.predict(volumes_all, batch_size=16)[:, 0]
    with open(out_filepath_pred, 'a') as f:
        for study_id, pred in zip(study_ids, sexdet_preds):
            sex = 'F' if pred > 0.5 else 'M'
            f.write('{},{}\n'.format(study_id, sex))
    print('...done. File written to:\n{}'.format(out_filepath_pred))
