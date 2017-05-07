import os
import pickle
import random
import time
import numpy as np
from scipy.ndimage.interpolation import affine_transform
from skimage.transform import resize
from skimage.util import pad, crop
import scipy.ndimage
import multiprocessing
from joblib import Parallel, delayed
import argparse
from uuid import uuid4
import itertools
import json


def process_study(study_id, out_dir):
    isometric_volume = np.load('../data_proc/stage1/isotropic_volumes_1mm/{}.npy'.format(study_id))
    mean = np.mean(isometric_volume).astype(np.float32)
    std = np.std(isometric_volume).astype(np.float32)
    volume_resized = scipy.ndimage.interpolation.zoom(isometric_volume,
                                                      np.divide(64, isometric_volume.shape),
                                                      mode='nearest')
    volume_resized = (volume_resized.astype(np.float32) - mean) / (std + 1e-7)
    for i in range(7):
        z_shift = random.randint(0, 5)
        z0 = (volume_resized.shape[0]//2) - z_shift
        z1 = volume_resized.shape[0] - z_shift
        y_shift = random.randint(0, 5)
        y0 = y_shift
        y1 = (volume_resized.shape[1]//2) + y_shift
        volume_resized_sample = volume_resized[z0:z1, y0:y1, :]
        volume_resized_sample = np.expand_dims(volume_resized_sample, axis=3)
        out_filepath = os.path.join(out_dir, '{}.npy'.format(uuid4()))
        np.save(out_filepath, volume_resized_sample)
    return


if __name__ == '__main__':
    SETTINGS_FILE_PATH = '../SETTINGS.json'
    BASEPATH = os.path.dirname(os.path.abspath(SETTINGS_FILE_PATH))

    with open(SETTINGS_FILE_PATH, 'r') as f:
        SETTINGS = json.load(f)

    out_dir = os.path.join(BASEPATH, 'data_train/stage1/sex_det/volumes_1')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'train', '0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'train', '1'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val', '0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val', '1'), exist_ok=True)

    studies_train = pd.read_csv(os.path.join(BASEPATH, SETTINGS['STAGE1_LABELS_FILE_PATH'])).sort_values(by='id')['id'].tolist()

    with open(os.path.join(BASEPATH, SETTINGS['ANNOTATIONS_PATH'], 'sex_labels_train_dict.pkl'), 'rb') as f:
        sex_labels_train_dict = pickle.load(f)

    study_ids = sorted(list(studies_train))
    print('# study ids:', len(study_ids))
    random.seed(46)
    study_ids_train_set = set(random.sample(study_ids, int(len(study_ids) * 0.9)))
    out_subdirs = {}
    for study_id in study_ids:
        # 0 = M, 1 = F
        if (study_id in study_ids_train_set) and (sex_labels_train_dict[study_id] == 'M'):
            out_subdirs[study_id] = os.path.join(out_dir, 'train', '0')
        elif (study_id in study_ids_train_set) and (sex_labels_train_dict[study_id] == 'F'):
            out_subdirs[study_id] = os.path.join(out_dir, 'train', '1')
        elif (study_id not in study_ids_train_set) and (sex_labels_train_dict[study_id] == 'M'):
            out_subdirs[study_id] = os.path.join(out_dir, 'val', '0')
        elif (study_id not in study_ids_train_set) and (sex_labels_train_dict[study_id] == 'F'):
            out_subdirs[study_id] = os.path.join(out_dir, 'val', '1')

    n_jobs = multiprocessing.cpu_count() - 1
    print('# jobs processing in parallel:', n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_study)(study_id, out_subdirs[study_id]) for study_id in study_ids
    )
    print('# processed:', len(results))
