"""
Create 3D ROI probability maps
"""

import os
import pickle
import numpy as np
import time
import argparse
import json

from models.nodule.m05a import define_model as define_m05a
from models.nodule.m09a import define_model as define_m09a
model_keys = ['m05a', 'm09a']
models = {}
for n, key in enumerate(model_keys):
    models[key] = locals()['define_{}'.format(key)]()


def process_study(study_id, isotropic_volumes_path, out_dir, config):
    out_filepath = os.path.join(out_dir, '{}.npy'.format(study_id))
    print(study_id, end=' ...', flush=True)
    start_time = time.time()

    initial_probmap_threshold, coarse_stride, fine_stride = config

    isometric_volume = np.load(os.path.join(isotropic_volumes_path, '{}.npy'.format(study_id)))
    mean = np.mean(isometric_volume).astype(np.float32)
    std = np.std(isometric_volume).astype(np.float32)

    batch_size = 512
    patchsize = 64

    # Coarse probmap

    stride = coarse_stride
    preds = {}
    for n, key in enumerate(model_keys):
        patches_arr = []
        for i in range(0, isometric_volume.shape[0] - 5):
            for j in range(0, isometric_volume.shape[1] - patchsize, stride):
                for k in range(0, isometric_volume.shape[2] - patchsize, stride):
                    if key in ['m05a']:
                        patch = np.moveaxis(isometric_volume[i:i+3, j:j+patchsize, k:k+patchsize], 0, 2)
                    elif key in ['m09a']:
                        patch = np.moveaxis(isometric_volume[i:i+5, j:j+patchsize, k:k+patchsize], 0, 2)
                    else:
                        raise
                    patches_arr.append(patch)
        patches_arr = np.array(patches_arr, dtype=np.float32)
        patches_arr -= mean
        patches_arr /= (std + 1e-7)
        preds[key] = models[key].predict(patches_arr, batch_size=batch_size)

    probmap_ensemb = np.zeros(isometric_volume.shape)
    overlap_ensemb = np.zeros(isometric_volume.shape)
    for n, key in enumerate(model_keys):
        patch_index = 0
        for i in range(0, isometric_volume.shape[0] - 5):
            for j in range(0, isometric_volume.shape[1] - patchsize, stride):
                for k in range(0, isometric_volume.shape[2] - patchsize, stride):
                    j0_inner = j + patchsize // 4
                    k0_inner = k + patchsize // 4
                    j1_inner = j0_inner + patchsize // 2
                    k1_inner = k0_inner + patchsize // 2
                    probmap_ensemb[i, j:j+patchsize, k:k+patchsize] += preds[key][patch_index, 0]
                    overlap_ensemb[i, j:j+patchsize, k:k+patchsize] += 1
                    patch_index += 1
    overlap_ensemb[np.where(overlap_ensemb == 0)] = 1
    probmap_ensemb /= overlap_ensemb

    initial_probmap = probmap_ensemb
    initial_probmap_thresholded = initial_probmap > (initial_probmap_threshold / 100)

    # Fine probmap

    stride = fine_stride
    preds = {}
    for n, key in enumerate(model_keys):
        patches_arr = []
        for i in range(0, isometric_volume.shape[0] - 5):
            for j in range(0, isometric_volume.shape[1] - patchsize, stride):
                for k in range(0, isometric_volume.shape[2] - patchsize, stride):
                    if np.any(initial_probmap_thresholded[i, j:j+patchsize, k:k+patchsize]):
                        if key in ['m05a']:
                            patch = np.moveaxis(isometric_volume[i:i+3, j:j+patchsize, k:k+patchsize], 0, 2)
                        elif key in ['m09a']:
                            patch = np.moveaxis(isometric_volume[i:i+5, j:j+patchsize, k:k+patchsize], 0, 2)
                        else:
                            raise
                        patches_arr.append(patch)
        if not patches_arr:
            # initial_probmap_thresholded is empty, save zero array and return
            np.save(out_filepath, np.zeros(isometric_volume.shape))
            print('{} s'.format(time.time() - start_time))
            return
        patches_arr = np.array(patches_arr, dtype=np.float32)
        patches_arr -= mean
        patches_arr /= (std + 1e-7)
        preds[key] = models[key].predict(patches_arr, batch_size=batch_size)

    # Create ensemble probmap

    probmap_ensemb = np.zeros(isometric_volume.shape)
    overlap_ensemb = np.zeros(isometric_volume.shape)

    for n, key in enumerate(model_keys):
        patch_index = 0
        for i in range(0, isometric_volume.shape[0] - 5):
            for j in range(0, isometric_volume.shape[1] - patchsize, stride):
                for k in range(0, isometric_volume.shape[2] - patchsize, stride):
                    if np.any(initial_probmap_thresholded[i, j:j+patchsize, k:k+patchsize]):
                        j0_inner = j + patchsize // 4
                        k0_inner = k + patchsize // 4
                        j1_inner = j0_inner + patchsize // 2
                        k1_inner = k0_inner + patchsize // 2
                        probmap_ensemb[i, j0_inner:j1_inner, k0_inner:k1_inner] += preds[key][patch_index, 0]
                        overlap_ensemb[i, j0_inner:j1_inner, k0_inner:k1_inner] += 1
                        patch_index += 1

    overlap_ensemb[np.where(overlap_ensemb == 0)] = 1
    probmap_ensemb /= overlap_ensemb

    np.save(out_filepath, probmap_ensemb)
    print('{} s'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--group', type=int, default=1)
    parser.add_argument('--dataset', choices=['stage1', 'stage2', 'sample'], default='sample')
    parser.add_argument('--config', choices=['50_16_2', '30_16_4'], default='50_16_2')
    args = parser.parse_args()

    if args.split < 1:
        raise ValueError('--split must be an integer 1 or greater')
    if args.group < 1 or args.group > args.split:
        raise ValueError('specified --group invalid given --split value of {}'.format(args.split))

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

    groups = [g.tolist() for g in np.array_split(study_ids, args.split)]
    print('split into {} groups'.format(args.split))
    print('running group {}.'.format(args.group))

    for key in model_keys:
        models[key].load_weights(os.path.join(BASEPATH, SETTINGS['MODEL_WEIGHTS_PATH'], '{}.hdf5'.format(key)))

    if args.config == '50_16_2':
        config = (50, 16, 2)
    elif args.config == '30_16_4':
        config = (30, 16, 4)
    else:
        raise ValueError('Invalid --config')
    print('config:', config)

    out_dir = os.path.join(BASEPATH, 'data_proc/{}/nodule_detect_probmaps_{}'.format(args.dataset, args.config))
    os.makedirs(out_dir, exist_ok=True)

    isotropic_volumes_path = os.path.join(BASEPATH, 'data_proc/{}/isotropic_volumes_1mm'.format(args.dataset))

    for study_id in groups[args.group - 1]:
        process_study(study_id, isotropic_volumes_path, out_dir, config)
