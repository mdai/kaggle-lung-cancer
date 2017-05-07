# Create nodule patches

import os
import pickle
import random
import numpy as np
from scipy.ndimage.interpolation import affine_transform
from skimage.transform import resize
from skimage.util import pad, crop
import multiprocessing
import joblib
from joblib import Parallel, delayed
import argparse
from uuid import uuid4

with open('../annotations/annotations_by_study_id.pkl', 'rb') as f:
    annotations_by_study_id = pickle.load(f)
with open('../data_proc/stage1/isotropic_volumes_1mm.pkl', 'rb') as f:
    isotropic_volumes_metadata = pickle.load(f)


def process_study(study_id, annotations, out_dir, nstack):
    volumes_metadata = isotropic_volumes_metadata[study_id]
    isometric_volume = np.load('../data_proc/stage1/isotropic_volumes_1mm/{}.npy'.format(study_id))
    mean = np.mean(isometric_volume).astype(np.float32)
    std = np.std(isometric_volume).astype(np.float32)
    resize_factor = np.divide(volumes_metadata['volume_resampled_shape'], volumes_metadata['volume_shape'])

    coords_list = []
    for a in annotations:
        d = a['data']
        z = int(round(resize_factor[0] * a['sliceNum']))
        y0 = resize_factor[1] * d['y']
        y1 = resize_factor[1] * (d['y'] + d['height'])
        x0 = resize_factor[2] * d['x']
        x1 = resize_factor[2] * (d['x'] + d['width'])
        coords_list.append((z, y0, y1, x0, x1))

    samples = []
    for coords in coords_list:
        z, y0, y1, x0, x1 = coords
        for i in range(40):
            sample_id = uuid4()
            rand_y0 = max(0, int(round(y0 - random.randint(0, 32))))
            rand_y1 = min(isometric_volume.shape[1], int(round(y1 + random.randint(0, 32))))
            rand_x0 = max(0, int(round(x0 - random.randint(0, 32))))
            rand_x1 = min(isometric_volume.shape[2], int(round(x1 + random.randint(0, 32))))
            patch = []
            for zi in range(nstack):
                patch.append(resize(isometric_volume[z+zi, rand_y0:rand_y1, rand_x0:rand_x1], [32, 32],
                                    mode='edge', clip=True, preserve_range=True))
            patch = np.array(patch, dtype=np.float32)
            patch = (patch - mean) / (std + 1e-7)
            patch = np.moveaxis(patch, 0, 2)
            bb_x = (x0 - rand_x0) / (rand_x1 - rand_x0)
            bb_y = (y0 - rand_y0) / (rand_y1 - rand_y0)
            bb_w = (x1 - x0) / (rand_x1 - rand_x0)
            bb_h = (y1 - y0) / (rand_y1 - rand_y0)
            samples.append((patch, bb_x, bb_y, bb_w, bb_h))

    joblib.dump(samples, os.path.join(out_dir, 'samples', '{}.pkl'.format(study_id)))
    return len(samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstack', type=int, default=1)
    args = parser.parse_args()

    out_dir = os.path.abspath('../data_train/stage1/nodule_bbox/patches_{}'.format(args.nstack))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'samples'), exist_ok=True)

    study_ids = sorted(list(annotations_by_study_id.keys()))
    print('# study ids:', len(study_ids))
    random.seed(43)
    study_ids_train_set = set(random.sample(study_ids, int(len(study_ids) * 0.9)))
    study_ids_val_set = set(study_ids) - study_ids_train_set

    n_jobs = multiprocessing.cpu_count() - 1
    print('# jobs processing in parallel:', n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_study)(
            study_id, annotations_by_study_id[study_id], out_dir, args.nstack
        ) for study_id in study_ids
    )
    print('# processed:', len(results))

    patches_train = []
    bboxes_train = []
    patches_val = []
    bboxes_val = []
    for study_id in study_ids:
        samples = joblib.load(os.path.join(out_dir, 'samples', '{}.pkl'.format(study_id)))
        for patch, bb_x, bb_y, bb_w, bb_h in samples:
            if study_id in study_ids_train_set:
                patches_train.append(patch)
                bboxes_train.append(np.array([bb_x, bb_y, bb_w, bb_h]))
            else:
                patches_val.append(patch)
                bboxes_val.append(np.array([bb_x, bb_y, bb_w, bb_h]))

    patches_train = np.array(patches_train, dtype=np.float32)
    bboxes_train = np.array(bboxes_train, dtype=np.float32)
    indices_shuffled_train = np.arange(patches_train.shape[0])
    np.random.shuffle(indices_shuffled_train)
    patches_train = patches_train[indices_shuffled_train]
    bboxes_train = bboxes_train[indices_shuffled_train]

    patches_val = np.array(patches_val, dtype=np.float32)
    bboxes_val = np.array(bboxes_val, dtype=np.float32)
    indices_shuffled_val = np.arange(patches_val.shape[0])
    np.random.shuffle(indices_shuffled_val)
    patches_val = patches_val[indices_shuffled_val]
    bboxes_val = bboxes_val[indices_shuffled_val]

    np.save(os.path.join(out_dir, 'patches_train.npy'), patches_train)
    np.save(os.path.join(out_dir, 'bboxes_train.npy'), bboxes_train)

    np.save(os.path.join(out_dir, 'patches_val.npy'), patches_val)
    np.save(os.path.join(out_dir, 'bboxes_val.npy'), bboxes_val)
