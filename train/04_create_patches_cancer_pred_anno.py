# Create nodule patches

import os
import pickle
import random
import numpy as np
from scipy.ndimage.interpolation import affine_transform
from skimage.transform import resize
from skimage.util import pad, crop
import multiprocessing
from joblib import Parallel, delayed
import argparse
from uuid import uuid4

labels = pd.read_csv('../data/stage1_labels.csv').sort_values(by='id')
labels.index = labels['id']

with open('../annotations/study_annotations_grouped.pkl', 'rb') as f:
    study_annotations_grouped = pickle.load(f)


def random_rotation(volume, rotation):
    theta_x = np.pi / 180 * np.random.uniform(-rotation, rotation)
    theta_y = np.pi / 180 * np.random.uniform(-rotation, rotation)
    theta_z = np.pi / 180 * np.random.uniform(-rotation, rotation)
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(theta_x), -np.sin(theta_x)],
                                  [0, np.sin(theta_x), np.cos(theta_x)]])
    rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                                  [0, 1, 0],
                                  [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                                  [np.sin(theta_z), np.cos(theta_z), 0],
                                  [0, 0, 1]])
    transform_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
    volume_rotated = affine_transform(volume, transform_matrix, mode='nearest')
    return volume_rotated


def process_study(study_id, in_train_set,
                  isotropic_volumes_folder, volumes_metadata,
                  annotations_grouped, out_dir, config):
    dimensions, patchsize, scaling, multiple, padding, offcenter, rotation, view = config

    isometric_volume = np.load('../data_proc/stage1/{}/{}.npy'.format(isotropic_volumes_folder, study_id))
    mean = np.mean(isometric_volume).astype(np.float32)
    std = np.std(isometric_volume).astype(np.float32)
    resize_factor = np.divide(volumes_metadata['volume_resampled_shape'], volumes_metadata['volume_shape'])

    nodules = []
    for i, group in enumerate(annotations_grouped):
        data = [a['data'] for a in group]
        z0 = int(round(resize_factor[0] * min([a['sliceNum'] for a in group])))
        z1 = int(round(resize_factor[0] * max([a['sliceNum'] for a in group])) + 1)
        y0 = int(round(resize_factor[1] * min([d['y'] for d in data])))
        y1 = int(round(resize_factor[1] * max([(d['y'] + d['height']) for d in data])) + 1)
        x0 = int(round(resize_factor[2] * min([d['x'] for d in data])))
        x1 = int(round(resize_factor[2] * max([(d['x'] + d['width']) for d in data])) + 1)
        # add padding to patch if specified
        z0 = max(0, int(z0 - ((z1 - z0) * padding / 100)))
        z1 = min(isometric_volume.shape[0], int(z1 + ((z1 - z0) * padding / 100)))
        y0 = max(0, int(y0 - ((y1 - y0) * padding / 100)))
        y1 = min(isometric_volume.shape[1], int(y1 + ((y1 - y0) * padding / 100)))
        x0 = max(0, int(x0 - ((x1 - x0) * padding / 100)))
        x1 = min(isometric_volume.shape[2], int(x1 + ((x1 - x0) * padding / 100)))
        if scaling == 'original':
            if (z1 - z0) < patchsize:
                z0 -= (patchsize - (z1 - z0)) // 2
                z1 += (patchsize - (z1 - z0)) - (patchsize - (z1 - z0)) // 2
            if (y1 - y0) < patchsize:
                y0 -= (patchsize - (y1 - y0)) // 2
                y1 += (patchsize - (y1 - y0)) - (patchsize - (y1 - y0)) // 2
            if (x1 - x0) < patchsize:
                x0 -= (patchsize - (x1 - x0)) // 2
                x1 += (patchsize - (x1 - x0)) - (patchsize - (x1 - x0)) // 2

        shape = (z1-z0, y1-y0, x1-x0)
        nodule = isometric_volume[z0:z1, y0:y1, x0:x1]
        nodules.append(nodule)

    if len(nodules) == 0:
        return 0, []

    if multiple == 'largest':
        nodule_sizes_sorted = np.argsort([np.prod(nodule.shape) for nodule in nodules])
        nodules = [nodules[nodule_sizes_sorted[-1]]]

    if scaling == 'stretch':
        nodules = [resize(nodule, [patchsize, patchsize, patchsize],
                          mode='edge', clip=True, preserve_range=True) for nodule in nodules]
    elif scaling == 'original':
        nodules_reshaped = []
        for nodule in nodules:
            pad_width = []
            for s in nodule.shape:
                if s < patchsize:
                    pad_width.append(((patchsize - s) // 2, (patchsize - s) - (patchsize - s) // 2))
                else:
                    pad_width.append((0, 0))
            crop_width = []
            for s in nodule.shape:
                if s > patchsize:
                    crop_width.append(((s - patchsize) // 2, (s - patchsize) - (s - patchsize) // 2))
                else:
                    crop_width.append((0, 0))
            nodules_reshaped.append(crop(pad(nodule, pad_width, mode='minimum'), crop_width))
        nodules = nodules_reshaped

    nodules = [(nodule.astype(np.float32) - mean) / (std + 1e-7) for nodule in nodules]

    if offcenter > 0:
        offcenter_range = range(-int(patchsize * offcenter / 100), int(patchsize * offcenter / 100))
    else:
        offcenter_range = [0]

    patches = []
    if dimensions == '2d':
        for nodule in nodules:
            for i in offcenter_range:
                if view == 'axial':
                    patches.append(np.expand_dims(nodule[int(round(i + nodule.shape[0] / 2)), :, :], axis=2))
                else:
                    patches.append(np.expand_dims(nodule[int(round(i + nodule.shape[0] / 2)), :, :], axis=2))
                    patches.append(np.expand_dims(nodule[:, int(round(i + nodule.shape[1] / 2)), :], axis=2))
                    patches.append(np.expand_dims(nodule[:, :, int(round(i + nodule.shape[2] / 2))], axis=2))
    elif dimensions == '3d':
        patches = [np.expand_dims(nodule, axis=3) for nodule in nodules]

    if in_train_set:
        dirname1 = 'train'
    else:
        dirname1 = 'val'
    dirname2 = str(labels.ix[study_id, 'cancer'])

    for patch in patches:
        out_filepath = os.path.join(out_dir, dirname1, dirname2, '{}.npy'.format(uuid4()))
        np.save(out_filepath, patch)

    return len(patches), [nodule.shape for nodule in nodules]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', choices=['2d', '3d'])
    parser.add_argument('--patchsize', type=int, default=32)
    parser.add_argument('--scaling', choices=['stretch', 'original'])
    parser.add_argument('--multiple', choices=['largest', 'separate'])
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--offcenter', type=int, default=0)
    parser.add_argument('--rotation', type=int, default=0)
    parser.add_argument('--view', type=str, default='axial')
    args = parser.parse_args()

    out_dir = os.path.abspath(
        '../data_train/stage1/cancer_pred_anno/patches_1mm_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.dimensions, args.patchsize, args.scaling,
            args.multiple, args.padding, args.offcenter, args.rotation, args.view
        )
    )
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'train', '0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'train', '1'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val', '0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val', '1'), exist_ok=True)
    metadata_filepath = os.path.abspath(
        '../data_train/stage1/cancer_pred_anno/patches_1mm_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(
            args.dimensions, args.patchsize, args.scaling,
            args.multiple, args.padding, args.offcenter, args.rotation, args.view
        )
    )

    with open('../data_proc/stage1/isotropic_volumes_1mm.pkl', 'rb') as f:
        isotropic_volumes_metadata = pickle.load(f)
    isotropic_volumes_folder = 'isotropic_volumes_1mm'

    study_ids = list(study_annotations_grouped.keys())
    print('# study ids:', len(study_ids))
    random.seed(42)
    study_ids_train_set = set(random.sample(study_ids, int(len(study_ids) * 0.8)))
    in_train_set = {study_id: (study_id in study_ids_train_set) for study_id in study_ids}

    config = (
        args.dimensions,
        args.patchsize,
        args.scaling,
        args.multiple,
        args.padding,
        args.offcenter,
        args.rotation,
        args.view
    )

    n_jobs = multiprocessing.cpu_count() - 1
    print('# jobs processing in parallel:', n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_study)(
            study_id, in_train_set[study_id],
            isotropic_volumes_folder, isotropic_volumes_metadata[study_id],
            study_annotations_grouped[study_id], out_dir, config
        ) for study_id in study_ids
    )
    print('# processed:', len(results))

    metadata = {}
    for i, (num_patches, nodule_orig_shapes) in enumerate(results):
        if num_patches > 0:
            metadata[study_ids[i]] = {
                'num_patches': num_patches,
                'nodule_orig_shapes': nodule_orig_shapes
            }

    print('saving metadata file to:', metadata_filepath)
    with open(metadata_filepath, 'wb') as f:
        pickle.dump(metadata, f)


if __name__ == '__main__':
    main()
