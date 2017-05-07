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
import itertools

studies_train = pd.read_csv('../data/stage1_labels.csv').sort_values(by='id')['id'].tolist()

with open('../annotations/study_annotations_grouped.pkl', 'rb') as f:
    study_annotations_grouped = pickle.load(f)
with open('../data_proc/stage1/isotropic_volumes_1mm.pkl', 'rb') as f:
    isotropic_volumes_metadata = pickle.load(f)


def sample_positive_2daxial(isometric_volume, mean, std,
                            resize_factor, annotations, split_out_dir,
                            patchsize, num_samples=10):
    patch_coords = []
    for a in annotations:
        d = a['data']
        z = int(round(resize_factor[0] * a['sliceNum']))
        y0 = resize_factor[1] * d['y']
        y1 = resize_factor[1] * (d['y'] + d['height'])
        x0 = resize_factor[2] * d['x']
        x1 = resize_factor[2] * (d['x'] + d['width'])
        if (y1 - y0) > patchsize:
            sample_range_y = sorted([int(round(y0 - patchsize / 4)), int(round(y1 - 3 * patchsize / 4))])
        else:
            sample_range_y = sorted([int(round(y1 - 3 * patchsize / 4)), int(round(y0 - patchsize / 4))])
        if (x1 - x0) > patchsize:
            sample_range_x = sorted([int(round(x0 - patchsize / 4)), int(round(x1 - 3 * patchsize / 4))])
        else:
            sample_range_x = sorted([int(round(x1 - 3 * patchsize / 4)), int(round(x0 - patchsize / 4))])

        for i in range(num_samples):
            rand_y0 = random.randint(*sample_range_y)
            rand_x0 = random.randint(*sample_range_x)
            rand_y1 = rand_y0 + patchsize
            rand_x1 = rand_x0 + patchsize
            if rand_y0 < 0 or rand_y1 > isometric_volume.shape[1]:
                if rand_y0 < 0:
                    rand_y1 += -rand_y0
                    rand_y0 = 0
                else:
                    rand_y0 -= (rand_y1 - isometric_volume.shape[1])
                    rand_y1 = isometric_volume.shape[1]
            if rand_x0 < 0 or rand_x1 > isometric_volume.shape[2]:
                if rand_x0 < 0:
                    rand_x1 += -rand_x0
                    rand_x0 = 0
                else:
                    rand_x0 -= (rand_x1 - isometric_volume.shape[2])
                    rand_x1 = isometric_volume.shape[2]
            patch = isometric_volume[z, rand_y0:rand_y1, rand_x0:rand_x1]
            patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
            patch = np.expand_dims(patch, axis=2)
            patch_coords.append((z, rand_y0, rand_y1, rand_x0, rand_x1))
            out_filepath = os.path.join(split_out_dir, '1', '{}.npy'.format(uuid4()))
            np.save(out_filepath, patch)

    return patch_coords


def sample_negative_2daxial(isometric_volume, mean, std,
                            resize_factor, annotations, split_out_dir,
                            patchsize, num_samples=100):

    anno_coords = []
    for a in annotations:
        d = a['data']
        z = int(round(resize_factor[0] * a['sliceNum']))
        y0 = resize_factor[1] * d['y']
        y1 = resize_factor[1] * (d['y'] + d['height'])
        x0 = resize_factor[2] * d['x']
        x1 = resize_factor[2] * (d['x'] + d['width'])
        anno_coords.append((z, y0, y1, x0, x1))

    patch_coords = []
    for i in range(num_samples):
        rand_z = random.randint(0, isometric_volume.shape[0] - 1)
        rand_y0 = random.randint(0, isometric_volume.shape[1] - 1 - patchsize)
        rand_x0 = random.randint(0, isometric_volume.shape[2] - 1 - patchsize)
        rand_y1 = rand_y0 + patchsize
        rand_x1 = rand_x0 + patchsize
        overlaps = []
        for (z, y0, y1, x0, x1) in anno_coords:
            overlap_z = rand_z == z
            overlap_xy = max(0, max(rand_y1, y1) - min(rand_y0, y0)) * max(0, max(rand_x1, x1) - min(rand_x0, x0)) > 0
            overlaps.append(overlap_z and overlap_xy)
        if any(overlaps):
            continue
        patch = isometric_volume[rand_z, rand_y0:rand_y1, rand_x0:rand_x1]
        patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
        patch = np.expand_dims(patch, axis=2)
        patch_coords.append((rand_z, rand_y0, rand_y1, rand_x0, rand_x1))
        out_filepath = os.path.join(split_out_dir, '0', '{}.npy'.format(uuid4()))
        np.save(out_filepath, patch)

    return patch_coords


def sample_positive_2daxial_stack(isometric_volume, mean, std,
                                  resize_factor, annotations_grouped, split_out_dir,
                                  patchsize, num_samples=10, nb_stacks=2):
    patch_coords = []
    for group in annotations_grouped:
        z_min = int(round(resize_factor[0] * min([a['sliceNum'] for a in group])))
        z_max = int(round(resize_factor[0] * (max([a['sliceNum'] for a in group]) + 1)))
        z_annotations = np.array([int(round(resize_factor[0] * a['sliceNum'])) for a in group])
        for z in range(z_min, z_max):
            a = group[(np.abs(z_annotations - z)).argmin()]
            d = a['data']
            y0 = resize_factor[1] * d['y']
            y1 = resize_factor[1] * (d['y'] + d['height'])
            x0 = resize_factor[2] * d['x']
            x1 = resize_factor[2] * (d['x'] + d['width'])
            if (y1 - y0) > patchsize:
                sample_range_y = sorted([int(round(y0 - patchsize / 4)), int(round(y1 - 3 * patchsize / 4))])
            else:
                sample_range_y = sorted([int(round(y1 - 3 * patchsize / 4)), int(round(y0 - patchsize / 4))])
            if (x1 - x0) > patchsize:
                sample_range_x = sorted([int(round(x0 - patchsize / 4)), int(round(x1 - 3 * patchsize / 4))])
            else:
                sample_range_x = sorted([int(round(x1 - 3 * patchsize / 4)), int(round(x0 - patchsize / 4))])

            for i in range(num_samples):
                rand_y0 = random.randint(*sample_range_y)
                rand_x0 = random.randint(*sample_range_x)
                rand_y1 = rand_y0 + patchsize
                rand_x1 = rand_x0 + patchsize
                if rand_y0 < 0 or rand_y1 > isometric_volume.shape[1]:
                    if rand_y0 < 0:
                        rand_y1 += -rand_y0
                        rand_y0 = 0
                    else:
                        rand_y0 -= (rand_y1 - isometric_volume.shape[1])
                        rand_y1 = isometric_volume.shape[1]
                if rand_x0 < 0 or rand_x1 > isometric_volume.shape[2]:
                    if rand_x0 < 0:
                        rand_x1 += -rand_x0
                        rand_x0 = 0
                    else:
                        rand_x0 -= (rand_x1 - isometric_volume.shape[2])
                        rand_x1 = isometric_volume.shape[2]
                patch = isometric_volume[z:z+nb_stacks, rand_y0:rand_y1, rand_x0:rand_x1]
                patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
                patch = np.moveaxis(patch, 0, 2)
                patch_coords.append((z, z+nb_stacks, rand_y0, rand_y1, rand_x0, rand_x1))
                out_filepath = os.path.join(split_out_dir, '1', '{}.npy'.format(uuid4()))
                np.save(out_filepath, patch)

    return patch_coords


def sample_negative_2daxial_stack(isometric_volume, mean, std,
                                  resize_factor, annotations_grouped, split_out_dir,
                                  patchsize, num_samples=10, nb_stacks=2):

    anno_coords = []
    for a in list(itertools.chain(*annotations_grouped)):
        d = a['data']
        z = int(round(resize_factor[0] * a['sliceNum']))
        y0 = resize_factor[1] * d['y']
        y1 = resize_factor[1] * (d['y'] + d['height'])
        x0 = resize_factor[2] * d['x']
        x1 = resize_factor[2] * (d['x'] + d['width'])
        anno_coords.append((z, y0, y1, x0, x1))

    patch_coords = []
    for i in range(num_samples):
        rand_z = random.randint(0, isometric_volume.shape[0] - nb_stacks)
        rand_y0 = random.randint(0, isometric_volume.shape[1] - 1 - patchsize)
        rand_x0 = random.randint(0, isometric_volume.shape[2] - 1 - patchsize)
        rand_y1 = rand_y0 + patchsize
        rand_x1 = rand_x0 + patchsize
        overlaps = []
        for (z, y0, y1, x0, x1) in anno_coords:
            overlap_z = (rand_z <= z and (rand_z + nb_stacks) > z)
            overlap_xy = max(0, max(rand_y1, y1) - min(rand_y0, y0)) * max(0, max(rand_x1, x1) - min(rand_x0, x0)) > 0
            overlaps.append(overlap_z and overlap_xy)
        if any(overlaps):
            continue
        patch = isometric_volume[rand_z:rand_z+nb_stacks, rand_y0:rand_y1, rand_x0:rand_x1]
        patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
        patch = np.moveaxis(patch, 0, 2)
        patch_coords.append((rand_z, rand_z+nb_stacks, rand_y0, rand_y1, rand_x0, rand_x1))
        out_filepath = os.path.join(split_out_dir, '0', '{}.npy'.format(uuid4()))
        np.save(out_filepath, patch)

    return patch_coords


def sample_positive_2d3view(isometric_volume, mean, std,
                            resize_factor, annotations, split_out_dir,
                            patchsize):
    patch_coords = []
    for a in annotations:
        d = a['data']
        z = int(round(resize_factor[0] * a['sliceNum']))
        z0 = int(round(z - patchsize // 2))
        z1 = z0 + patchsize
        y0 = int(round(resize_factor[1] * d['y']))
        y1 = int(round(resize_factor[1] * (d['y'] + d['height'])))
        x0 = int(round(resize_factor[2] * d['x']))
        x1 = int(round(resize_factor[2] * (d['x'] + d['width'])))
        y = int(round((y0 + y1) / 2))
        x = int(round((x0 + x1) / 2))

        for ii in range(-2, 3):
            for jj in range(-2, 3):
                y0 = int(round(y + ii - patchsize // 2))
                y1 = y0 + patchsize
                x0 = int(round(x + jj - patchsize // 2))
                x1 = x0 + patchsize
                y = int(round((y0 + y1) / 2))
                x = int(round((x0 + x1) / 2))
                if z0 < 0 or z1 > isometric_volume.shape[0]:
                    if z0 < 0:
                        z1 += -z0
                        z0 = 0
                    else:
                        z0 -= (z1 - isometric_volume.shape[0])
                        z1 = isometric_volume.shape[0]
                if y0 < 0 or y1 > isometric_volume.shape[1]:
                    if y0 < 0:
                        y1 += -y0
                        y0 = 0
                    else:
                        y0 -= (y1 - isometric_volume.shape[1])
                        y1 = isometric_volume.shape[1]
                if x0 < 0 or x1 > isometric_volume.shape[2]:
                    if x0 < 0:
                        x1 += -x0
                        x0 = 0
                    else:
                        x0 -= (x1 - isometric_volume.shape[2])
                        x1 = isometric_volume.shape[2]
                patch = np.moveaxis(np.array([
                    isometric_volume[z, y0:y1, x0:x1],
                    isometric_volume[z0:z1, y, x0:x1],
                    isometric_volume[z0:z1, y0:y1, x]
                ], dtype=np.float32), 0, 2)
                patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
                patch_coords.append((z0, z1, y0, y1, x0, x1))
                out_filepath = os.path.join(split_out_dir, '1', '{}.npy'.format(uuid4()))
                np.save(out_filepath, patch)

    return patch_coords


def sample_negative_2d3view(isometric_volume, mean, std,
                            resize_factor, annotations, split_out_dir,
                            patchsize, num_samples=10):

    anno_coords = []
    for a in annotations:
        d = a['data']
        z = int(round(resize_factor[0] * a['sliceNum']))
        y0 = resize_factor[1] * d['y']
        y1 = resize_factor[1] * (d['y'] + d['height'])
        x0 = resize_factor[2] * d['x']
        x1 = resize_factor[2] * (d['x'] + d['width'])
        anno_coords.append((z, y0, y1, x0, x1))

    patch_coords = []
    for i in range(num_samples):
        rand_z0 = random.randint(0, isometric_volume.shape[0] - 1 - patchsize)
        rand_y0 = random.randint(0, isometric_volume.shape[1] - 1 - patchsize)
        rand_x0 = random.randint(0, isometric_volume.shape[2] - 1 - patchsize)
        rand_z1 = rand_z0 + patchsize
        rand_y1 = rand_y0 + patchsize
        rand_x1 = rand_x0 + patchsize
        overlaps = []
        for (z, y0, y1, x0, x1) in anno_coords:
            overlap_z = (rand_z0 <= z and rand_z1 > z)
            overlap_xy = max(0, max(rand_y1, y1) - min(rand_y0, y0)) * max(0, max(rand_x1, x1) - min(rand_x0, x0)) > 0
            overlaps.append(overlap_z and overlap_xy)
        if any(overlaps):
            continue
        volume = isometric_volume[rand_z0:rand_z1, rand_y0:rand_y1, rand_x0:rand_x1]
        volume = (volume.astype(np.float32) - mean) / (std + 1e-7)
        patches = []
        for ii in range(volume.shape[0] // 2 - 2, volume.shape[0] // 2 + 2):
            patch = np.moveaxis(
                np.array([volume[ii, :, :], volume[:, ii, :], volume[:, :, ii]], dtype=np.float32),
                0, 2
            )
            patch_coords.append((rand_z0, rand_z1, rand_y0, rand_y1, rand_x0, rand_x1))
            out_filepath = os.path.join(split_out_dir, '0', '{}.npy'.format(uuid4()))
            np.save(out_filepath, patch)

    return patch_coords


def sample_positive_3d(isometric_volume, mean, std,
                       resize_factor, annotations, split_out_dir,
                       patchsize):
    patch_coords = []
    for a in annotations:
        d = a['data']
        z = resize_factor[0] * a['sliceNum']
        y = ((resize_factor[1] * d['y']) + (resize_factor[1] * (d['y'] + d['height']))) / 2
        x = ((resize_factor[2] * d['x']) + (resize_factor[2] * (d['x'] + d['width']))) / 2

        z0 = int(round(z - patchsize // 2))
        z1 = z0 + patchsize
        y0 = int(round(y - patchsize // 2))
        y1 = y0 + patchsize
        x0 = int(round(x - patchsize // 2))
        x1 = x0 + patchsize
        if z0 < 0 or z1 > isometric_volume.shape[0]:
            if z0 < 0:
                z1 += -z0
                z0 = 0
            else:
                z0 -= (z1 - isometric_volume.shape[0])
                z1 = isometric_volume.shape[0]
        if y0 < 0 or y1 > isometric_volume.shape[1]:
            if y0 < 0:
                y1 += -y0
                y0 = 0
            else:
                y0 -= (y1 - isometric_volume.shape[1])
                y1 = isometric_volume.shape[1]
        if x0 < 0 or x1 > isometric_volume.shape[2]:
            if x0 < 0:
                x1 += -x0
                x0 = 0
            else:
                x0 -= (x1 - isometric_volume.shape[2])
                x1 = isometric_volume.shape[2]
        patch = isometric_volume[z0:z1, y0:y1, x0:x1]
        patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
        patch = np.expand_dims(patch, axis=3)
        patch_coords.append((z0, z1, y0, y1, x0, x1))
        out_filepath = os.path.join(split_out_dir, '1', '{}.npy'.format(uuid4()))
        np.save(out_filepath, patch)

    return patch_coords


def sample_negative_3d(isometric_volume, mean, std,
                       resize_factor, annotations, split_out_dir,
                       patchsize, num_samples=10):

    anno_coords = []
    for a in annotations:
        d = a['data']
        z = int(round(resize_factor[0] * a['sliceNum']))
        y0 = resize_factor[1] * d['y']
        y1 = resize_factor[1] * (d['y'] + d['height'])
        x0 = resize_factor[2] * d['x']
        x1 = resize_factor[2] * (d['x'] + d['width'])
        anno_coords.append((z, y0, y1, x0, x1))

    patch_coords = []
    for i in range(num_samples):
        rand_z0 = random.randint(0, isometric_volume.shape[0] - 1 - patchsize)
        rand_y0 = random.randint(0, isometric_volume.shape[1] - 1 - patchsize)
        rand_x0 = random.randint(0, isometric_volume.shape[2] - 1 - patchsize)
        rand_z1 = rand_z0 + patchsize
        rand_y1 = rand_y0 + patchsize
        rand_x1 = rand_x0 + patchsize
        overlaps = []
        for (z, y0, y1, x0, x1) in anno_coords:
            overlap_z = (rand_z0 <= z and rand_z1 > z)
            overlap_xy = max(0, max(rand_y1, y1) - min(rand_y0, y0)) * max(0, max(rand_x1, x1) - min(rand_x0, x0)) > 0
            overlaps.append(overlap_z and overlap_xy)
        if any(overlaps):
            continue
        patch = isometric_volume[rand_z0:rand_z1, rand_y0:rand_y1, rand_x0:rand_x1]
        patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
        patch = np.expand_dims(patch, axis=3)
        patch_coords.append((rand_z0, rand_z1, rand_y0, rand_y1, rand_x0, rand_x1))
        out_filepath = os.path.join(split_out_dir, '0', '{}.npy'.format(uuid4()))
        np.save(out_filepath, patch)

    return patch_coords


def process_study(study_id, in_train_set, volumes_metadata, annotations_grouped, out_dir, config):
    dimensions, patchsize = config

    isometric_volume = np.load('../data_proc/stage1/isotropic_volumes_1mm/{}.npy'.format(study_id))
    resize_factor = np.divide(volumes_metadata['volume_resampled_shape'], volumes_metadata['volume_shape'])
    mean = np.mean(isometric_volume).astype(np.float32)
    std = np.std(isometric_volume).astype(np.float32)

    annotations = list(itertools.chain(*annotations_grouped))
    split_out_dir = os.path.join(out_dir, 'train' if in_train_set else 'val')

    if dimensions == '2daxial':
        patch_coords_neg = sample_negative_2daxial(
            isometric_volume, mean, std, resize_factor, annotations, split_out_dir, patchsize,
            num_samples=100
        )
        patch_coords_pos = sample_positive_2daxial(
            isometric_volume, mean, std, resize_factor, annotations, split_out_dir, patchsize,
            num_samples=7
        )
    elif dimensions in ['2daxial2stack', '2daxial3stack', '2daxial4stack', '2daxial5stack']:
        if dimensions == '2daxial2stack':
            nb_stacks = 2
        elif dimensions == '2daxial3stack':
            nb_stacks = 3
        elif dimensions == '2daxial4stack':
            nb_stacks = 4
        elif dimensions == '2daxial5stack':
            nb_stacks = 5
        patch_coords_neg = sample_negative_2daxial_stack(
            isometric_volume, mean, std, resize_factor, annotations_grouped, split_out_dir, patchsize,
            num_samples=100,
            nb_stacks=nb_stacks
        )
        patch_coords_pos = sample_positive_2daxial_stack(
            isometric_volume, mean, std, resize_factor, annotations_grouped, split_out_dir, patchsize,
            num_samples=7,
            nb_stacks=nb_stacks
        )
    elif dimensions == '2d3view':
        patch_coords_neg = sample_negative_2d3view(
            isometric_volume, mean, std, resize_factor, annotations, split_out_dir, patchsize,
            num_samples=100
        )
        patch_coords_pos = sample_positive_2d3view(
            isometric_volume, mean, std, resize_factor, annotations, split_out_dir, patchsize
        )
    elif dimensions == '3d':
        patch_coords_neg = sample_negative_3d(
            isometric_volume, mean, std, resize_factor, annotations, split_out_dir, patchsize,
            num_samples=20
        )
        patch_coords_pos = sample_positive_3d(
            isometric_volume, mean, std, resize_factor, annotations, split_out_dir, patchsize
        )
    else:
        raise Exception('invalid dimensions arg')

    return patch_coords_neg, patch_coords_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', choices=[
        '2daxial', '2daxial2stack', '2daxial3stack', '2daxial4stack', '2daxial5stack', '2d3view', '3d'
    ])
    parser.add_argument('--patchsize', type=int, default=32)
    parser.add_argument('--iternum', type=int, default=1)
    args = parser.parse_args()

    out_dir = os.path.abspath(
        '../data_train/stage1/nodule_detect/patches_1mm_{}_{}_{}_1'.format(
            args.dimensions, args.patchsize, args.iternum
        )
    )
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'train', '0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'train', '1'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val', '0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val', '1'), exist_ok=True)
    metadata_filepath = os.path.abspath(
        '../data_train/stage1/nodule_detect/patches_1mm_{}_{}_{}_1.pkl'.format(
            args.dimensions, args.patchsize, args.iternum
        )
    )

    study_ids = list(studies_train)
    print('# study ids:', len(study_ids))
    random.seed(42 + args.iternum)
    study_ids_train_set = set(random.sample(study_ids, int(len(study_ids) * 0.8)))
    in_train_set = {study_id: (study_id in study_ids_train_set) for study_id in study_ids}

    config = (
        args.dimensions,
        args.patchsize
    )

    n_jobs = multiprocessing.cpu_count() - 1
    print('# jobs processing in parallel:', n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_study)(
            study_id, in_train_set[study_id], isotropic_volumes_metadata[study_id],
            study_annotations_grouped[study_id], out_dir, config
        ) for study_id in study_ids
    )
    print('# processed:', len(results))

    metadata = {}
    metadata['num_patches_neg'] = sum([len(patch_coords_neg) for (patch_coords_neg, patch_coords_pos) in results])
    metadata['num_patches_pos'] = sum([len(patch_coords_pos) for (patch_coords_neg, patch_coords_pos) in results])
    metadata['patch_coords_neg'] = list(
        itertools.chain(*[patch_coords_neg for (patch_coords_neg, patch_coords_pos) in results])
    )
    metadata['patch_coords_pos'] = list(
        itertools.chain(*[patch_coords_pos for (patch_coords_neg, patch_coords_pos) in results])
    )

    print('saving metadata file to:', metadata_filepath)
    with open(metadata_filepath, 'wb') as f:
        pickle.dump(metadata, f)


if __name__ == '__main__':
    main()
