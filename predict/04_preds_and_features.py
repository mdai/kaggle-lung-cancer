import os
import pickle
import random
import time
import numpy as np
import pandas as pd
from skimage.transform import resize
import argparse
import json
from uuid import uuid1
from scipy.ndimage import measurements
from scipy.stats.mstats import gmean
import redis

redis_client = redis.StrictRedis()

from configs import configs


def get_group_bounds(coords):
    coords_z, coords_y, coords_x = coords

    # outer bounds
    z_min = np.min(coords_z)
    z_max = np.max(coords_z)
    y_min = np.min(coords_y)
    y_max = np.max(coords_y)
    x_min = np.min(coords_x)
    x_max = np.max(coords_x)
    z_outer, y_outer, x_outer = ((z_max - z_min + 1), (y_max - y_min + 1), (x_max - x_min + 1))
    volume_size_outer = z_outer * y_outer * x_outer

    # per view bounds
    x_zy = {}
    y_zx = {}
    z_yx = {}
    for i in range(coords_z.shape[0]):
        zy = (coords_z[i], coords_y[i])
        if zy in x_zy:
            x_zy[zy].append(coords_x[i])
        else:
            x_zy[zy] = [coords_x[i]]
        zx = (coords_z[i], coords_x[i])
        if zx in y_zx:
            y_zx[zx].append(coords_y[i])
        else:
            y_zx[zx] = [coords_y[i]]
        yx = (coords_y[i], coords_x[i])
        if yx in z_yx:
            z_yx[yx].append(coords_z[i])
        else:
            z_yx[yx] = [coords_z[i]]
    z_yx = [(np.max(z_arr) - np.min(z_arr)) for z_arr in list(z_yx.values())]
    y_zx = [(np.max(y_arr) - np.min(y_arr)) for y_arr in list(y_zx.values())]
    x_zy = [(np.max(x_arr) - np.min(x_arr)) for x_arr in list(x_zy.values())]
    z_inner, y_inner, x_inner = np.max(z_yx) + 1, np.max(y_zx) + 1, np.max(x_zy) + 1
    volume_size_inner = z_inner * y_inner * x_inner

    bounds_outer = (z_outer, y_outer, x_outer)
    bounds_inner = (z_inner, y_inner, x_inner)
    return bounds_outer, bounds_inner, volume_size_outer, volume_size_inner


def create_bbs(coords):
    coords_z, coords_y, coords_x = coords
    bboxes = []
    # axial
    for z in set(coords_z):
        slice_coords_y = coords_y[np.where(coords_z == z)]
        slice_coords_x = coords_x[np.where(coords_z == z)]
        y_min = np.min(slice_coords_y)
        y_max = np.max(slice_coords_y)
        x_min = np.min(slice_coords_x)
        x_max = np.max(slice_coords_x)
        bboxes.append((z, y_min, y_max, x_min, x_max))
    return bboxes


def process_study(study_id, probmaps_dir, isometric_volumes_dir, out_dir, config, config_num):
    print(study_id, end=' ...', flush=True)
    start_time = time.time()

    (probmap, nodule_model_key, bbox_model_key, cancer_model_keys,
     probthresh_2d, probthresh_3d, sizethresh, topk,
     volume_mode, size_mode, middle_slices, aggreg_func) = config

    isometric_volume = np.load(os.path.join(isometric_volumes_dir, '{}.npy'.format(study_id)))
    mean = np.mean(isometric_volume).astype(np.float32)
    std = np.std(isometric_volume).astype(np.float32)

    probmap = np.load(os.path.join(probmaps_dir, '{}.npy'.format(study_id)))
    probmap[np.where(probmap < probthresh_2d/100)] = 0
    probmap[np.where(probmap > 0)] = 1

    patches_bbox_orig = []
    patches_bbox_pred = []
    for z in range(probmap.shape[0]):
        groups, nb_groups = measurements.label(probmap[z, :, :])
        for n in range(nb_groups):
            coords = np.where(groups == n+1)
            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])
            patches_bbox_orig.append((z, y_min, y_max, x_min, x_max))
            if bbox_model_key == 'm02a':
                patch = isometric_volume[z, y_min:y_max, x_min:x_max]
                patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
                patch = resize(patch, [32, 32], mode='edge', clip=True, preserve_range=True)
                patch = np.expand_dims(patch, axis=2)
            elif bbox_model_key == 'm04a':
                patch = isometric_volume[z:z+2, y_min:y_max, x_min:x_max]
                patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
                patch = np.moveaxis(patch, 0, 2)
                patch = resize(patch, [32, 32, 2], mode='edge', clip=True, preserve_range=True)
            else:
                raise ValueError('Invalid bbox_model_key')
            patches_bbox_pred.append(patch)

    if not patches_bbox_pred:
        print('{} s'.format(time.time() - start_time))
        return 0

    patches_bbox_pred = np.array(patches_bbox_pred, dtype=np.float32)
    task = {
        'config_num': config_num,
        'study_id': study_id,
        'id': uuid1(),
        'type': 'bbox',
        'input_data': patches_bbox_pred,
        'keys': bbox_model_key,
    }
    redis_client.rpush('tasks', pickle.dumps(task))
    has_result = False
    while not has_result:
        if redis_client.hexists('results', task['id']):
            bbox_preds = pickle.loads(redis_client.hget('results', task['id']))
            redis_client.hdel('results', task['id'])
            has_result = True
        else:
            time.sleep(0.1)
    bbox_preds = np.clip(bbox_preds, 0.0, 1.0)

    probmap_refined = np.zeros(probmap.shape)
    for bbox_orig, bbox_pred in zip(patches_bbox_orig, bbox_preds):
        z, y_min, y_max, x_min, x_max = bbox_orig
        x_offset = bbox_pred[0] * (x_max - x_min)
        y_offset = bbox_pred[1] * (y_max - y_min)
        w = bbox_pred[2] * (x_max - x_min)
        h = bbox_pred[3] * (y_max - y_min)
        y0 = int(round(y_min + y_offset))
        y1 = int(round(y_min + y_offset + h))
        x0 = int(round(x_min + x_offset))
        x1 = int(round(x_min + x_offset + w))
        probmap_refined[z, y0:y1, x0:x1] = 1

    groups, nb_groups = measurements.label(probmap_refined)
    group_coords = [np.where(groups == n+1) for n in range(nb_groups)]
    group_bounds = [get_group_bounds(coords) for coords in group_coords]
    thresh_group_volumes = []
    thresh_group_coords = []
    thresh_group_bounds = []
    thresh_group_bboxes = []
    for coords, bounds in zip(group_coords, group_bounds):
        b_o, b_i, v_o, v_i = bounds
        if volume_mode == 'outer':
            volume_estimate = v_o
        else:
            volume_estimate = v_i
        if size_mode == 'outer':
            bound_single_axis = b_o
        else:
            bound_single_axis = b_i
        if all([b > sizethresh for b in bound_single_axis]):
            thresh_group_volumes.append(volume_estimate)
            thresh_group_coords.append(coords)
            thresh_group_bounds.append(bounds)
            bboxes = create_bbs(coords)
            thresh_group_bboxes.append(bboxes)
    sorted_vol_indices = np.argsort(thresh_group_volumes)[::-1]
    thresh_group_volumes = [thresh_group_volumes[i] for i in sorted_vol_indices]
    thresh_group_coords = [thresh_group_coords[i] for i in sorted_vol_indices]
    thresh_group_bounds = [thresh_group_bounds[i] for i in sorted_vol_indices]
    thresh_group_bboxes = [thresh_group_bboxes[i] for i in sorted_vol_indices]

    if not thresh_group_bboxes:
        print('{} s'.format(time.time() - start_time))
        return 0

    patches_3d = []
    for bboxes in thresh_group_bboxes:
        bbox = bboxes[len(bboxes) // 2]
        z, y_min, y_max, x_min, x_max = bbox
        z0 = z - 32
        z1 = z0 + 64
        y0 = int(round((y_min+y_max)/2 - 32))
        y1 = y0 + 64
        x0 = int(round((x_min+x_max)/2 - 32))
        x1 = x0 + 64
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
        patches_3d.append(patch)
    patches_3d = np.array(patches_3d, dtype=np.float32)

    task = {
        'config_num': config_num,
        'study_id': study_id,
        'id': uuid1(),
        'type': 'nodule',
        'input_data': patches_3d[:487],  # truncate to top n=487 to prevent hitting redis 512MB max string size
        'keys': nodule_model_key,
    }
    redis_client.rpush('tasks', pickle.dumps(task))
    has_result = False
    while not has_result:
        if redis_client.hexists('results', task['id']):
            nodule_preds = pickle.loads(redis_client.hget('results', task['id']))
            redis_client.hdel('results', task['id'])
            has_result = True
        else:
            time.sleep(0.1)

    nodule_indices = np.where(nodule_preds > probthresh_3d/100)[0]
    thresh_group_volumes = [thresh_group_volumes[i] for i in nodule_indices]
    thresh_group_bboxes = [thresh_group_bboxes[i] for i in nodule_indices]

    if not thresh_group_bboxes:
        print('{} s'.format(time.time() - start_time))
        return 0

    patches = []
    for bboxes in thresh_group_bboxes[0:topk]:
        for bbox in bboxes:
            z, y_min, y_max, x_min, x_max = bbox
            try:
                patch = isometric_volume[z, y_min:y_max, x_min:x_max]
                patch = (patch.astype(np.float32) - mean) / (std + 1e-7)
                patch = resize(patch, [32, 32], mode='edge', clip=True, preserve_range=True)
                patch = np.expand_dims(patch, axis=2)
                patches.append(patch)
            except:
                continue

    if not patches:
        print('{} s'.format(time.time() - start_time))
        return 0

    patches = np.array(patches, dtype=np.float32)
    task = {
        'config_num': config_num,
        'study_id': study_id,
        'id': uuid1(),
        'type': 'cancer',
        'input_data': patches,
        'keys': cancer_model_keys,
    }
    redis_client.rpush('tasks', pickle.dumps(task))
    has_result = False
    while not has_result:
        if redis_client.hexists('results', task['id']):
            cancer_preds = pickle.loads(redis_client.hget('results', task['id']))
            redis_client.hdel('results', task['id'])
            has_result = True
        else:
            time.sleep(0.1)

    (bz_o, by_o, bx_o), (bz_i, by_i, bx_i), v_o, v_i = [thresh_group_bounds[i] for i in nodule_indices][0]
    z_min_norm = np.min([z for z, y_min, y_max, x_min, x_max in thresh_group_bboxes[0]]) / isometric_volume.shape[0]
    z_max_norm = np.max([z for z, y_min, y_max, x_min, x_max in thresh_group_bboxes[0]]) / isometric_volume.shape[0]
    z_mid_norm = (z_min_norm + z_max_norm) / 2
    y_min_norm = np.min([y_min for z, y_min, y_max, x_min, x_max in thresh_group_bboxes[0]]) / isometric_volume.shape[1]
    y_max_norm = np.max([y_max for z, y_min, y_max, x_min, x_max in thresh_group_bboxes[0]]) / isometric_volume.shape[1]
    y_mid_norm = (y_min_norm + y_max_norm) / 2
    x_min_norm = np.min([x_min for z, y_min, y_max, x_min, x_max in thresh_group_bboxes[0]]) / isometric_volume.shape[2]
    x_max_norm = np.max([x_max for z, y_min, y_max, x_min, x_max in thresh_group_bboxes[0]]) / isometric_volume.shape[2]
    x_mid_norm = (x_min_norm + x_max_norm) / 2
    features = (
        bz_o, by_o, bx_o, bz_i, by_i, bx_i, v_o, v_i,
        nb_groups, len(nodule_indices),
        z_min_norm, z_max_norm, z_mid_norm, y_min_norm, y_max_norm, y_mid_norm, x_min_norm, x_max_norm, x_mid_norm
    )
    with open(os.path.join(out_dir, '{}.pkl'.format(study_id)), 'wb') as f:
        pickle.dump(features, f)

    print('{} s'.format(time.time() - start_time))
    if middle_slices < 100:
        slice_start = int(((100-middle_slices)/200)*len(cancer_preds))
        slice_end = int(((100+middle_slices)/200)*len(cancer_preds))
        if slice_end == slice_start:
            slice_end += 1
        cancer_preds_slices = cancer_preds[slice_start:slice_end]
    else:
        cancer_preds_slices = cancer_preds
    if aggreg_func == 'median':
        return np.median(cancer_preds_slices)
    elif aggreg_func == 'mean':
        return np.mean(cancer_preds_slices)
    elif aggreg_func == 'gmean':
        return gmean(cancer_preds_slices)
    elif aggreg_func == 'max':
        return np.max(cancer_preds_slices)
    elif aggreg_func.startswith('maxmin'):
        _, aggreg_func_maxmin, aggreg_func_maxmin_thresh = aggreg_func.split(',')
        if aggreg_func_maxmin == 'mean':
            aggreg_func_maxmin_value = np.mean(cancer_preds_slices)
        elif aggreg_func_maxmin == 'gmean':
            aggreg_func_maxmin_value = gmean(cancer_preds_slices)
        elif aggreg_func_maxmin == 'median':
            aggreg_func_maxmin_value = np.median(cancer_preds_slices)
        else:
            raise ValueError('Invalid aggreg_func_maxmin')
        if aggreg_func_maxmin_value > float(aggreg_func_maxmin_thresh):
            return np.max(cancer_preds_slices)
        else:
            return np.min(cancer_preds_slices)
    else:
        raise ValueError('Invalid aggreg_func')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['stage1', 'stage2', 'sample'], default='sample')
    parser.add_argument('--config', type=int, default=1)
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

    # keep track of running config preds in redis sets
    redis_client.sadd('processing', args.config)

    out_dir_pred = os.path.join(BASEPATH, SETTINGS['PREDICTIONS_PATH'])
    os.makedirs(out_dir_pred, exist_ok=True)
    out_filepath_pred = os.path.join(out_dir_pred, 'preds_{}.csv'.format(args.config))

    with open(out_filepath_pred, 'w') as f:
        f.write('id,cancer\n')

    probmaps_dir = os.path.join(
        BASEPATH,
        'data_proc/{}/nodule_detect_probmaps_{}'.format(args.dataset, configs[args.config][0])
    )
    isometric_volumes_dir = os.path.join(
        BASEPATH,
        'data_proc/{}/isotropic_volumes_1mm'.format(args.dataset)
    )
    out_dir = os.path.join(BASEPATH, 'data_proc/{}/features/config_{}'.format(args.dataset, args.config))
    os.makedirs(out_dir, exist_ok=True)

    # write to predictions file
    for study_id in study_ids:
        cancer_pred = process_study(study_id, probmaps_dir, isometric_volumes_dir, out_dir, configs[args.config], args.config)
        with open(out_filepath_pred, 'a') as f:
            f.write('{},{}\n'.format(study_id, cancer_pred))

    # write to features file
    feat_df = pd.DataFrame({'id': study_ids}, index=study_ids)
    feat_keys = ['bz_o', 'by_o', 'bx_o', 'bz_i', 'by_i', 'bx_i', 'v_o', 'v_i', 'nod_pre', 'nod_post',
                 'z_min', 'z_max', 'z_mid', 'y_min', 'y_max', 'y_mid', 'x_min', 'x_max', 'x_mid']
    for key in feat_keys:
        feat_df[key] = np.zeros((len(study_ids),))
    for study_id in study_ids:
        try:
            with open(os.path.join(out_dir, '{}.pkl'.format(study_id)), 'rb') as f:
                study_feats = pickle.load(f)
        except:
            study_feats = [0] * len(feat_keys)
        for key, val in zip(feat_keys, study_feats):
            feat_df.ix[study_id, key] = val
    out_file_path = os.path.join(BASEPATH, 'data_proc/{}/features/config_{}_features.csv'.format(args.dataset, args.config))
    feat_df.to_csv(out_file_path, index=False)

    # add/remove from redis sets
    redis_client.srem('processing', args.config)
    redis_client.sadd('finished', args.config)
