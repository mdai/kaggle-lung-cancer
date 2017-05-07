import sys
import os
import pickle
import redis
import numpy as np
import time
import traceback
import argparse
import json

redis_client = redis.StrictRedis()

cwd = os.path.split(os.getcwd())[0]
if cwd not in sys.path:
    sys.path.append(cwd)

# Nodule models

from models.nodule.m10a import define_model as define_m10a

nodule_model_keys_all = ['m10a']
nodule_models = {}
for key in nodule_model_keys_all:
    nodule_models[key] = locals()['define_{}'.format(key)]()

# Bounding box models

from models.bbox.m02a import define_model as define_m02a
from models.bbox.m04a import define_model as define_m04a

bbox_model_keys_all = ['m02a', 'm04a']
bbox_models = {}
for key in bbox_model_keys_all:
    bbox_models[key] = locals()['define_{}'.format(key)]()

# Cancer pred models

from models.cancer.resnet2d09d import define_model as define_resnet2d09d
from models.cancer.resnet2d09e import define_model as define_resnet2d09e
from models.cancer.resnet2d09f import define_model as define_resnet2d09f

cancer_model_keys_all = ['resnet2d09d', 'resnet2d09e', 'resnet2d09f']
cancer_models = {}
for key in cancer_model_keys_all:
    cancer_models[key] = locals()['define_{}'.format(key)]()


def run_nodule_model(input_data, nodule_model_key):
    nodule_preds = nodule_models[nodule_model_key].predict(input_data)[:, 0]
    return nodule_preds


def run_bbox_model(input_data, bbox_model_key):
    bbox_preds = bbox_models[bbox_model_key].predict(input_data)
    return bbox_preds


def run_cancer_model(input_data, cancer_model_keys):
    cancer_preds = np.zeros((input_data.shape[0],))
    for key in cancer_model_keys:
        cancer_preds += cancer_models[key].predict(input_data)[:, 0]
    cancer_preds /= len(cancer_model_keys)
    return cancer_preds


task_funcs = {
    'nodule': run_nodule_model,
    'bbox': run_bbox_model,
    'cancer': run_cancer_model,
}

if __name__ == '__main__':
    SETTINGS_FILE_PATH = '../SETTINGS.json'
    BASEPATH = os.path.dirname(os.path.abspath(SETTINGS_FILE_PATH))

    with open(SETTINGS_FILE_PATH, 'r') as f:
        SETTINGS = json.load(f)

    # load NN weights
    for key in nodule_model_keys_all:
        nodule_models[key].load_weights(os.path.join(BASEPATH, SETTINGS['MODEL_WEIGHTS_PATH'], '{}.hdf5'.format(key)))
    for key in bbox_model_keys_all:
        bbox_models[key].load_weights(os.path.join(BASEPATH, SETTINGS['MODEL_WEIGHTS_PATH'], '{}.hdf5'.format(key)))
    for key in cancer_model_keys_all:
        cancer_models[key].load_weights(os.path.join(BASEPATH, SETTINGS['MODEL_WEIGHTS_PATH'], '{}.hdf5'.format(key)))

    while True:
        _, val = redis_client.blpop('tasks')
        start_time = time.time()
        task = pickle.loads(val)
        print(task['config_num'], task['study_id'], task['id'], end=' ...', flush=True)
        try:
            output_data = task_funcs[task['type']](
                task['input_data'],
                task['keys']
            )
            redis_client.hset('results', task['id'], pickle.dumps(output_data))
        except Exception:
            traceback.print_exc()
        print('{} s'.format(time.time() - start_time))
