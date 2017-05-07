import os
import pickle
import json
import pydicom

SETTINGS_FILE_PATH = '../SETTINGS.json'
BASEPATH = os.path.dirname(os.path.abspath(SETTINGS_FILE_PATH))

with open(SETTINGS_FILE_PATH, 'r') as f:
    SETTINGS = json.load(f)

with open(os.path.join(BASEPATH, SETTINGS['ANNOTATIONS_PATH'], 'data_20170219.pkl'), 'rb') as f:
    (labels_dict, annotations_dict, labels_applied, instance_id_to_filename_dict) = pickle.load(f)

PROJECT_ID = '0acb9bc6-fb54-4279-890e-e6922c0df9c2'

studies_train = pd.read_csv('../data/stage1_labels.csv').sort_values(by='id')['id'].tolist()


def get_files(root):
    """Yields all file paths recursively from root filepath.
    """
    for item in os.scandir(root):
        if item.is_file():
            yield item.path
        elif item.is_dir():
            yield from get_files(item.path)


def get_study_filenames_sorted(study_id):
    study_root_path = os.path.join(BASEPATH, SETTINGS['STAGE1_DATA_PATH'], study_id)
    instance_filepaths = sorted(list(get_files(study_root_path)))
    slices = [pydicom.read_file(fp) for fp in instance_filepaths]
    indices_sorted = [tup[0] for tup in sorted(enumerate(slices), key=lambda tup: int(tup[1].ImagePositionPatient[2]))]
    return [instance_filepaths[i].replace(s.path.join(BASEPATH, SETTINGS['STAGE1_DATA_PATH']), '') for i in indices_sorted]


study_filenames_sorted = {}
for study_id in studies_train:
    study_filenames_sorted[study_id] = get_study_filenames_sorted(study_id)


def create_annotations_by_study_id():
    annotations_by_study_id = {}

    for instance_id, instance_labels_applied in labels_applied['instance'].items():
        for applied_label in instance_labels_applied:
            if applied_label['labelId'] != 4:
                # select nodule labels only
                continue
            anno_id = '{}/{}/{}/{}'.format(
                PROJECT_ID, instance_id, applied_label['labelId'], applied_label['appliedLabelNumber']
            )
            try:
                anno = annotations_dict[anno_id]
                filename = instance_id_to_filename_dict[anno['instanceId']]
                study_id = filename.split('/')[0]
                if study_id in annotations_by_study_id:
                    annotations_by_study_id[study_id].append(anno)
                else:
                    annotations_by_study_id[study_id] = [anno]
            except:
                pass

    return annotations_by_study_id


def is_overlapping(anno1, anno2):
    if anno1['data']['x'] < anno2['data']['x'] and anno2['data']['x'] > anno1['data']['x'] + anno1['data']['width']:
        return False
    elif anno2['data']['x'] < anno1['data']['x'] and anno1['data']['x'] > anno2['data']['x'] + anno2['data']['width']:
        return False
    if anno1['data']['y'] < anno2['data']['y'] and anno2['data']['y'] > anno1['data']['y'] + anno1['data']['height']:
        return False
    elif anno2['data']['y'] < anno1['data']['y'] and anno1['data']['y'] > anno2['data']['y'] + anno2['data']['height']:
        return False
    return True


def is_part_of_group(annotations_group, anno):
    last_anno_in_group = annotations_group[-1]
    return is_overlapping(last_anno_in_group, anno) and last_anno_in_group['sliceNum'] == anno['sliceNum'] - 1


def create_study_annotations_grouped():
    annotations_by_study_id = create_annotations_by_study_id()

    study_annotations_grouped = {}

    for i, study_id in enumerate(list(studies_train)):
        filenames_sorted = study_filenames_sorted[study_id]
        try:
            study_annotations = annotations_by_study_id[study_id]
        except:
            continue

        for anno in study_annotations:
            filename = instance_id_to_filename_dict[anno['instanceId']]
            slice_num = filenames_sorted.index(filename)
            anno['sliceNum'] = slice_num
        study_annotations = sorted(study_annotations, key=lambda anno: anno['sliceNum'])
        slice_nums = [anno['sliceNum'] for anno in study_annotations]

        # group into consecutive overlapping slices
        annotations_grouped = []
        for n in range(min(slice_nums), max(slice_nums) + 1):
            annotations_current_slice = [anno for anno in study_annotations if anno['sliceNum'] == n]
            if len(annotations_grouped) == 0:
                [annotations_grouped.append([anno]) for anno in annotations_current_slice]
            else:
                for anno in annotations_current_slice:
                    joined_existing_group = False
                    for annotations_group in annotations_grouped:
                        if is_part_of_group(annotations_group, anno):
                            annotations_group.append(anno)
                            joined_existing_group = True
                    if not joined_existing_group:
                        annotations_grouped.append([anno])

        study_annotations_grouped[study_id] = annotations_grouped

    return study_annotations_grouped


annotations_by_study_id = create_annotations_by_study_id()
study_annotations_grouped = create_study_annotations_grouped()

for study_id in list(studies_train):
    if study_id not in annotations_by_study_id:
        annotations_by_study_id[study_id] = []
    if study_id not in study_annotations_grouped:
        study_annotations_grouped[study_id] = []

with open(os.path.join(BASEPATH, SETTINGS['ANNOTATIONS_PATH'], 'annotations_by_study_id.pkl'), 'wb') as f:
    pickle.dump(annotations_by_study_id, f)
with open(os.path.join(BASEPATH, SETTINGS['ANNOTATIONS_PATH'], 'study_annotations_grouped.pkl'), 'wb') as f:
    pickle.dump(study_annotations_grouped, f)
