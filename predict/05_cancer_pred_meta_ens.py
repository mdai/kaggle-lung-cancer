import os
import pickle
import random
import numpy as np
import pandas as pd
import argparse
import json
from sklearn.metrics import classification_report, roc_auc_score, log_loss, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost as xgb
import hyperopt


def make_final_preds(model_path, predictions_path, features_path, sample_submission_file_path, submission_file_path):
    with open(model_path, 'rb') as f:
        meta_model_kfold_list = pickle.load(f)

    sample_submission = pd.read_csv(sample_submission_file_path).sort_values(by='id')
    study_sex_det = pd.read_csv(os.path.join(predictions_path, 'study_sex_det.csv')).sort_values(by='id')

    config_nums = sorted(list(range(1, 751)))

    preds = {}
    for i in config_nums:
        preds[i] = pd.read_csv(os.path.join(predictions_path, 'preds_{}.csv'.format(i))).sort_values(by='id')

    features = {}
    for i in config_nums:
        features[i] = pd.read_csv(os.path.join(features_path, 'config_{}_features.csv'.format(i))).sort_values(by='id')

    study_features_df = study_sex_det.copy()
    study_features_df['sex'] = study_features_df['sex'].apply(lambda x: (1 if x == 'F' else 0))
    for i in config_nums:
        study_features_df = pd.merge(study_features_df, preds[i], how='inner', on='id', sort=True,
                                     suffixes=('', '_{}'.format(i)), copy=True)
        study_features_df = pd.merge(study_features_df, features[i], how='inner', on='id', sort=True,
                                     suffixes=('', '_{}'.format(i)), copy=True)

    X_meta_test_df = sample_submission.copy()
    del X_meta_test_df['cancer']
    X_meta_test_df = pd.merge(X_meta_test_df, study_features_df, how='inner', on='id', sort=True, copy=True)
    del X_meta_test_df['id']
    X_meta_test = X_meta_test_df.as_matrix().astype(np.float32)
    print('test:', X_meta_test.shape)

    y_pred_list = []
    for idx in range(len(meta_model_kfold_list)):
        meta_model = meta_model_kfold_list[idx]
        y_pred_list.append(meta_model.predict_proba(
            X_meta_test,
            ntree_limit=meta_model.best_ntree_limit
        )[:, 1])
    y_pred_comb_test = np.mean(np.vstack(y_pred_list), axis=0)
    cancer_preds_test = sample_submission.copy().sort_values(by='id')
    cancer_preds_test['cancer'] = y_pred_comb_test
    cancer_preds_test.to_csv(submission_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['stage1', 'stage2', 'sample'], default='sample')
    parser.add_argument('--submission', choices=[1, 2], type=int, default=1)
    args = parser.parse_args()

    SETTINGS_FILE_PATH = '../SETTINGS.json'
    BASEPATH = os.path.dirname(os.path.abspath(SETTINGS_FILE_PATH))

    with open(SETTINGS_FILE_PATH, 'r') as f:
        SETTINGS = json.load(f)

    if args.dataset == 'stage1':
        sample_submission_file_path = os.path.join(BASEPATH, SETTINGS['STAGE1_SAMPLE_SUBMISSION_FILE_PATH'])
    elif args.dataset == 'stage2':
        sample_submission_file_path = os.path.join(BASEPATH, SETTINGS['STAGE2_SAMPLE_SUBMISSION_FILE_PATH'])
    elif args.dataset == 'sample':
        sample_submission_file_path = os.path.join(BASEPATH, SETTINGS['SAMPLE_SUBMISSION_FILE_PATH'])
    else:
        raise ValueError('Invalid --dataset, must be stage1 or stage2 or sample.')

    # submission 1 uses the final model trained on the stage1 training set
    # submission 2 uses the final model retrained using the entire stage1 data
    if args.submission == 1:
        meta_model_kfold_list_name = 'ens_1_750_meta_model_4fold_list'
    elif args.submission == 2:
        meta_model_kfold_list_name = 'ens_1_750_meta_model_4fold_retrained_list'
    else:
        raise ValueError('Invalid --submission, must be 1 or 2.')

    model_path = os.path.join(BASEPATH, SETTINGS['MODEL_WEIGHTS_PATH'], '{}.pkl'.format(meta_model_kfold_list_name))
    predictions_path = os.path.join(BASEPATH, SETTINGS['PREDICTIONS_PATH'])
    features_path = os.path.join(BASEPATH, 'data_proc/{}/features'.format(args.dataset))

    out_dir = os.path.join(BASEPATH, SETTINGS['SUBMISSIONS_PATH'])
    os.makedirs(out_dir, exist_ok=True)
    submission_file_path = os.path.join(out_dir, 'submission_{}.csv'.format(args.submission))

    make_final_preds(model_path, predictions_path, features_path, sample_submission_file_path, submission_file_path)
