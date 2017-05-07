import os
import pickle
import random
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import classification_report, roc_auc_score, log_loss, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost as xgb
import hyperopt

stage1_labels = pd.read_csv('../data/stage1_labels.csv').sort_values(by='id')
stage1_solution = pd.read_csv('../data/stage1_solution.csv').sort_values(by='id')
del stage1_solution['Usage']
stage1_sample_submission = pd.read_csv('../data/stage1_sample_submission.csv').sort_values(by='id')
study_sex_det = pd.read_csv('../predictions/study_sex_det.csv').sort_values(by='id')

config_nums = sorted(list(range(1, 751)))

preds = {}
for i in config_nums:
    preds[i] = pd.read_csv('../predictions/preds_{}.csv'.format(i)).sort_values(by='id')

features = {}
for i in config_nums:
    features[i] = pd.read_csv('../data_proc/stage1/features/config_{}_features.csv'.format(i))

study_features_df = study_sex_det.copy()
study_features_df['sex'] = study_features_df['sex'].apply(lambda x: (1 if x == 'F' else 0))
for i in config_nums:
    study_features_df = pd.merge(study_features_df, preds[i], how='inner', on='id', sort=True,
                                 suffixes=('', '_{}'.format(i)), copy=True)
    study_features_df = pd.merge(study_features_df, features[i], how='inner', on='id', sort=True,
                                 suffixes=('', '_{}'.format(i)), copy=True)

Xy_meta_train_df = pd.concat([stage1_labels, stage1_solution]).sort_values(by='id')
Xy_meta_train_df.columns = ['id', 'y_true']
Xy_meta_train_df = pd.merge(Xy_meta_train_df, study_features_df, how='inner', on='id', sort=True, copy=True)
y_meta_train = Xy_meta_train_df['y_true'].as_matrix().astype(np.bool)
del Xy_meta_train_df['id']
del Xy_meta_train_df['y_true']
X_meta_train = Xy_meta_train_df.as_matrix().astype(np.float32)
print('train:', X_meta_train.shape, y_meta_train.shape)


def run(folds):
    print('')
    xgb_best_params_list = []

    k_fold = KFold(folds, shuffle=True, random_state=9999)
    for k, (sp0_indices, sp1_indices) in enumerate(k_fold.split(X_meta_train)):
        print('fold:', k)
        X_meta_train_sp0 = X_meta_train[sp0_indices]
        y_meta_train_sp0 = y_meta_train[sp0_indices]
        X_meta_train_sp1 = X_meta_train[sp1_indices]
        y_meta_train_sp1 = y_meta_train[sp1_indices]

        def xgb_objective(space):
            model = xgb.XGBClassifier(
                max_depth=int(space['max_depth']),
                n_estimators=int(space['n_estimators']),
                subsample=space['subsample'],
                colsample_bytree=space['colsample_bytree'],
                learning_rate=space['learning_rate'],
                min_child_weight=int(space['min_child_weight'])
            )
            model.fit(X_meta_train_sp0, y_meta_train_sp0,
                      eval_set=[(X_meta_train_sp1, y_meta_train_sp1)], eval_metric='logloss',
                      early_stopping_rounds=50, verbose=False)
            y_pred_train_sp1 = model.predict_proba(
                X_meta_train_sp1,
                ntree_limit=model.best_ntree_limit
            )[:, 1]
            loss = log_loss(y_meta_train_sp1, y_pred_train_sp1, eps=1e-15)
            print(round(loss, 3), end=' ', flush=True)
            return {'loss': loss, 'status': hyperopt.STATUS_OK}

        xgb_trials_space = {
            'max_depth': hyperopt.hp.quniform('max_depth', 2, 25, 1),
            'n_estimators': hyperopt.hp.quniform('n_estimators', 100, 1000, 1),
            'subsample': hyperopt.hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.1, 1),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 0.003, 0.1),
            'min_child_weight': hyperopt.hp.quniform('min_child_weight', 1, 10, 1),
        }

        xgb_trials = hyperopt.Trials()
        xgb_best_params = hyperopt.fmin(fn=xgb_objective, space=xgb_trials_space, trials=xgb_trials,
                                        algo=hyperopt.tpe.suggest, max_evals=500)

        xgb_best_params_list.append(xgb_best_params)
        print('')

    print('')
    meta_model_kfold_list = []

    k_fold = KFold(folds, shuffle=True, random_state=9999)
    for k, (sp0_indices, sp1_indices) in enumerate(k_fold.split(X_meta_train)):
        X_meta_train_sp0 = X_meta_train[sp0_indices]
        y_meta_train_sp0 = y_meta_train[sp0_indices]
        X_meta_train_sp1 = X_meta_train[sp1_indices]
        y_meta_train_sp1 = y_meta_train[sp1_indices]

        xgb_params = {
            'max_depth': int(xgb_best_params_list[k]['max_depth']),
            'n_estimators': int(xgb_best_params_list[k]['n_estimators']),
            'subsample': xgb_best_params_list[k]['subsample'],
            'colsample_bytree': xgb_best_params_list[k]['colsample_bytree'],
            'learning_rate': xgb_best_params_list[k]['learning_rate'],
            'min_child_weight': int(xgb_best_params_list[k]['min_child_weight']),
        }

        meta_model = xgb.XGBClassifier(**xgb_params)
        meta_model.fit(X_meta_train_sp0, y_meta_train_sp0,
                       eval_set=[(X_meta_train_sp1, y_meta_train_sp1)],
                       eval_metric='logloss', early_stopping_rounds=50, verbose=False)
        print('[{}][{}] best epoch: {} val loss: {}'.format(i, k, meta_model.best_iteration, meta_model.best_score))
        meta_model_kfold_list.append(meta_model)

    meta_model_kfold_list_name = 'ens_1_750_meta_model_{}fold_retrained_list'.format(folds)
    os.makedirs('../weights/stage1/cancer_pred_meta_cv', exist_ok=True)
    with open('../weights/stage1/cancer_pred_meta_cv/{}.pkl'.format(meta_model_kfold_list_name), 'wb') as f:
        pickle.dump(meta_model_kfold_list, f)

    print('')
    metrics_columns = ['fold', 'logloss', 'rocauc', 'prec(-)', 'rec(-)', 'prec(+)', 'rec(+)', 'mic-f1', 'mac-f1']
    print('\t'.join(metrics_columns))
    metrics_vals_list = []
    k_fold = KFold(folds, shuffle=True, random_state=9999)
    for k, (sp0_indices, sp1_indices) in enumerate(k_fold.split(X_meta_train)):
        X_meta_train_sp0 = X_meta_train[sp0_indices]
        y_meta_train_sp0 = y_meta_train[sp0_indices]
        X_meta_train_sp1 = X_meta_train[sp1_indices]
        y_meta_train_sp1 = y_meta_train[sp1_indices]

        meta_model = meta_model_kfold_list[k]
        y_pred_train_sp1 = meta_model.predict_proba(
            X_meta_train_sp1,
            ntree_limit=meta_model.best_ntree_limit
        )[:, 1]

        metrics_vals = [
            log_loss(y_meta_train_sp1, y_pred_train_sp1, eps=1e-15),
            roc_auc_score(y_meta_train_sp1, y_pred_train_sp1),
            precision_score(y_meta_train_sp1, np.round(y_pred_train_sp1), pos_label=0, average='binary'),
            recall_score(y_meta_train_sp1, np.round(y_pred_train_sp1), pos_label=0, average='binary'),
            precision_score(y_meta_train_sp1, np.round(y_pred_train_sp1), pos_label=1, average='binary'),
            recall_score(y_meta_train_sp1, np.round(y_pred_train_sp1), pos_label=1, average='binary'),
            f1_score(y_meta_train_sp1, np.round(y_pred_train_sp1), average='micro'),
            f1_score(y_meta_train_sp1, np.round(y_pred_train_sp1), average='macro'),
        ]
        metrics_vals_list.append(metrics_vals)
        print('\t'.join(['[{}]'.format(k)] + ['{:.3f}'.format(x) for x in metrics_vals]))

    print('')
    metrics_vals_kfold_avg = [np.mean([metrics_vals_list[k][i] for k in range(folds)]) for i in range(8)]
    metrics_vals_kfold_std = [np.std([metrics_vals_list[k][i] for k in range(folds)]) for i in range(8)]
    metrics_columns = ['logloss', 'rocauc', 'prec(-)', 'rec(-)', 'prec(+)', 'rec(+)', 'mic-f1', 'mac-f1']
    print('\t'.join(metrics_columns))
    print('\t'.join(['{:.3f}'.format(x) for x in metrics_vals_kfold_avg]))
    print('\t'.join(['{:.3f}'.format(x) for x in metrics_vals_kfold_std]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=4)
    args = parser.parse_args()

    run(args.folds)
