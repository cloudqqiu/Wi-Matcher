import argparse
import time
import os
import sys
sys.path.append(os.getcwd()+"/../..")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer

import models.model_config as configs
import models.common.utils as utils

def yandexmatcher_gbdt(data, use_lbs=False, folds=5, plot=False):

    time_log = utils.Timer()

    if data == 'zh':
        data_path = configs.zh_yandexmatcher_data_path
    elif data == 'ru':
        data_path = configs.ru_yandexmatcher_data_path

    time_log.start()
    start_time_data_preprocess = timer()
    try:
        filename = f'{data_path}/dataset_lbs.csv' if use_lbs else f'{data_path}/dataset_no_lbs.csv'
        df = pd.read_csv(filename)
        print('Read file', filename)
    except Exception as e:
        import warnings
        warnings.warn(e.__str__())

    features = df.columns.tolist()
    f_features = np.where(df.dtypes == np.float64)[0].tolist()
    f_f_name = [features[i] for i in f_features]
    df[f_f_name] = df[f_f_name].astype('str')
    # print(df.info())
    features.remove('Unnamed: 0')
    features.remove('label')
    features.remove('name')  ##
    features.remove('ssid')  ##
    new_df = df.drop(['label', 'name', 'ssid', 'Unnamed: 0'], axis=1) # name ssid

    end_time_data_preprocess = timer()
    print(f"# Timer - Data Preprocess - {end_time_data_preprocess - start_time_data_preprocess}")

    time_log.log("Data Load and Feature Generation")

    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    all_pred_result = list()
    for index, (train, test) in enumerate(skf.split(df, df['label'])):
        time_log.log(f"YandexMatcher Fold {index}")
        print('Fold: ', index)
        # print(train, test)

        skf_v = StratifiedKFold(n_splits=folds - 1, shuffle=True)
        trainwv, valwv = next(skf_v.split(df.iloc[train], df.iloc[train]['label']))

        model = CatBoostClassifier(iterations=2000, depth=9, cat_features=features, l2_leaf_reg=3,
                                   early_stopping_rounds=5,
                                   eval_metric='Logloss',
                                   learning_rate=0.05,
                                   loss_function='Logloss')
        # model.fit(new_df.iloc[train], df.iloc[train]['label'], plot=True)
        start_time_model_train = timer()

        model.fit(new_df.iloc[train].iloc[trainwv], df.iloc[train].iloc[trainwv]['label'], plot=True,
                  eval_set=(new_df.iloc[train].iloc[valwv], df.iloc[train].iloc[valwv]['label']),
                  )
        # print(model.tree_count_)
        end_time_model_train = timer()
        print(f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

        time_log.log(f'YandexMatcher Fold {index} Model Training')

        # pred = model.predict(df.iloc[test], df.iloc[test]['label'])
        start_time_model_predict = timer()

        preds_class = model.predict(new_df.iloc[test])
        end_time_model_predict = timer()
        print(f"# Timer - Model Predict - Samples: {len(new_df.iloc[test])} - {end_time_model_predict - start_time_model_predict}")

        # preds_proba = model.predict_proba(df.iloc[test])
        print("class=", preds_class)

        time_log.log(f'YandexMatcher Fold {index} Prediction')

        # print("proba=", preds_proba)
        tp, fp, fn = 0, 0, 0
        test_data = df.iloc[test]
        for ind, i in enumerate(preds_class):
            # print(df.iloc[test].iloc[ind]['name'], df.iloc[test].iloc[ind]['ssid'],  i, df.iloc[test].iloc[ind]['label'])
            if i == 1:
                if test_data.iloc[ind]['label'] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if test_data.iloc[ind]['label'] == 1:
                    fn += 1
        print(tp, fp, fn)
        print(tp / (tp + fp), tp / (tp + fn))
        all_pred_result.append([tp, fp, fn])

        if plot:
            import matplotlib.pyplot as plt
            fea_ = model.feature_importances_
            fea_name = model.feature_names_
            plt.figure(figsize=(10, 10))
            plt.barh(fea_name, fea_, height=0.5)
            plt.show()

    time_log.stop(f'YandexMatcher {folds} Folds Finished.')


    print(all_pred_result)
    tp, fp, fn = 0, 0, 0
    for result in all_pred_result:
        tp += result[0]
        fp += result[1]
        fn += result[2]
    tp /= len(all_pred_result)
    fp /= len(all_pred_result)
    fn /= len(all_pred_result)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)
    return precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Magellan model")
    parser.add_argument('--dataset', '-d', required=True, choices=['ru', 'zh'], help='Dataset zh or ru')
    parser.add_argument('--use_lbs', action='store_true')
    parser.add_argument('--repeat_run', type=int, default=5)
    args = parser.parse_args()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    times = args.repeat_run
    _p, _r = 0, 0
    for i in range(times):
        temp_p, temp_r = yandexmatcher_gbdt(data=args.dataset, use_lbs=args.use_lbs, folds=5, plot=False)
        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print(f"Run 5-Fold for {times} times ")
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
