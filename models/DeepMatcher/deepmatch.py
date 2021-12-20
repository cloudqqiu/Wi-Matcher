import argparse
import time
from timeit import default_timer as timer
import sys
import os
sys.path.append(os.getcwd()+"/../..")

import models.DeepMatcher.deepmatcher as dm
import models.common.utils as utils
import models.model_config as config

def deepmatch(data, folds=5):
    timer_log = utils.Timer()
    timer_log.start()
    f1_list = list()
    precision_list = []
    recall_list = []

    start_time_data_preprocess = timer()

    if data == 'zh':
        print('USE Chinese DATASET')
        data_path = config.zh_deepmatcher_data_path
        ignore_columns = ['wifi', 'match']#, 'ltable_name', 'rtable_name']
        left_prefix = 'ltable_'
        right_prefix = 'rtable_'
    elif data == 'ru':
        print('USE Russian DATASET')
        data_path = config.ru_deepmatcher_data_path
        ignore_columns = ['ssid_id', 'venue_id']
        left_prefix = 'ssid_'
        right_prefix = 'venue_'

    end_time_data_preprocess = timer()
    print(f"# Timer - Data Preprocess - {end_time_data_preprocess - start_time_data_preprocess}")

    for i in range(folds):
        timer_log.log(f'DeepMatcher Fold {i} start')
        train, validation, test = dm.data.process(
            path=data_path,
            train='dm_train_{}.csv'.format(i),
            validation='dm_val_{}.csv'.format(i),
            test='dm_test_{}.csv'.format(i),
            # cache='m_dp_cache.pth',
            ignore_columns=ignore_columns,
            left_prefix=left_prefix,
            right_prefix=right_prefix,
            id_attr='index',
            label_attr='label',
            embeddings='glove.42B.300d',
            pca=False
        )
        # use_magellan_convention=True)

        m = 'hybrid'  # 'attention'
        # model = dm.MatchingModel(attr_summarizer=m)
        model = dm.MatchingModel(
            attr_summarizer=dm.attr_summarizers.Hybrid(word_contextualizer='gru',
                                                       word_aggregator='max-pool'
                                                       ),
            attr_comparator='abs-diff')

        timer_log.log(f'Data and Model load')

        start_time_model_train = timer()

        model.run_train(
            train,
            validation,
            epochs=10,
            batch_size=64,
            best_save_path='{}_model.pth'.format(m),
            pos_neg_ratio=5)

        end_time_model_train = timer()
        print(f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

        timer_log.log(f'DeepMatcher Train')

        # f1_this, p, r = model.run_eval(test)

        start_time_model_predict = timer()

        f1_this, p, r = model.run_eval(test)

        end_time_model_predict = timer()
        print(f"# Timer - Model Predict - Samples: {len(test)} - {end_time_model_predict - start_time_model_predict}")

        # f1_this, p, r = 0 ,0,0
        timer_log.log(f'DeepMatcher Prediction')

        f1_list.append(f1_this)
        precision_list.append(p)
        recall_list.append(r)
        print('{}: {}'.format(i, p))
        print('{}: {}'.format(i, r))
        print('{}: {}'.format(i, f1_this))

    timer_log.stop(f'DeepMatcher Finished')

    print('average precision: {}'.format(sum(precision_list[:]) / folds))
    print('average recall: {}'.format(sum(recall_list[:]) / folds))
    print('average f1: {}'.format(sum(f1_list[:]) / folds))
    return sum(precision_list[:]) / folds, sum(recall_list[:]) / folds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepMatcher model")
    parser.add_argument('--dataset', '-d', required=True, choices=['ru', 'zh'], help='Dataset zh or ru')
    parser.add_argument('--repeat_run', type=int, default=5)
    args = parser.parse_args()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    times = args.repeat_run
    _p, _r = 0, 0
    for i in range(times):
        temp_p, temp_r = deepmatch(data=args.dataset)

        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print(f"Run 5-Fold for {times} times ")
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

