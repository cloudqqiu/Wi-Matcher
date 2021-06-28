import sys, os
import time
import warnings
import deepmatcher as dm
import pandas as pd
import torch
print(torch.cuda.is_available())

ex_path = './src/experiment'


def deepmatch(pois, refolds=False, folds=10):
    # if refolds:
    #     print('Make or remake data folds')
    #     dps.pro4deepmatcher_folds(pois, folds=folds)

    start_time = time.time()

    total_tp, total_fp, total_fn = 0, 0, 0
    for i in range(folds):
        print('Fold:', i + 1)
        train, validation, test = dm.data.process(
            path='{}/linking/matching/deepmatcher'.format(ex_path),
            train='dm_train_{}.csv'.format(i),
            validation='dm_val_{}.csv'.format(i),
            test='dm_test_{}.csv'.format(i),
            cache='m_dp_cache.pth',
            ignore_columns=['wifi', 'match'],  # , 'ltable_name', 'rtable_name'],
            left_prefix='ltable_',
            right_prefix='rtable_',
            id_attr='index',
            label_attr='label',
            embeddings='glove.42B.300d',
            pca=False
        )
            # use_magellan_convention=True)

        m = 'hybrid'  # 'attention'
        print('create {} model'.format(m))
        model = dm.MatchingModel(attr_summarizer=m)
        # model = dm.MatchingModel(attr_summarizer=dm.attr_summarizers.Hybrid(word_contextualizer='lstm', word_aggregator='max-pool'), attr_comparator='concat')
        # model = model.cuda()
        print('start training')
        model.run_train(
            train,
            validation,
            epochs=10,
            batch_size=64,
            best_save_path='{}_model.pth'.format(m),
            pos_neg_ratio=3)

        # f1_this = model.run_eval(test)
        # f1_list.append(f1_this)
        # print('{}: {}'.format(i, f1_this))
        predictions = model.run_prediction(test)
        origin_table = pd.read_csv('{}/linking/matching/deepmatcher/dm_test_{}.csv'.format(ex_path, i))
        tp, fp, fn = 0, 0, 0
        for i in range(len(predictions)):
            # this = origin_table[origin_table['id'] == predictions.iloc[i]['id']]
            # if len(this) < 1:
            #     warnings.warn('id {} not found!'.format(predictions.iloc[i]['id']))
            this = origin_table.iloc[i]
            if predictions.iloc[i]['match_score'] > 0.5:
                if this['label'] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if this['label'] == 1:
                    fn += 1
        print(tp, fp, fn)
        try:
            print(tp / (tp + fp), tp / (tp + fn))
        except Exception as e:
            print(e)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    total_tp /= folds
    total_fp /= folds
    total_fn /= folds
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('P:', precision)
    print('R:', recall)
    print('F1:', f1)

    end_time = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    print('time consume: {} mins'.format((end_time - start_time) / 60))


if __name__ == '__main__':
    # magellan(['39.92451,116.51533', '39.93483,116.45241'])
    # deepmatcher(['39.92451,116.51533', '39.93483,116.45241'], refolds=False, folds=10)
    # dm_test()

    pois = ['39.92451,116.51533', '39.93483,116.45241',  # 这两个是第一批
            '39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
            '39.96333,116.45187', '39.98850,116.41674', '40.00034,116.46960']

    # magellan(pois)
    deepmatch(pois, refolds=False, folds=10)
