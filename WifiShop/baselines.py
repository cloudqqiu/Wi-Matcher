import sys
import os
import math

sys.path.append(os.path.abspath('..'))
import WifiShop.data_process as dps

import time
import numpy as np
from collections import Counter
import pandas as pd
import pro_func as pf


ex_path = '../src/experiment'


def VSM(mod='char', df=False, thres=0.5):
    assert mod == 'char' or mod == 'gram'
    print('VSM {}-model tf{} thres={}'.format(mod, '-idf' if df else '', thres))
    raw_data = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    c_wifi, c_shop = list(),list()
    c_frequent = dict()
    if df:
        df_set = set()
    for index, i in raw_data.iterrows():
        wifi, shop = i['ltable_pinyin'], i['rtable_pinyin']
        if df:
            df_set.add(wifi)
            df_set.add(shop)
        if mod == 'char':
            w_d, s_d = Counter(wifi), Counter(shop)
            for c in w_d.keys():
                w_d[c] /= len(wifi)
                if c_frequent.__contains__(c):
                    if wifi not in c_frequent[c]:
                        c_frequent[c].add(wifi)
                else:
                    c_frequent[c] = set([wifi])
            for c in s_d.keys():
                s_d[c] /= len(shop)
                if c_frequent.__contains__(c):
                    if wifi not in c_frequent[c]:
                        c_frequent[c].add(shop)
                else:
                    c_frequent[c] = set([shop])
        elif mod == 'gram':
            w, s = pf.get_ngram(wifi, 3, True), pf.get_ngram(shop, 3, True)
            w_d, s_d = Counter(w), Counter(s)
            for c in w_d.keys():
                w_d[c] /= len(w)
                if c_frequent.__contains__(c):
                    if wifi not in c_frequent[c]:
                        c_frequent[c].add(wifi)
                else:
                    c_frequent[c] = set([wifi])
            for c in s_d.keys():
                s_d[c] /= len(s)
                if c_frequent.__contains__(c):
                    if wifi not in c_frequent[c]:
                        c_frequent[c].add(shop)
                else:
                    c_frequent[c] = set([shop])
        c_wifi.append(w_d)
        c_shop.append(s_d)
    tp, fp, fn = 0, 0, 0
    for index, i in raw_data.iterrows():
        temp_sum, w_sum, s_sum = 0, 0, 0
        for j in c_wifi[index].keys():
            if df:
                c_wifi[index][j] *= math.log(len(df_set) / len(c_frequent[j]), 10)
            w_sum += math.pow(c_wifi[index][j], 2)
        for j in c_shop[index].keys():
            if df:
                c_shop[index][j] *= math.log(len(df_set) / len(c_frequent[j]), 10)
            s_sum += math.pow(c_shop[index][j], 2)
            if c_wifi[index].__contains__(j):
                temp_sum += c_wifi[index][j] * c_shop[index][j]
        temp_sum /= math.sqrt(w_sum) * math.sqrt(s_sum)
        if temp_sum > thres:
            if i['label'] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if i['label'] == 1:
                fn += 1
    print(tp, fp, fn)
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print('micro P:', precision)
        print('micro R:', recall)
        print('micro F1:', f1)
    except Exception as e:
        print(e)


def magellan_choose_model(H):
    import py_entitymatching as em
    print('Choose the best Matcher')

    dt = em.DTMatcher(name='DecisionTree', random_state=0)
    svm = em.SVMMatcher(name='SVM', random_state=0)
    rf = em.RFMatcher(name='RF', random_state=0)
    lg = em.LogRegMatcher(name='LogReg', random_state=0)
    ln = em.LinRegMatcher(name='LinReg')
    nb = em.NBMatcher(name='NaiveBayes')
    # xg = em.XGBoostMatcher(name='XGBoost')

    result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H,
                               exclude_attrs=['index', 'wifi', 'match', 'label'],
                               k=5,
                               target_attr='label', metric_to_select_matcher='precision', random_state=0)
    print(result['cv_stats'])
    return


def magellan_model_ana(H, L):
    import py_entitymatching as em
    print('Analyze Matcher')
    dt = em.DTMatcher(name='DecisionTree', random_state=0)
    svm = em.SVMMatcher(name='SVM', random_state=0)
    rf = em.RFMatcher(name='RF', random_state=0)
    lg = em.LogRegMatcher(name='LogReg', random_state=0)
    ln = em.LinRegMatcher(name='LinReg')
    nb = em.NBMatcher(name='NaiveBayes')
    for matcher in [dt, rf, svm, ln, lg, nb]:
        print(matcher)
        matcher.fit(table=H,
                    exclude_attrs=['index', 'wifi', 'match', 'label'],
                    target_attr='label')
        predictions = matcher.predict(table=L, exclude_attrs=['index', 'wifi', 'match', 'label'],
                                      append=True, target_attr='predicted', inplace=False)
        eval_result = em.eval_matches(predictions, 'label', 'predicted')
        em.print_eval_summary(eval_result)
    return


def magellan(pois):
    import py_entitymatching as em
    # t_wifi, t_shop, m_train, m_test = dps.pro4magellan_folds(pois)
    # t_wifi, t_shop, t_match = dps.pro4magellan_without_folds(pois, redata=False)
    # em.set_key(t_wifi, 'id')
    # em.set_key(t_shop, 'id')
    # em.set_key(t_match, 'index')
    # em.to_csv_metadata(t_wifi, '{}/linking/matching/magellan/magellan_wifi.csv'.format(ex_path))
    # em.to_csv_metadata(t_shop, '{}/linking/matching/magellan/magellan_shop.csv'.format(ex_path))
    # em.to_csv_metadata(t_match, '{}/linking/matching/magellan/magellan_match.csv'.format(ex_path))
    w = em.read_csv_metadata('{}/linking/matching/magellan/magellan_wifi.csv'.format(ex_path), key='id')
    s = em.read_csv_metadata('{}/linking/matching/magellan/magellan_shop.csv'.format(ex_path), key='id')
    m = em.read_csv_metadata('{}/linking/matching/magellan/magellan_match.csv'.format(ex_path), key='index')

    em.set_ltable(m, w)
    em.set_rtable(m, s)
    em.set_fk_ltable(m, 'wifi')
    em.set_fk_rtable(m, 'match')

    ite_k = 10
    # avg_pre, avg_rec = 0, 0
    tp, fp, fn = 0, 0, 0
    for i in range(ite_k):
        rng = np.random.RandomState(int(time.time()))
        rng.rand(4)
        IJ = em.split_train_test(m, train_proportion=0.7, random_state=rng)
        I = IJ['train']
        J = IJ['test']

        match_t = em.get_tokenizers_for_matching()
        match_s = em.get_sim_funs_for_matching()
        atypes1 = em.get_attr_types(w)  # don't need, if atypes1 exists from blocking step
        atypes2 = em.get_attr_types(s)  # don't need, if atypes2 exists from blocking step
        # print(atypes1, atypes2)
        atypes1['id'] = 'str_bt_1w_5w'
        # atypes1['name'] = 'str_bt_1w_5w'
        atypes1['pinyin'] = 'str_bt_1w_5w'
        match_c = em.get_attr_corres(w, s)
        feature_table = em.get_features(w, s, atypes1, atypes2, match_c, match_t, match_s)
        print(feature_table)
        # feature_table = em.get_features_for_matching(w, s, validate_inferred_attr_types=True)
        # return
        H = em.extract_feature_vecs(I,
                                    feature_table=feature_table,
                                    attrs_after='label',
                                    show_progress=False)

        L = em.extract_feature_vecs(J, feature_table=feature_table,
                                    attrs_after='label', show_progress=False)

        # magellan_choose_model(H)
        # magellan_model_ana(H, L)
        # return

        rf = em.RFMatcher(name='RF', random_state=0)
        rf.fit(table=H, exclude_attrs=['index', 'wifi', 'match', 'label'], target_attr='label')
        predictions = rf.predict(table=L, exclude_attrs=['index', 'wifi', 'match', 'label'],
                                 append=True, target_attr='predicted', inplace=False)
        eval_result = em.eval_matches(predictions, 'label', 'predicted')
        tp += eval_result['prec_numerator']
        fp += eval_result['false_pos_num']
        fn += eval_result['false_neg_num']
        # em.print_eval_summary(eval_result)
        # avg_pre += eval_result['precision']
        # avg_rec += eval_result['recall']
    # avg_pre /= ite_k
    # avg_rec /= ite_k
    # print('Avg pre =', avg_pre)
    # print('Avg rec =', avg_rec)
    # print('Avg F1 =', 2 * avg_pre * avg_rec / (avg_pre + avg_rec))
    tp /= ite_k
    fp /= ite_k
    fn /= ite_k
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)

    return


def deepmatch(pois, refolds=False, folds=10):
    if refolds:
        print('Make or remake data folds')
        dps.pro4deepmatcher_folds(pois, folds=folds)

    import deepmatcher as dm
    start_time = time.time()
    f1_list = list()

    for i in range(folds):
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
        # model = dm.MatchingModel(attr_summarizer=m)
        model = dm.MatchingModel(attr_summarizer=dm.attr_summarizers.Hybrid(word_contextualizer='lstm', word_aggregator='max-pool'), attr_comparator='concat')

        model.run_train(
            train,
            validation,
            epochs=10,
            batch_size=64,
            best_save_path='{}_model.pth'.format(m),
            pos_neg_ratio=3)

        f1_this = model.run_eval(test)
        f1_list.append(f1_this)
        print('{}: {}'.format(i, f1_this))

    end_time = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    print('time consume: {} mins'.format((end_time - start_time) / 60))
    print('average f1: {}'.format(sum(f1_list[:]) / folds))


def dm_test():
    import deepmatcher as dm
    train, validation, test = dm.data.process(
        path='{}/linking/matching/deepmatcher'.format(ex_path),
        train='dm_train_test.csv',
        validation='dm_val_test.csv',
        test='dm_test_test.csv',
        cache='m_dp_cache.pth',
        ignore_columns=['wifi', 'match', 'ltable_name', 'rtable_name'],
        left_prefix='ltable_',
        right_prefix='rtable_',
        id_attr='index',
        label_attr='label',
        embeddings='fasttext.zh.bin',pca=False)
    # use_magellan_convention=True)


if __name__ == '__main__':
    # magellan(['39.92451,116.51533'])
    # deepmatcher(['39.92451,116.51533', '39.93483,116.45241'], refolds=False, folds=10)
    # dm_test()

    pois = ['39.92451,116.51533', '39.93483,116.45241',  # 这两个是第一批
            '39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
            '39.96333,116.45187', '39.98850,116.41674', '40.00034,116.46960']

    magellan(pois)
    # deepmatch(pois, refolds=False, folds=10)

    # VSM(mod='gram', df=True, thres=0.9)
