import py_entitymatching as em
import readdata as rd
import api_amap_analyse as aaa
import api_dp_analyse as ada
import pro_func as pf
import pandas as pd
import time
import numpy as np
import os


def magellan_sample():
    A = em.read_csv_metadata('./src/experiment/amap_39.89136,116.46484.csv', key='id')
    D = em.read_csv_metadata('./src/experiment/dp_39.89136,116.46484.csv', key='id')

    # bb = em.BlackBoxBlocker()
    # bb.set_black_box_function(match_black_box)
    # C = bb.block_tables(A, D,
    #                     l_output_attrs=['id', 'Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
    #                     r_output_attrs=['id', 'Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
    #                     n_jobs=-2)
    #
    # print(len(C))
    # print(C.head())
    # em.to_csv_metadata(C, './src/experiment/blocked_39.89136,116.46484.csv')

    # C = em.read_csv_metadata('./src/experiment/blocked_39.89136,116.46484.csv', key='_id')

    # S = em.sample_table(C, 500)
    # G = em.label_table(S, 'label')
    # print(G.head())
    # em.to_csv_metadata(G, './src/experiment/500_labeled_39.89136,116.46484.csv')

    path_G = './src/experiment/500_labeled_39.89136,116.46484.csv'
    G = em.read_csv_metadata(path_G,
                             key='_id',
                             ltable=A, rtable=D,
                             fk_ltable='ltable_id', fk_rtable='rtable_id')

    rng = np.random.RandomState(int(time.time()))
    rng.rand(4)
    IJ = em.split_train_test(G, train_proportion=0.7, random_state=rng)
    I = IJ['train']
    J = IJ['test']


    feature_table = em.get_features_for_matching(A, D, validate_inferred_attr_types=False)
    # match_t = em.get_tokenizers_for_matching()
    # match_s = em.get_sim_funs_for_matching()
    # atypes1 = em.get_attr_types(A)  # don't need, if atypes1 exists from blocking step
    # atypes2 = em.get_attr_types(D)  # don't need, if atypes2 exists from blocking step
    # match_c = em.get_attr_corres(A,D)
    # match_f = em.get_features(A, D, atypes1, atypes2, match_c, match_t, match_s)


    H = em.extract_feature_vecs(I,
                                feature_table=feature_table,
                                attrs_after='label',
                                show_progress=False)
    # print(H.head())

    print('Choose the best Matcher')

    dt = em.DTMatcher(name='DecisionTree', random_state=0)
    svm = em.SVMMatcher(name='SVM', random_state=0)
    rf = em.RFMatcher(name='RF', random_state=0)
    lg = em.LogRegMatcher(name='LogReg', random_state=0)
    ln = em.LinRegMatcher(name='LinReg')
    nb = em.NBMatcher(name='NaiveBayes')

    # result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H,
    #                            exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #                            k=5,
    #                            target_attr='label', metric_to_select_matcher='precision', random_state=0)
    # print(result['cv_stats'])

    print('Label the test set')
    L = em.extract_feature_vecs(J, feature_table=feature_table,
                                attrs_after='label', show_progress=False)

    # print('Decision Tree')
    # dt.fit(table=H,
    #        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #        target_attr='label')
    # predictions = dt.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #                          append=True, target_attr='predicted', inplace=False)
    # eval_result = em.eval_matches(predictions, 'label', 'predicted')
    # em.print_eval_summary(eval_result)
    #
    # print('Random Forest')
    # rf.fit(table=H,
    #        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #        target_attr='label')
    # predictions = rf.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #                          append=True, target_attr='predicted', inplace=False)
    # eval_result = em.eval_matches(predictions, 'label', 'predicted')
    # em.print_eval_summary(eval_result)

    for matcher in [dt, rf, svm, ln, lg, nb]:
        print(matcher)
        matcher.fit(table=H,
                    exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                    target_attr='label')
        predictions = matcher.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                 append=True, target_attr='predicted', inplace=False)
        eval_result = em.eval_matches(predictions, 'label', 'predicted')
        em.print_eval_summary(eval_result)


def magellan_random_forest(path, ltable, rtable, train, t_ltable, t_rtable, to_pred):
    print('Random Forest Prediction:')
    em.set_key(train, '_id')
    em.set_ltable(train, ltable)
    em.set_rtable(train, rtable)
    em.set_fk_ltable(train, 'ltable_id')
    em.set_fk_rtable(train, 'rtable_id')
    em.set_key(to_pred, '_id')
    em.set_ltable(to_pred, t_ltable)
    em.set_rtable(to_pred, t_rtable)
    em.set_fk_ltable(to_pred, 'ltable_id')
    em.set_fk_rtable(to_pred, 'rtable_id')

    feature_table = em.get_features_for_matching(ltable, rtable, validate_inferred_attr_types=False)

    H = em.extract_feature_vecs(train,
                                feature_table=feature_table,
                                attrs_after='label',
                                show_progress=False)
    rf = em.RFMatcher(name='RF', random_state=0)
    rf.fit(table=H,
           exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
           target_attr='label')
    L = em.extract_feature_vecs(to_pred,
                                feature_table=feature_table,
                                # attrs_after='label',
                                show_progress=False)
    predictions = rf.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id'],
                             append=True, target_attr='predicted', inplace=False)
    # em.to_csv_metadata(predictions, './prediction.csv')

    pred_dict_amap = dict()
    for index, i in enumerate(predictions['predicted']):
        if i != 0:
            if pred_dict_amap.__contains__(predictions['ltable_id'][index]):
                pred_dict_amap[predictions['ltable_id'][index]].append(predictions['rtable_id'][index])
            else:
                pred_dict_amap[predictions['ltable_id'][index]] = [predictions['rtable_id'][index]]
    print('amap_key leng = {}'.format(len(pred_dict_amap)))
    with open('{}/target_prediction.txt'.format(path), 'w+') as f:
        for key in pred_dict_amap.keys():
            f.write('{}\t{}\n'.format(key, pred_dict_amap[key]))


def random_forest_train(ltable, rtable, train):
    print('Random Forest Train')
    em.set_key(train, '_id')
    em.set_ltable(train, ltable)
    em.set_rtable(train, rtable)
    em.set_fk_ltable(train, 'ltable_id')
    em.set_fk_rtable(train, 'rtable_id')

    feature_table = em.get_features_for_matching(ltable, rtable, validate_inferred_attr_types=False)

    H = em.extract_feature_vecs(train,
                                feature_table=feature_table,
                                attrs_after='label',
                                show_progress=False)
    rf = em.RFMatcher(name='RF', random_state=0)
    rf.fit(table=H,
           exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
           target_attr='label')
    print('Training over')
    return rf, feature_table


def magellan_predict_test(ltable, rtable, manual):
    # G = em.read_csv_metadata(path_G,
    #                          key='_id',
    #                          ltable=A, rtable=D,
    #                          fk_ltable='ltable_id', fk_rtable='rtable_id')
    em.set_key(manual, '_id')
    em.set_ltable(manual, ltable)
    em.set_rtable(manual, rtable)
    em.set_fk_ltable(manual, 'ltable_id')
    em.set_fk_rtable(manual, 'rtable_id')

    rng = np.random.RandomState(int(time.time()))
    rng.rand(4)
    IJ = em.split_train_test(manual, train_proportion=0.7, random_state=rng)
    I = IJ['train']
    J = IJ['test']

    feature_table = em.get_features_for_matching(ltable, rtable, validate_inferred_attr_types=False)

    H = em.extract_feature_vecs(I,
                                feature_table=feature_table,
                                attrs_after='label',
                                show_progress=False)
    # print(H.head())

    print('Choose the best Matcher')

    dt = em.DTMatcher(name='DecisionTree', random_state=0)
    svm = em.SVMMatcher(name='SVM', random_state=0)
    rf = em.RFMatcher(name='RF', random_state=0)
    lg = em.LogRegMatcher(name='LogReg', random_state=0)
    ln = em.LinRegMatcher(name='LinReg')
    nb = em.NBMatcher(name='NaiveBayes')

    result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H,
                               exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                               k=5,
                               target_attr='label', metric_to_select_matcher='precision', random_state=0)
    print(result['cv_stats'])

    print('Label the test set')
    L = em.extract_feature_vecs(J, feature_table=feature_table,
                                attrs_after='label', show_progress=False)

    # print('Decision Tree')
    # dt.fit(table=H,
    #        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #        target_attr='label')
    # predictions = dt.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #                          append=True, target_attr='predicted', inplace=False)
    # eval_result = em.eval_matches(predictions, 'label', 'predicted')
    # em.print_eval_summary(eval_result)
    #
    # print('Random Forest')
    # rf.fit(table=H,
    #        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #        target_attr='label')
    # predictions = rf.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
    #                          append=True, target_attr='predicted', inplace=False)
    # eval_result = em.eval_matches(predictions, 'label', 'predicted')
    # em.print_eval_summary(eval_result)

    for matcher in [dt, rf, svm, ln, lg, nb]:
        print(matcher)
        matcher.fit(table=H,
                    exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                    target_attr='label')
        predictions = matcher.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                      append=True, target_attr='predicted', inplace=False)
        eval_result = em.eval_matches(predictions, 'label', 'predicted')
        em.print_eval_summary(eval_result)


def magellan_learn_manual():
    amap_poi_shop = aaa.read_new_poi_shop(0)
    amap_shop_list = aaa.read_new_shop_list()
    dp_poi_shop = ada.read_poi_shop(0)
    dp_shop_list = ada.read_shop_list()
    dp_shop_addr = ada.read_poi_address(1)
    dp_shop_poi = ada.read_shop_poi(1)

    path = './src/experiment/manual'
    m_path = './src/experiment/merged'
    pois = ['39.89136,116.46484', '39.99636,116.47291', '39.95874,116.45154', '39.92028,116.48288',
            '39.97743,116.31017', '39.94000,116.40227']
    target_poi = ['39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
                  '39.97564,116.30627', '40.00034,116.46960']

    this_manual = read_test(path, pois)
    em.to_csv_metadata(this_manual, '{}/temp_manual.csv'.format(path))
    this_manual = em.read_csv_metadata('{}/temp_manual.csv'.format(path), key='_id')
    print('manual_list:{}'.format(len(this_manual)))

    a, d = get_a_d_table(pois,
                         amap_poi_shop, amap_shop_list, dp_poi_shop, dp_shop_list, dp_shop_addr, dp_shop_poi,
                         path, retain_temp=False)

    rf, feature_table = random_forest_train(a, d, this_manual)

    # magellan_predict_test(a, d, this_manual)
    # return

    for index, p in enumerate(target_poi):
        print('target {} : {}'.format(index, p))
        target_a, target_d, to_pred = get_target_poi_tables(p, amap_poi_shop, amap_shop_list, dp_poi_shop,
                                                            dp_shop_list, dp_shop_addr, dp_shop_poi, path)
        em.set_key(to_pred, '_id')
        em.set_ltable(to_pred, target_a)
        em.set_rtable(to_pred, target_d)
        em.set_fk_ltable(to_pred, 'ltable_id')
        em.set_fk_rtable(to_pred, 'rtable_id')

        L = em.extract_feature_vecs(to_pred,
                                    feature_table=feature_table,
                                    # attrs_after='label',
                                    show_progress=False)
        predictions = rf.predict(table=L, exclude_attrs=['_id', 'ltable_id', 'rtable_id'],
                                 append=True, target_attr='predicted', inplace=False)
        em.to_csv_metadata(predictions, '{}/prediction.csv'.format(path))
        # predictions = em.read_csv_metadata('{}/prediction.csv'.format(path), key='_id', ltable=target_a, rtable=target_d)

        shop_df = pred_to_final_shop_list(predictions, target_a, target_d)

        shop_df.to_csv('{}/merged_shop_list_{}.csv'.format(m_path, p), index=False)

        simple_shop_df = shop_df[['source', 'a_id', 'a_full_name', 'd_id', 'd_full_name']]
        simple_shop_df.to_csv('{}/simple/s_merged_shop_list_{}.csv'.format(m_path, p), index=False)
        del target_a, target_d, to_pred, shop_df
        # with open('{}/target_prediction.txt'.format(path), 'w+') as f:
        #     for key in pred_dict_amap.keys():
        #         f.write('{}\t{}\n'.format(key, pred_dict_amap[key]))


def pred_to_final_shop_list(predictions, target_a, target_d):
    pred_dict_amap = dict()
    for index, i in enumerate(predictions['predicted']):
        if i != 0:
            if pred_dict_amap.__contains__(predictions['ltable_id'][index]):
                pred_dict_amap[predictions['ltable_id'][index]].append(predictions['rtable_id'][index])
            else:
                pred_dict_amap[predictions['ltable_id'][index]] = [predictions['rtable_id'][index]]
    print('amap_key leng = {}'.format(len(pred_dict_amap)))
    pred_dict_dp = dict()
    for key in pred_dict_amap.keys():
        for i in pred_dict_amap[key]:
            if pred_dict_dp.__contains__(i):
                pred_dict_dp[i].append(key)
            else:
                pred_dict_dp[i] = [key]
    print('dp_key leng = {}'.format(len(pred_dict_dp)))

    shop_df = pd.DataFrame(columns=['source', 'a_id', 'a_shop_name', 'a_full_name', 'a_addr', 'a_lat', 'a_lng',
                                    'd_id', 'd_shop_name', 'd_full_name', 'd_addr', 'd_lat', 'd_lng'])
    count = 0
    done_dp = dict()

    for i in range(len(target_a)):
        a_shop_id = target_a.ix[i]['id']
        a_shop = target_a.ix[i]
        # print(i, a_shop_id)
        if pred_dict_amap.__contains__(a_shop_id):
            if len(pred_dict_amap[a_shop_id]) > 1:
                flag = 0
                for j in pred_dict_amap[a_shop_id]:
                    this_dp_shop = target_d.ix[target_d['id'] == j].iloc[0]
                    if a_shop['Name_Full'] == this_dp_shop['Name_Full'] and a_shop['Addr'] == this_dp_shop['Addr']:
                        shop_df.loc[count] = [0, a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                             a_shop['Addr'], a_shop['Lat'], a_shop['Lng'],
                                             this_dp_shop['id'], this_dp_shop['Name'], this_dp_shop['Name_Full'],
                                             this_dp_shop['Addr'], this_dp_shop['Lat'], this_dp_shop['Lng']]
                        count += 1
                        done_dp[j] = 1
                        flag = 1
                        break
                if not flag:
                    shop_df.loc[count] = [1, a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                         a_shop['Addr'], a_shop['Lat'], a_shop['Lng'],
                                         a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                         a_shop['Addr'], a_shop['Lat'], a_shop['Lng']]
                    count += 1
            else:
                this_dp_shop_id = pred_dict_amap[a_shop_id][0]
                this_dp_shop = target_d.ix[target_d['id'] == this_dp_shop_id].iloc[0]
                if len(pred_dict_dp[this_dp_shop_id]) == 1:
                    shop_df.loc[count] = [0, a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                          a_shop['Addr'], a_shop['Lat'], a_shop['Lng'],
                                          this_dp_shop['id'], this_dp_shop['Name'], this_dp_shop['Name_Full'],
                                          this_dp_shop['Addr'], this_dp_shop['Lat'], this_dp_shop['Lng']]
                    count += 1
                    done_dp[this_dp_shop_id] = 1
                else:
                    if a_shop['Name_Full'] == this_dp_shop['Name_Full'] and a_shop['Addr'] == this_dp_shop['Addr']:
                        shop_df.loc[count] = [0, a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                             a_shop['Addr'], a_shop['Lat'], a_shop['Lng'],
                                             this_dp_shop['id'], this_dp_shop['Name'], this_dp_shop['Name_Full'],
                                             this_dp_shop['Addr'], this_dp_shop['Lat'], this_dp_shop['Lng']]
                        count += 1
                        done_dp[this_dp_shop_id] = 1
                    else:
                        shop_df.loc[count] = [1, a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                             a_shop['Addr'], a_shop['Lat'], a_shop['Lng'],
                                             a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                             a_shop['Addr'], a_shop['Lat'], a_shop['Lng']]
                        count += 1
        else:
            shop_df.loc[count] = [1, a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                 a_shop['Addr'], a_shop['Lat'], a_shop['Lng'],
                                 a_shop_id, a_shop['Name'], a_shop['Name_Full'],
                                 a_shop['Addr'], a_shop['Lat'], a_shop['Lng']]
            count += 1
    print('after key=amap, got {} shops'.format(len(shop_df)))
    for i in range(len(target_d)):
        d_shop_id = target_d.ix[i]['id']
        if not done_dp.__contains__(d_shop_id):
            this_dp_shop = target_d.ix[i]
            shop_df.ix[count] = [2, this_dp_shop['id'], this_dp_shop['Name'], this_dp_shop['Name_Full'],
                                 this_dp_shop['Addr'], this_dp_shop['Lat'], this_dp_shop['Lng'],
                                 this_dp_shop['id'], this_dp_shop['Name'], this_dp_shop['Name_Full'],
                                 this_dp_shop['Addr'], this_dp_shop['Lat'], this_dp_shop['Lng']]
            count += 1
            done_dp[d_shop_id] = 1
    print('after key=dp, got {} shops'.format(len(shop_df)))
    print('({} + {}) - {} = {}'.format(len(target_a), len(target_d), len(shop_df), len(target_a) + len(target_d) - len(shop_df)))

    return shop_df


def get_target_poi_tables(poi, amap_poi_shop, amap_shop_list, dp_poi_shop, dp_shop_list, dp_shop_addr, dp_shop_poi,
                          path):
    print('get target poi tables')
    a, d = get_a_d_table([poi], amap_poi_shop, amap_shop_list, dp_poi_shop, dp_shop_list, dp_shop_addr, dp_shop_poi,
                         path, retain_temp=False)

    if os.path.exists('{}/blocked_{}.csv'.format(path, poi)):
        C = em.read_csv_metadata('{}/blocked_{}.csv'.format(path, poi), key='_id', ltable=a, rtable=d)
    else:
        bb = em.BlackBoxBlocker()
        bb.set_black_box_function(match_black_box)
        C = bb.block_tables(a, d,
                            l_output_attrs=['id', 'Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
                            r_output_attrs=['id', 'Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
                            n_jobs=-2)

        print('target poi {} blocked leng = {}'.format(poi, len(C)))
        # C['label'] = 0
        # C['label'] = C['label'].astype(int)
        em.to_csv_metadata(C, '{}/blocked_{}.csv'.format(path, poi))

    return a, d, C


def get_a_d_table(pois, amap_poi_shop, amap_shop_list, dp_poi_shop, dp_shop_list, dp_shop_addr, dp_shop_poi,
                  path, retain_temp=False):
    this_amap, this_dp = {'id': [], 'Name': [], 'Name_Full': [], 'Addr': [], 'Lat': [], 'Lng': []}, \
                         {'id': [], 'Name': [], 'Name_Full': [], 'Addr': [], 'Lat': [], 'Lng': []}
    for poi in pois:
        print(poi)
        dp_shops = dp_poi_shop[poi]
        amap_shops = amap_poi_shop[poi]
        for i in amap_shops.keys():
            if amap_shop_list.__contains__(i):
                amap_shop = amap_shop_list[i]
                name = amap_shop['name'].replace('(', '').replace(')', '')
                if amap_shop['location']:
                    lat, lng = pf.exchange_lag_lng(amap_shop['location']).split(',')
                else:
                    lat, lng = 0, 0
                if amap_shop['address']:
                    addr = amap_shop['address']
                else:
                    addr = 'NO'
                if not this_amap['id'].__contains__(i):
                    this_amap['id'].append(i)
                    this_amap['Name'].append(name)
                    this_amap['Name_Full'].append(name)
                    this_amap['Addr'].append(addr)
                    this_amap['Lat'].append(lat)
                    this_amap['Lng'].append(lng)
        for j in dp_shops.keys():
            if dp_shop_list.__contains__(j):
                dp_shop = dp_shop_list[j]
                if dp_shop_addr.__contains__(j):
                    dp_shop_a = dp_shop_addr[j]['address']
                else:
                    dp_shop_a = 'NO'
                if dp_shop_poi.__contains__(j):
                    lat, lng = dp_shop_poi[j][1].split(',')
                else:
                    lat, lng = 0, 0
                if not this_dp['id'].__contains__(j):
                    this_dp['id'].append(j)
                    this_dp['Name'].append(dp_shop['name'])
                    this_dp['Name_Full'].append('{}{}'.format(dp_shop['name'], dp_shop['branch']))
                    this_dp['Addr'].append(dp_shop_a)
                    this_dp['Lat'].append(lat)
                    this_dp['Lng'].append(lng)

    a, d = pd.DataFrame(this_amap), pd.DataFrame(this_dp)
    # em.set_key(a, 'id')
    # em.set_key(d, 'id')
    print('amap_list:{}\tdp_list:{}'.format(len(a), len(d)))
    if retain_temp:
        prefix = 'retain'
    else:
        prefix = 'temp'
    em.to_csv_metadata(a, '{}/{}_a.csv'.format(path, prefix))
    em.to_csv_metadata(d, '{}/{}_d.csv'.format(path, prefix))
    try:
        os.remove('{}/{}_a.metadata'.format(path, prefix))
        os.remove('{}/{}_d.metadata'.format(path, prefix))
    except Exception:
        pass
    a = em.read_csv_metadata('{}/{}_a.csv'.format(path, prefix), key='id')
    d = em.read_csv_metadata('{}/{}_d.csv'.format(path, prefix), key='id')
    if not retain_temp:
        for c in ['a', 'd']:
            try:
                os.remove('{}/temp_{}.csv'.format(path, c))
                os.remove('{}/temp_{}.metadata'.format(path, c))
            except Exception:
                pass

    return a, d


def read_test(path, pois: list):
    for index, poi in enumerate(pois):
        if index == 0:
            l = pd.read_csv('{}/500_labeled_{}.csv'.format(path, poi))
        else:
            l = l.append(pd.read_csv('{}/500_labeled_{}.csv'.format(path, poi)), ignore_index=True)
    l = pd.DataFrame(l)
    l['_id'] = [i for i in range(len(l['_id']))]
    l['rtable_id'] = l['rtable_id'].astype(str)
    em.set_key(l, '_id')

    return l


def test():
    amap_poi_shop = aaa.read_new_poi_shop(0)
    amap_shop_list = aaa.read_new_shop_list()
    dp_poi_shop = ada.read_poi_shop(0)
    dp_shop_list = ada.read_shop_list()
    dp_shop_addr = ada.read_poi_address(1)
    dp_shop_poi = ada.read_shop_poi(1)

    path = './src/experiment/manual'
    # 39.99636,116.47291 39.97743,116.31017 39.95874,116.45154 39.94000,116.40227 39.92028,116.48288
    # done '39.89136,116.46484', '39.99636,116.47291', '39.95874,116.45154', '39.92028,116.48288', '39.97743,116.31017', '39.94000,116.40227'
    pois = ['39.94000,116.40227']

    for poi in pois:
        print(poi)
        dp_shops = dp_poi_shop[poi]
        amap_shops = amap_poi_shop[poi]
        this_amap, this_dp = {'id': [], 'Name': [], 'Name_Full': [], 'Addr': [], 'Lat': [], 'Lng': []}, \
                             {'id': [], 'Name': [], 'Name_Full': [], 'Addr': [], 'Lat': [], 'Lng': []}
        for i in amap_shops.keys():
            if amap_shop_list.__contains__(i):
                amap_shop = amap_shop_list[i]
                name = amap_shop['name'].replace('(', '').replace(')', '')
                if amap_shop['location']:
                    lat, lng = pf.exchange_lag_lng(amap_shop['location']).split(',')
                else:
                    lat, lng = 0, 0
                if amap_shop['address']:
                    addr = amap_shop['address']
                else:
                    addr = 'NO'
                this_amap['id'].append(i)
                this_amap['Name'].append(name)
                this_amap['Name_Full'].append(name)
                this_amap['Addr'].append(addr)
                this_amap['Lat'].append(lat)
                this_amap['Lng'].append(lng)
        for j in dp_shops.keys():
            if dp_shop_list.__contains__(j):
                dp_shop = dp_shop_list[j]
                if dp_shop_addr.__contains__(j):
                    dp_shop_a = dp_shop_addr[j]['address']
                else:
                    dp_shop_a = 'NO'
                if dp_shop_poi.__contains__(j):
                    lat, lng = dp_shop_poi[j][1].split(',')
                else:
                    lat, lng = 0, 0
                this_dp['id'].append(j)
                this_dp['Name'].append(dp_shop['name'])
                this_dp['Name_Full'].append('{}{}'.format(dp_shop['name'], dp_shop['branch']))
                this_dp['Addr'].append(dp_shop_a)
                this_dp['Lat'].append(lat)
                this_dp['Lng'].append(lng)

        a, d = pd.DataFrame(this_amap), pd.DataFrame(this_dp)
        em.set_key(a, 'id')
        em.set_key(d, 'id')
        print(len(a), len(d))
        # a.to_csv('./src/experiment/amap_39.89136,116.46484.csv')
        # d.to_csv('./src/experiment/dp_39.89136,116.46484.csv')
        em.to_csv_metadata(a, '{}/amap_{}.csv'.format(path, poi))
        em.to_csv_metadata(d, '{}/dp_{}.csv'.format(path, poi))


        # ob = em.OverlapBlocker()
        # C = ob.block_tables(a, d, 'name', 'name',
        #                     l_output_attrs=['Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
        #                     r_output_attrs=['Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
        #                     l_output_prefix='left_',
        #                     r_output_prefix='right_',
        #                     overlap_size=8, show_progress=False, word_level=False, q_val=2
        #                     )

        bb = em.BlackBoxBlocker()
        bb.set_black_box_function(match_black_box)
        C = bb.block_tables(a, d,
                            l_output_attrs=['id', 'Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
                            r_output_attrs=['id', 'Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
                            n_jobs=-2)

        print(len(C))
        # print(C.head())
        em.to_csv_metadata(C, '{}/blocked_{}.csv'.format(path, poi))

        S = em.sample_table(C, 500)
        G = em.label_table(S, 'label')
        # print(G.head())
        em.to_csv_metadata(G, '{}/500_labeled_{}.csv'.format(path, poi))

        # T = em.read_csv_metadata('./src/experiment/m_train.csv', key='id', ltable=a, rtable=d, fk_ltable='left_Id',
        #                          fk_rtable='right_Id')
        # IJ = em.split_train_test(T, train_proportion=0.7, random_state=0)
        # I = IJ['train']
        # J = IJ['test']
        # feature_table = em.get_features_for_matching(a, d, validate_inferred_attr_types=False)
        # H = em.extract_feature_vecs(I,
        #                             feature_table=feature_table,
        #                             attrs_after='label',
        #                             show_progress=False)
        # print(H.head())


def match_black_box(ltuple, rtuple):
    if pf.jaccard(ltuple['Name'], rtuple['Name']) >= 0.4:
        return False
    else:
        return True


def crossvalid(path, fold=10):
    all_labeled = em.read_csv_metadata('{}/temp_manual.csv'.format(path), key='_id')
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=fold)
    index = 0
    for train_index, test_index in skf.split(all_labeled, all_labeled['label']):
        print(index)
        print("TRAIN:", train_index, "TEST:", test_index)
        train_labeled = all_labeled.iloc[train_index]
        test_labeled = all_labeled.iloc[test_index]
        skff = StratifiedKFold(n_splits=fold-1)
        for train_index_indeed, valid_index in skff.split(train_labeled, train_labeled['label']):
            # print("TRAIN:", train_index_indeed, "TEST:", valid_index)
            train_labeled_indeed = train_labeled.iloc[train_index_indeed]
            valid_labeled = train_labeled.iloc[valid_index]
            break
        em.to_csv_metadata(train_labeled_indeed, '{}/dp/m_train_{}.csv'.format(path, index))
        em.to_csv_metadata(valid_labeled, '{}/dp/m_valid_{}.csv'.format(path, index))
        em.to_csv_metadata(test_labeled, '{}/dp/m_test_{}.csv'.format(path, index))
        index += 1


if __name__ == '__main__':
    # test()
    # magellan_sample()
    magellan_learn_manual()

    # crossvalid('./src/experiment/manual')

    # find_error()
    # little_work()