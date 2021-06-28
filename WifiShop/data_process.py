import sys
import os
sys.path.append(os.path.abspath('..'))
import pro_func as pf
import pandas as pd
import time
import random
import warnings
# import emoji
import math

ex_path = '../src/experiment'
ngram = 3
pois_global = ['39.92451,116.51533', '39.93483,116.45241', '39.86184,116.42517', '39.88892,116.32670',
               '39.90184,116.41196', '39.94735,116.35581', '39.96333,116.45187', '39.98850,116.41674',
               '40.00034,116.46960']


def pro_shop_list(shops):
    shops = shops[['source', 'a_id', 'a_full_name', 'd_id', 'd_full_name']]
    id_list, name_list,  pinyin_list, shengmu_list = list(), list(), list(), list()

    for index, row in shops.iterrows():
        if row['source'] == 0:
            sid = row['a_id'] + '|' + row['d_id']
            # sname = row['a_full_name'] + ' ' + row['d_full_name']
            sname = row['d_full_name']  # 新版更改 合并的店铺只保留点评的名字 ID不变
        else:
            sid = row['a_id']
            sname = row['a_full_name']
        # spinyin = pf.chinese2pinyin(sname)
        # sshengmu = pf.chinese2shengmu(sname, short=False)
        spinyin, sshengmu = pf.chinese2pyandsm(sname, short=False)
        id_list.append(sid)
        name_list.append(sname)
        pinyin_list.append(spinyin)
        shengmu_list.append(sshengmu)

    d = {'id': id_list, 'name': name_list, 'pinyin': pinyin_list, 'shengmu': shengmu_list}
    df = pd.DataFrame(d)
    return df


def pro_wifi_list(match):
    wifi_list, wpinyin_list, wshengmu_list = list(), list(), list()
    m_wifi_list, m_shop_list, label = list(), list(), list()
    for index, row in match.iterrows():
        py, sm = pf.chinese2pyandsm(row['wifi'], short=False)
        wifi_list.append(row['wifi'])
        wpinyin_list.append(py)
        wshengmu_list.append(sm)
        try:
            matches = row['match'].split(';')
        except Exception as e:
            print(row)
        for i in matches:
            m_wifi_list.append(row['wifi'])
            m_shop_list.append(i)
            label.append(1)
    d_wifi = {'name': wifi_list, 'pinyin': wpinyin_list, 'shengmu': wshengmu_list}
    df_wifi = pd.DataFrame(d_wifi)
    d_match = {'wifi': m_wifi_list, 'match': m_shop_list, 'label': label}
    df_match = pd.DataFrame(d_match)
    return df_wifi, df_match


def gene_neg_matches(pos_matches, shops, n_p_ratio=1):
    p_dic = dict()
    for index, row in pos_matches.iterrows():
        if p_dic.__contains__(row['wifi']):
            p_dic[row['wifi']].append(row['match'])
        else:
            p_dic[row['wifi']] = [row['match']]
    m_wifi_list, m_shop_list, label, = list(), list(), list()
    rand_max = len(shops)
    for index, row in pos_matches.iterrows():
        for i in range(n_p_ratio):
            r = random.randint(0, rand_max - 1)
            while shops.iloc[r]['id'] in p_dic[row['wifi']]:
                r = random.randint(0, rand_max - 1)
            m_wifi_list.append(row['wifi'])
            m_shop_list.append(shops.iloc[r]['id'])
            label.append(0)
    d = {'wifi': m_wifi_list, 'match': m_shop_list, 'label': label}
    df = pd.DataFrame(d)
    return df


def gene_neg_matches_www(pos_matches, shops, n_p_ratio=1):
    import Levenshtein as lv
    p_dic = dict()
    for index, row in pos_matches.iterrows():
        if p_dic.__contains__(row['wifi']):
            p_dic[row['wifi']].append(row['match'])
        else:
            p_dic[row['wifi']] = [row['match']]
    m_wifi_list, m_shop_list, label, = list(), list(), list()
    for key in p_dic.keys():
        scores = list()
        for index, row in shops.iterrows():
            pinyin_key = pf.chinese2pinyin(key)
            pinyin_key_gram = [pinyin_key[i:i + ngram] for i in range(len(pinyin_key) - ngram + 1)]
            row_gram = [row['pinyin'][i:i + ngram] for i in range(len(row['pinyin']) - ngram + 1)]
            if not pinyin_key_gram and not row_gram:
                scores.append(0)
            else:
                # if '|' in row['id']:
                #     # p1, p2 = row['pinyin'].split(' ')  # 有的shop自带空格 比较难办
                #     # overlap1 = len(set(pinyin_key) & set(p1))
                #     # overlap2 = len(set(pinyin_key) & set(p2))
                #     # score = (overlap1 + overlap2) * 5 + (lv.distance(pinyin_key, p1) + lv.distance(pinyin_key, p2)) * 0.5
                #     overlap = pf.jaccard(pinyin_key_gram, row_gram)  # / math.log(len(row['pinyin']) / 2)
                #     overlap2 = pf.jaccard(set(key), set(row['name']))
                #     score = (overlap + overlap2) * 100.0 + lv.distance(pinyin_key, row['pinyin']) * 0.5
                # else:
                #     overlap = pf.jaccard(pinyin_key_gram, row_gram)  # / math.log(len(row['pinyin']))
                #     overlap2 = pf.jaccard(set(key), set(row['name']))
                #     score = (overlap + overlap2) * 200.0 + lv.distance(pinyin_key, row['pinyin'])

                # 新版
                overlap = pf.jaccard(pinyin_key_gram, row_gram)  # / math.log(len(row['pinyin']))
                overlap2 = pf.jaccard(set(key), set(row['name']))
                score = (overlap + overlap2) * 100.0 + lv.distance(pinyin_key, row['pinyin'])
                scores.append(score)
        sort_scores = sorted(scores, reverse=True)
        print(sort_scores)
        neg_num, neg_index = 0, 0
        while neg_num < n_p_ratio * len(p_dic[key]):
            i = scores.index(sort_scores[neg_index])
            if shops.iloc[i]['id'] not in p_dic[key]:
                m_wifi_list.append(key)
                m_shop_list.append(shops.iloc[i]['id'])
                label.append(0)
                neg_num += 1
            scores[i] = -1
            neg_index += 1

    d = {'wifi': m_wifi_list, 'match': m_shop_list, 'label': label}
    df = pd.DataFrame(d)
    return df


def pro_data(pois, n_p_ratio=5, remake=False):
    time_start = time.time()
    total_wifi, total_pos = 0, 0

    # linking_path = '../src/experiment/linking'
    for poi in pois:
        print(poi)
        if not remake and os.path.exists('{}/linking/prepro/tables/shops_{}.csv'.format(ex_path, poi)):
            print('{} shop file exists. Read it'.format(poi))
            shops = pd.read_csv('{}/linking/prepro/tables/shops_{}.csv'.format(ex_path, poi))
        else:
            shops = pd.read_csv('{}/merged/merged_shop_list_{}.csv'.format(ex_path, poi))
            shops = pro_shop_list(shops)
            print(len(shops))
            shops.to_csv('{}/linking/prepro/tables/shops_{}.csv'.format(ex_path, poi), index=False)

        if not remake and os.path.exists('{}/linking/prepro/tables/wifis_{}.csv'.format(ex_path, poi)) and \
                os.path.exists('{}/linking/prepro/pos_{}.csv'.format(ex_path, poi)):
            print('{} wifi&match file exists. Read it'.format(poi))
            wifis = pd.read_csv('{}/linking/prepro/tables/wifis_{}.csv'.format(ex_path, poi))
            pos_matches = pd.read_csv('{}/linking/prepro/pos_{}.csv'.format(ex_path, poi))
        else:
            instance = pd.read_csv('{}/linking/instance/{}.csv'.format(ex_path, poi))
            pos_instance = instance[instance['match'] != '0']
            print(len(pos_instance))
            wifis, pos_matches = pro_wifi_list(pos_instance)
            wifis.to_csv('{}/linking/prepro/tables/wifis_{}.csv'.format(ex_path, poi))
            pos_matches.to_csv('{}/linking/prepro/pos_{}.csv'.format(ex_path, poi))
        print(len(wifis), len(pos_matches))
        total_wifi += len(wifis)
        total_pos += len(pos_matches)

        if remake or not os.path.exists('{}/linking/prepro/neg_{}.csv'.format(ex_path, poi)):
            print('Make or Remake neg_matches')
            # neg_matches = gene_neg_matches(pos_matches, shops, n_p_ratio=n_p_ratio)  # 随机生成反例
            neg_matches = gene_neg_matches_www(pos_matches, shops, n_p_ratio=n_p_ratio)  # 类www19生成相似的反例
            neg_matches.to_csv('{}/linking/prepro/neg_{}.csv'.format(ex_path, poi))
        else:
            print('{} neg file exists. Do nothing'.format(poi))
        # print(shops.iloc[2]['id'])

    time_end = time.time()
    print('process data cost:', time_end - time_start)
    print('{} wifis; {} pos_match'.format(total_wifi, total_pos))


def get_pos_neg_matches(pois):
    t_pos = pd.DataFrame(columns=('wifi', 'match', 'label'))
    t_neg = pd.DataFrame(columns=('wifi', 'match', 'label'))
    for index, poi in enumerate(pois):
        try:
            pos = pd.read_csv('{}/linking/prepro/pos_{}.csv'.format(ex_path, poi))
            neg = pd.read_csv('{}/linking/prepro/neg_{}.csv'.format(ex_path, poi))
        except Exception as e:
            print(e)
        if index:
            t_pos = pd.concat([t_pos, pos])
            t_neg = pd.concat([t_neg, neg])
        else:
            t_pos = pos
            t_neg = neg
    return t_pos, t_neg


def get_wifi_table(pois):
    t_wifi = pd.DataFrame(columns=('name', 'pinyin', 'shengmu'))
    for poi in pois:
        try:
            w = pd.read_csv('{}/linking/prepro/tables/wifis_{}.csv'.format(ex_path, poi))
        except Exception as e:
            print(e)
        t_wifi = pd.concat([t_wifi, w])
    return t_wifi


def get_shop_table(pois):
    t_shop = pd.DataFrame(columns=('id', 'name', 'pinyin', 'shengmu'))
    for poi in pois:
        try:
            s = pd.read_csv('{}/linking/prepro/tables/shops_{}.csv'.format(ex_path, poi))
        except Exception as e:
            print(e)
        t_shop = pd.concat([t_shop, s])
    return t_shop


def get_search_recommendation():
    result = dict()
    path = '../src/search recommendation'
    with open('{}/recom.txt'.format(path), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            wifi, recs = line.strip().split('\t')
            recs = eval(recs)
            if result.__contains__(wifi):
                result[wifi] = list(set(result[wifi]) | set(recs))
            else:
                result[wifi] = recs
    return result


def get_search_rec_fuzzy(f_list, ps):
    path = '../src/search recommendation'
    result_dict = dict()
    for fuzzy_name in f_list:
        temp_path = path + '/fuzzy_' + fuzzy_name
        if not os.path.exists('{}/recom.txt'.format(temp_path)):
            print('{}/recom.txt not exists. Work for it'.format(temp_path))
            pro4search_recom_fuzzy(ps, [fuzzy_name], save=True)
        result = dict()
        with open('{}/recom.txt'.format(temp_path), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wifi, recs = line.strip().split('\t')
                recs = eval(recs)
                if result.__contains__(wifi):
                    result[wifi] = list(set(result[wifi]) | set(recs))
                else:
                    result[wifi] = recs
        result_dict[fuzzy_name] = result
    return result_dict


def get_search_rec_all_with_fuzzy(pois, f_list, statistic=False):
    base_rec = get_search_recommendation()
    fuzzy_rec = get_search_rec_fuzzy(f_list, pois)
    result = base_rec.copy()
    for key in fuzzy_rec.keys():
        print('Combine fuzzy :', key)
        f_rec_leng = 0
        for w in fuzzy_rec[key].keys():
            f_rec_leng += len(fuzzy_rec[key][w])
            if result.__contains__(w):
                result[w] = list(set(result[w]) | set(fuzzy_rec[key][w]))
            else:
                result[w] = fuzzy_rec[key][w]
            if not base_rec.__contains__(w) and statistic:
                print(w, fuzzy_rec[key][w])
        if statistic:
            print('wifi:{}\trecs/wifis={}'.format(len(fuzzy_rec[key].keys()), f_rec_leng / len(fuzzy_rec[key].keys())))
    if statistic:
        temp_rec_leng = sum(len(result[key]) for key in result.keys())
        print('Final result: {}/{} = {}'.format(temp_rec_leng, len(result), temp_rec_leng / len(result)))
        print('total avg: {}/{} = {}'.format(temp_rec_leng, 1021, temp_rec_leng / 1021))
    return result


def gene_train_test_val(pois, folds=5):
    t_pos, t_neg = get_pos_neg_matches(pois)
    m_train, m_test = list(), list()
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=folds, shuffle=True)
    for train_index, test_index in k_fold.split(t_pos):
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        tra = t_pos.iloc[train_index]
        tes = t_pos.iloc[test_index]
        m_train.append(tra)
        m_test.append(tes)
    f = 0
    for train_index, test_index in k_fold.split(t_neg):
        tra = t_pos.iloc[train_index]
        tes = t_pos.iloc[test_index]
        m_train[f] = pd.concat([m_train[f], tra])
        m_test[f] = pd.concat([m_test[f], tes])
        f += 1
    return m_train, m_test


def pro4magellan_folds(pois):
    pro_data(pois, remake=False)
    m_train, m_test = gene_train_test_val(pois, folds=5)
    t_wifi = get_wifi_table(pois)
    t_shop = get_shop_table(pois)
    for index, train in enumerate(m_train):
        test = m_test[index]
        train = pd.concat([train, pd.DataFrame(columns=('ltable_name', 'rtable_name', 'ltable_pinyin', 'rtable_pinyin'))])
        test = pd.concat([test, pd.DataFrame(columns=('ltable_name', 'rtable_name', 'ltable_pinyin', 'rtable_pinyin'))])
        train['ltable_name'] = train['ltable_name'].fillna('')
        train['rtable_name'] = train['rtable_name'].fillna('')
        train['ltable_pinyin'] = train['ltable_pinyin'].fillna('')
        train['rtable_pinyin'] = train['rtable_pinyin'].fillna('')
        test['ltable_name'] = test['ltable_name'].fillna('')
        test['rtable_name'] = test['rtable_name'].fillna('')
        test['ltable_pinyin'] = test['ltable_pinyin'].fillna('')
        test['rtable_pinyin'] = test['rtable_pinyin'].fillna('')

        for i in range(len(train)):
            temp_train = train.iloc[i].copy()
            wifi = t_wifi[t_wifi['name'] == train.iloc[i]['wifi']]
            if wifi.empty:
                warnings.warn('wifi:{} not found!'.format(train.iloc[i]['wifi']))
            else:
                temp_train['ltable_name'], temp_train['ltable_pinyin'] = wifi.iloc[0]['name'], wifi.iloc[0]['pinyin']
            shop = t_shop[t_shop['id'] == train.iloc[i]['match']]
            if shop.empty:
                warnings.warn('id: {} not found!'.format(train.iloc[i]['match']))
            else:
                temp_train['rtable_name'], temp_train['rtable_pinyin'] = shop.iloc[0]['name'], shop.iloc[0]['pinyin']
            train.iloc[i] = temp_train
        for i in range(len(test)):
            temp_test = test.iloc[i].copy()
            wifi = t_wifi[t_wifi['name'] == test.iloc[i]['wifi']]
            if wifi.empty:
                warnings.warn('wifi:{} not found!'.format(test.iloc[i]['wifi']))
            else:
                temp_test['ltable_name'], temp_test['ltable_pinyin'] = wifi.iloc[0]['name'], wifi.iloc[0]['pinyin']
            shop = t_shop[t_shop['id'] == test.iloc[i]['match']]
            if shop.empty:
                warnings.warn('id: {} not found!'.format(test.iloc[i]['match']))
            else:
                temp_test['rtable_name'], temp_test['rtable_pinyin'] = shop.iloc[0]['name'], shop.iloc[0]['pinyin']
            test.iloc[i] = temp_test
    return t_wifi, t_shop, m_train, m_test


def pro4magellan_without_folds(pois, n_p_ratio=5, redata=False):
    pro_data(pois, n_p_ratio=n_p_ratio, remake=redata)
    t_pos, t_neg = get_pos_neg_matches(pois)
    t_wifi = get_wifi_table(pois)
    t_wifi.reset_index()
    t_wifi.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    for i in range(len(t_wifi)):
        temp = t_wifi.iloc[i].copy()
        temp['id'] = 'w' + str(i)
        t_wifi.iloc[i] = temp
    t_wifi['id'] = t_wifi['id'].astype('str')
    t_shop = get_shop_table(pois)
    t_match = pd.concat([t_pos, t_neg])
    t_match = pd.concat([t_match, pd.DataFrame(columns=('ltable_name', 'rtable_name', 'ltable_pinyin', 'rtable_pinyin'))])
    t_match['ltable_name'] = t_match['ltable_name'].fillna('')
    t_match['rtable_name'] = t_match['rtable_name'].fillna('')
    t_match['ltable_pinyin'] = t_match['ltable_pinyin'].fillna('')
    t_match['rtable_pinyin'] = t_match['rtable_pinyin'].fillna('')

    for i in range(len(t_match)):
        temp = t_match.iloc[i].copy()
        wifi = t_wifi[t_wifi['name'] == t_match.iloc[i]['wifi']]
        if wifi.empty:
            warnings.warn('wifi:{} not found!'.format(t_match.iloc[i]['wifi']))
        else:
            temp['ltable_name'], temp['ltable_pinyin'], temp['wifi'] = wifi.iloc[0]['name'], wifi.iloc[0]['pinyin'], wifi.iloc[0]['id']
        shop = t_shop[t_shop['id'] == t_match.iloc[i]['match']]
        if shop.empty:
            warnings.warn('id: {} not found!'.format(t_match.iloc[i]['match']))
        else:
            temp['rtable_name'], temp['rtable_pinyin'] = shop.iloc[0]['name'], shop.iloc[0]['pinyin']
        t_match.iloc[i] = temp
    t_match.reset_index()
    t_match.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    t_match['label'] = t_match['label'].astype('int32')
    for i in range(len(t_match)):
        temp = t_match.iloc[i].copy()
        temp['index'] = 'ws' + str(i)
        t_match.iloc[i] = temp
    t_match['index'] = t_match['index'].astype('str')
    # print(t_match)
    return t_wifi, t_shop, t_match


def pro4our_pairwise(pois, n_p_ratio=5, redata=False):
    pro_data(pois, n_p_ratio=n_p_ratio, remake=redata)
    t_pos, t_neg = get_pos_neg_matches(pois)
    t_wifi = get_wifi_table(pois)
    t_wifi.reset_index()
    t_wifi.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    for i in range(len(t_wifi)):
        temp = t_wifi.iloc[i].copy()
        temp['id'] = 'w' + str(i)
        t_wifi.iloc[i] = temp
    t_wifi['id'] = t_wifi['id'].astype('str')
    t_shop = get_shop_table(pois)
    t_match = t_neg.copy()
    t_match = pd.concat([t_match, pd.DataFrame(columns=('wifi_name', 'false_name', 'wifi_pinyin', 'false_pinyin', 'true_name', 'true_pinyin'))])
    t_match['wifi_name'] = t_match['wifi_name'].fillna('')
    t_match['false_name'] = t_match['false_name'].fillna('')
    t_match['wifi_pinyin'] = t_match['wifi_pinyin'].fillna('')
    t_match['false_pinyin'] = t_match['false_pinyin'].fillna('')
    t_match['true_name'] = t_match['true_name'].fillna('')
    t_match['true_pinyin'] = t_match['true_pinyin'].fillna('')

    wifi_dict = dict()
    for i in range(len(t_match)):
        temp = t_match.iloc[i].copy()
        temp['label'] = 1
        if wifi_dict.__contains__(t_match.iloc[i]['wifi']):
            wifi_dict[t_match.iloc[i]['wifi']] += 1
        else:
            wifi_dict[t_match.iloc[i]['wifi']] = 0
        wifi = t_wifi[t_wifi['name'] == t_match.iloc[i]['wifi']]
        if wifi.empty:
            warnings.warn('wifi:{} not found!'.format(t_match.iloc[i]['wifi']))
        else:
            temp['wifi_name'], temp['wifi_pinyin'], temp['wifi'] = wifi.iloc[0]['name'], wifi.iloc[0]['pinyin'], wifi.iloc[0]['id']
        wifitrue = t_pos[t_pos['wifi'] == t_match.iloc[i]['wifi']]
        if wifitrue.empty:
            warnings.warn('wifi:{} not found in Pos!'.format(t_match.iloc[i]['wifi']))
        else:
            shop_no = int(wifi_dict[t_match.iloc[i]['wifi']] / n_p_ratio)
            if shop_no < len(wifitrue):
                wifi_true_shop = t_shop[t_shop['id'] == wifitrue.iloc[shop_no]['match']]
                if wifi_true_shop.empty:
                    warnings.warn('id:{} not found in Shop!'.format(wifitrue.iloc[shop_no]['match']))
                else:
                    temp['true_name'], temp['true_pinyin'] = wifi_true_shop.iloc[0]['name'], wifi_true_shop.iloc[0]['pinyin']
            else:
                warnings.warn('no. {} match of wifi:{} not found!'.format(shop_no, t_match.iloc[i]))
        shop = t_shop[t_shop['id'] == t_match.iloc[i]['match']]
        if shop.empty:
            warnings.warn('id: {} not found!'.format(t_match.iloc[i]['match']))
        else:
            temp['false_name'], temp['false_pinyin'] = shop.iloc[0]['name'], shop.iloc[0]['pinyin']
        t_match.iloc[i] = temp
    t_match.reset_index()
    t_match.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    t_match['label'] = t_match['label'].astype('int32')
    for i in range(len(t_match)):
        temp = t_match.iloc[i].copy()
        temp['index'] = 'ws' + str(i)
        t_match.iloc[i] = temp
    t_match['index'] = t_match['index'].astype('str')
    # print(t_match)
    return t_wifi, t_shop, t_match


def pro4deepmatcher_folds(pois, folds=5):
    t_wifi, t_shop, t_match = pro4magellan_without_folds(pois, redata=False)
    # em.to_csv_metadata(t_match, '{}/linking/matching/deepmatcher/deepmatcher_match.csv'.format(ex_path))
    # m = em.read_csv_metadata('{}/linking/matching/deepmatcher/deepmatcher_match.csv'.format(ex_path), key='index')
    import py_entitymatching as em
    from sklearn.model_selection import StratifiedKFold
    k_fold = StratifiedKFold(n_splits=folds, shuffle=True)
    k = 0
    for train_index, test_index in k_fold.split(t_match, t_match['label']):
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        tra = t_match.iloc[train_index]
        tes = t_match.iloc[test_index]
        k1_fold = StratifiedKFold(n_splits=10, shuffle=True)
        for t_index, v_index in k1_fold.split(tra, tra['label']):
            train = tra.iloc[t_index]
            val = tra.iloc[v_index]
            break
        em.to_csv_metadata(train, '{}/linking/matching/deepmatcher/dm_train_{}.csv'.format(ex_path, k))
        em.to_csv_metadata(val, '{}/linking/matching/deepmatcher/dm_val_{}.csv'.format(ex_path, k))
        em.to_csv_metadata(tes, '{}/linking/matching/deepmatcher/dm_test_{}.csv'.format(ex_path, k))
        k += 1
    return


def pro4searchengine(pois, sources, save=False, pinyin=False):
    se_path = '../src/search engine'
    for source in sources:
        search_result = dict()
        for poi in pois:
            for f_name in os.listdir('{}/data_{}/{}'.format(se_path, source, poi)):
                search_docs = list()
                with open('{}/data_{}/{}/{}'.format(se_path, source, poi, f_name), 'r', encoding='utf-8') as f:
                    flag = 0
                    for line in f.readlines():
                        line = pf.filter_emoji(line.strip(), '')
                        if line:
                            if '$$$' in line:
                                if flag == 0:
                                    temp = [line.replace('$$$', '')]
                                else:
                                    temp.append(line.replace('$$$', ''))
                                flag += 1
                            else:
                                pass
                        else:
                            search_docs.append(temp)
                            flag = 0

                search_result[f_name.replace('.txt', '')] = search_docs
        if save:
            with open('{}/data_{}/wifi_search_result.txt'.format(se_path, source), 'w+', encoding='utf-8') as f:
                for wifi in search_result.keys():
                    content = list()
                    for doc in search_result[wifi]:
                        temp = str()
                        for i in doc:
                            if pinyin:
                                try:
                                    temp += pf.chinese2pinyin(i).lower() + ' '
                                except Exception as e:
                                    print(e, i)
                                    pass
                            else:
                                temp += i.lower() + ' '
                        content.append(temp.strip())
                    f.write('{}\t{}\n'.format(wifi, content))


def pro4se_title(pois, sources, save=False, pinyin=False):
    se_path = '../src/search engine'
    for source in sources:
        search_result = dict()
        for poi in pois:
            for f_name in os.listdir('{}/data_{}/{}'.format(se_path, source, poi)):
                search_docs = list()
                with open('{}/data_{}/{}/{}'.format(se_path, source, poi, f_name), 'r', encoding='utf-8') as f:
                    flag = 0
                    for line in f.readlines():
                        line = pf.filter_emoji(line.strip(), '')
                        if line:
                            if '$$$' in line:
                                if flag == 0:
                                    temp = [line.replace('$$$', '')]
                                else:
                                    temp.append(line.replace('$$$', ''))
                                flag += 1
                            else:
                                pass
                        else:
                            search_docs.append(temp)
                            flag = 0

                search_result[f_name.replace('.txt', '')] = search_docs
        if save:
            with open('{}/data_{}/title_wifi_search_result.txt'.format(se_path, source), 'w+', encoding='utf-8') as f:
                for wifi in search_result.keys():
                    content = list()
                    for doc in search_result[wifi]:
                        if len(doc) == 1:
                            temp = pf.chinese2pinyin(doc[0]).lower()
                        else:
                            temp = pf.chinese2pinyin(doc[1]).lower()
                        content.append(temp.strip())
                    f.write('{}\t{}\n'.format(wifi, content))


def pro4se_clean(pois, sources, save=False, pinyin=False, drop_stop=False):
    raw_data = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    raw_data = raw_data[['ltable_name', 'rtable_name']]
    charset = set()
    for index, row in raw_data.iterrows():
        charset = charset | set(row['ltable_name']) | set(row['rtable_name'])
    print('Get raw charset', len(charset))
    import re
    pattern = re.compile('[\uAC00-\uD7AF\u3040-\u31FF]+')
    charstr = ''.join(charset)
    perserve_chars = pattern.findall(charstr)
    perserve_charset = set()
    for i in perserve_chars:
        perserve_charset = perserve_charset | set(i)

    se_path = '../src/search engine'
    for source in sources:
        search_result = dict()
        for poi in pois:
            for f_name in os.listdir('{}/data_{}/{}'.format(se_path, source, poi)):
                search_docs = list()
                with open('{}/data_{}/{}/{}'.format(se_path, source, poi, f_name), 'r', encoding='utf-8') as f:
                    flag = 0
                    for line in f.readlines():
                        line = pf.filter_emoji(line.strip(), '')
                        if line:
                            if '$$$' in line:
                                if flag == 0:
                                    temp = [line.replace('$$$', '')]
                                else:
                                    temp.append(line.replace('$$$', ''))
                                flag += 1
                            else:
                                pass
                        else:
                            search_docs.append(temp)
                            flag = 0
                search_result[f_name.replace('.txt', '')] = search_docs
        if save:
            with open('{}/data_{}/clean_wifi_search_result.txt'.format(se_path, source), 'w+', encoding='utf-8') as f:
                for wifi in search_result.keys():
                    content = list()
                    for doc in search_result[wifi]:
                        temp = str()
                        for i in doc:
                            i = pf.clean_str(pattern, perserve_charset, i)  # 去日韩文
                            p2 = re.compile(r'\W+')  # 去符号
                            perserve2_chars = p2.findall(charstr)
                            perserve2_charset = set()
                            for _ in perserve2_chars:
                                perserve2_charset = perserve2_charset | set(_)
                            i = pf.clean_str(p2, perserve2_charset, i)
                            if drop_stop:  # 去停
                                stopwords = pf.get_stopwords('../src/search engine/哈工大停用词表.txt', charset)
                                i = pf.drop_stopwords(i, stopwords)
                            if pinyin:
                                try:
                                    temp += pf.chinese2pinyin(i).lower() + ' '
                                except Exception as e:
                                    print(e, i)
                                    pass
                            else:
                                temp += i.lower() + ' '
                        content.append(temp.strip())
                    f.write('{}\t{}\n'.format(wifi, content))


def pro4search_recom(pois, save=True):
    result = dict()
    path = '../src/search recommendation'
    for poi in pois:
        with open('{}/{}.txt'.format(path, poi), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wifi, recs = line.strip().split('\t')
                recs = eval(recs)
                if result.__contains__(wifi):
                    result[wifi] = list(set(result[wifi]) | set(recs))
                else:
                    result[wifi] = recs
    recs = sum(len(result[key]) for key in result.keys())
    print('keys:{}\tavg rec:{}'.format(len(result), recs  / len(result)))
    print('total avg rec:{} / {} = {}'.format(recs, 1021, recs / 1021))
    if save:
        with open('{}/recom.txt'.format(path), 'w', encoding='utf-8') as f:
            for key in result.keys():
                f.write('{}\t{}\n'.format(key, result[key]))


def pro4search_recom_fuzzy(pois, f_list, save=True):
    path = '../src/search recommendation'
    if type(f_list) != list:
        f_list = [f_list]
    for f_title in f_list:
        assert os.path.exists(path + '/fuzzy_' + f_title)
        print('Processing fuzzy recom', f_title)
        result = dict()
        for poi in pois:
            with open('{}/fuzzy_{}/{}.txt'.format(path, f_title, poi), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    wifi, recs = line.strip().split('\t')
                    recs = eval(recs)
                    if result.__contains__(wifi):
                        result[wifi] = list(set(result[wifi]) | set(recs))
                    else:
                        result[wifi] = recs
        print('keys:{}\tavg rec:{}'.format(len(result), sum(len(result[key]) for key in result.keys()) / len(result)))
        if save:
            with open('{}/fuzzy_{}/recom.txt'.format(path, f_title), 'w', encoding='utf-8') as f:
                for key in result.keys():
                    f.write('{}\t{}\n'.format(key, result[key]))


def generate_match_for_our(pois, n_p_ratio=1, redata=True):
    import py_entitymatching as em
    t_wifi, t_shop, t_match = pro4magellan_without_folds(pois, n_p_ratio=n_p_ratio, redata=redata)
    em.to_csv_metadata(t_match, '{}/linking/matching/our/our_match.csv'.format(ex_path))


def generate_match_for_our_pairwise(pois, n_p_ratio=1, redata=True):
    import py_entitymatching as em
    t_wifi, t_shop, t_match = pro4our_pairwise(pois, n_p_ratio=n_p_ratio, redata=redata)
    em.to_csv_metadata(t_match, '{}/linking/matching/our/our_match_pairwise.csv'.format(ex_path))


def statistic4searchengine(pois, sources, doc_length_thres=256):
    se_path = '../src/search engine'
    for source in sources:
        search_result = dict()
        for poi in pois:
            for f_name in os.listdir('{}/data_{}/{}'.format(se_path, source, poi)):
                search_docs = list()
                with open('{}/data_{}/{}/{}'.format(se_path, source, poi, f_name), 'r', encoding='utf-8') as f:
                    flag = 0
                    for line in f.readlines():
                        line = pf.filter_emoji(line.strip(), '')
                        if line:
                            if '$$$' in line:
                                if flag == 0:
                                    temp = [line.replace('$$$', '')]
                                else:
                                    temp.append(line.replace('$$$', ''))
                                flag += 1
                            else:
                                pass
                        else:
                            search_docs.append(temp)
                            flag = 0

                search_result[f_name.replace('.txt', '')] = search_docs
        total_docs = 0
        total_snippet = 0
        available_docs = 0
        avg_snippet_len = 0
        wifi_num = 0
        for wifi in search_result.keys():
            wifi_num += 1
            for doc in search_result[wifi]:
                total_docs += 1
                if len(doc) == 3:
                    total_snippet += 1
                    temp = pf.chinese2pinyin(doc[0])
                    if len(temp) <= doc_length_thres:
                        available_docs += 1
                    avg_snippet_len += len(temp)
        avg_snippet_len /= total_snippet
        print('snippet available {} / {} = {}'.format(available_docs, total_snippet, available_docs / total_snippet))
        print('avg snippet len =', avg_snippet_len)
        print('snippet / doc = {} / {} = {}'.format(total_snippet, total_docs, total_snippet / total_docs))
        print('docs / wifi = {} / {} = {}'.format(total_docs, wifi_num, total_docs / wifi_num))


def statistic4se(sources, clean=True):
    f_name = 'wifi_search_result.txt'
    if clean:
        f_name = 'clean_' + f_name
    se_path = '../src/search engine'
    for source in sources:
        wifi_num, doc_num, doc_len = 0, 0, 0
        with open('{}/data_{}/{}'.format(se_path, source, f_name), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wifi, docs = line.strip().split('\t')
                docs = eval(docs)
                if docs:
                    wifi_num += 1
                    doc_num += len(docs)
                for doc in docs:
                    doc_len += len(doc)
        print('{}:\nwifi with doc: {}\ndoc: {}\ndoc_len: {}'.format(source, wifi_num, doc_num, doc_len / doc_num))
        print('{}: \tavg docnum/wifi: {} / {} = {}'.format(source, doc_num, 1021, doc_num / 1021))


def statistic4rec(fuzzy_rec=False):
    if fuzzy_rec:
        rec_dict = get_search_rec_all_with_fuzzy(pois_global, ['r1', 'r2', 'r3'], statistic=False)
    else:
        rec_dict = get_search_recommendation()
    match = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    for index, i in match.iterrows():
        if i['label'] == 1:
            if rec_dict.__contains__(i['ltable_name']):
                print(i['ltable_name'] + '\t' + i['rtable_name'], end='\t')
                temp_score_dict = dict()
                for rec in rec_dict[i['ltable_name']]:
                    temp_score = sum(pf.jaccard(pf.get_ngram(rec, k, True), pf.get_ngram(i['rtable_name'], k, True))
                                     for k in range(1, 4)) / 3 + 1 / (pf.edit_dis(rec, i['rtable_name']) + 1)
                    temp_score_dict[rec] = temp_score
                for k in sorted(temp_score_dict, key=temp_score_dict.__getitem__, reverse=True):
                    print(k, temp_score_dict[k], end='   ')
                print()
    return


def statistic4paper(pois):
    shop_num, wifi_num = 0, 0
    for poi in pois:
        shops = pd.read_csv('{}/linking/prepro/tables/shops_{}.csv'.format(ex_path, poi))
        print(len(shops))
        shop_num += len(shops)
        wifis = pd.read_csv('{}/linking/all_poi/{}.csv'.format(ex_path, poi))
        print(len(wifis))
        wifi_num += len(wifis)
    print(shop_num, wifi_num)


def statistic4positive():
    match = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    match_dict = dict()
    temp_match = match[match['label']==1]
    print(len(temp_match))
    for i in range(len(temp_match)):
        temp = temp_match.iloc[i]
        if match_dict.__contains__(temp['wifi']):
            match_dict[temp['wifi']] += 1
        else:
            match_dict[temp['wifi']] = 1
    print(match_dict.keys())




if __name__ == '__main__':
    pois = ['39.92451,116.51533', '39.93483,116.45241',  # 这两个是第一批
            '39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
            '39.96333,116.45187', '39.98850,116.41674', '40.00034,116.46960']
    # pois = ['40.00034,116.46960']

    # pro_data(['39.92451,116.51533', '39.93483,116.45241'], remake=False)
    # gene_train_test_val(['39.92451,116.51533', '39.93483,116.45241'], folds=5)
    # pro4magellan_without_folds(['39.92451,116.51533', '39.93483,116.45241'])
    # pro4deepmatcher_folds(['39.92451,116.51533', '39.93483,116.45241'], folds=10)
    # pro4searchengine(['39.92451,116.51533', '39.93483,116.45241'], ['baidu'], save=True, pinyin=True)
    # statistic4searchengine(['39.92451,116.51533', '39.93483,116.45241'], ['baidu'], doc_length_thres=256)
    # statistic4searchengine(pois, ['baidu'], doc_length_thres=256)

    # generate_match_for_our(pois, n_p_ratio=5, redata=False)
    # generate_match_for_our_pairwise(pois, n_p_ratio=5, redata=True)
    # pro4deepmatcher_folds(pois)
    # pro4searchengine(pois, ['baidu'], save=True, pinyin=True)
    # pro4deepmatcher_folds(pois, folds=10)
    # pro4search_recom(pois, save=False)

    # pois = ['39.93483,116.45241']
    # pro_data(pois, remake=True)

    # pro4se_clean(pois, ['baidu'], save=True, pinyin=True, drop_stop=True)
    # pro4se_title(pois, ['baidu'], save=True, pinyin=True)
    # statistic4se(['baidu'], clean=False)
    # statistic4rec(fuzzy_rec=False)

    # get_search_rec_all_with_fuzzy(pois, ['r1', 'r2', 'r3'], statistic=True)

    # statistic4paper(pois)

    statistic4positive()