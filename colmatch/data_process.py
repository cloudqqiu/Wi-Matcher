import sys
import os
sys.path.append(os.path.abspath('..'))
import pro_func as pf
import pandas as pd
import time
import random
import warnings
import WifiShop.data_process as dp
import colmatch.go_process as gp

ex_path = '../src/experiment'


def generate_train_data(pois, test_poi, n_p_ratio=5, redata=False):
    # import py_entitymatching as em
    t_wifi, t_shop, t_match = dp.pro4magellan_without_folds(pois, n_p_ratio=n_p_ratio, redata=redata)
    # em.to_csv_metadata(t_match, '{}/colmatch/our_match.csv'.format(ex_path))
    t_match.to_csv('{}/colmatch/new_generate_data/train_data_{}other_{}.csv'.format(ex_path, len(pois), test_poi))


def generate_test_data(test_poi, test_num=5):
    test_poi_shop = pd.read_csv('{}/linking/prepro/tables/shops_{}.csv'.format(ex_path, test_poi))
    # test_poi_wifi = pd.read_csv('{}/linking/prepro/tables/wifis_{}.csv'.format(ex_path, test_poi), index=False)
    test_poi_pos = pd.read_csv('{}/linking/prepro/pos_{}.csv'.format(ex_path, test_poi))
    done_wifi = dict()
    done_num = 0
    for index, row in test_poi_pos.iterrows():
        if row['wifi'] not in done_wifi:
            if done_num == test_num:
                break
            else:
                done_wifi[row['wifi']] = [row['match']]
                done_num += 1
        else:
            done_wifi[row['wifi']].append(row['match'])
    print('done {} pos'.format(done_num))

    l_py, l_name, r_py, r_name, label = list(), list(), list(), list(), list()
    for d_wifi in done_wifi.keys():
        py, sm = pf.chinese2pyandsm(d_wifi, short=False)
        for match_shop in done_wifi[d_wifi]:
            this_shop = test_poi_shop[test_poi_shop['id'] == match_shop]
            if not this_shop.empty:
                l_name.append(d_wifi)
                l_py.append(py)
                r_name.append(this_shop.iloc[0]['name'])
                r_py.append(this_shop.iloc[0]['pinyin'])
                label.append(1)
        for index, row in test_poi_shop.iterrows():
            if row['id'] not in done_wifi[d_wifi]:
                l_name.append(d_wifi)
                l_py.append(py)
                r_name.append(row['name'])
                r_py.append(row['pinyin'])
                label.append(0)
    test_columns = {'ltable_name': l_name, 'ltable_pinyin': l_py, 'rtable_name': r_name, 'rtable_pinyin': r_py, 'label': label}
    test_data = pd.DataFrame(test_columns)
    test_data.to_csv('{}/colmatch/new_generate_data/test_data_n{}_{}.csv'.format(ex_path, test_num, test_poi))


def generate_test_data_topk(test_poi, test_num=5, k=50):
    test_poi_shop = pd.read_csv('{}/linking/prepro/tables/shops_{}.csv'.format(ex_path, test_poi))
    # test_poi_wifi = pd.read_csv('{}/linking/prepro/tables/wifis_{}.csv'.format(ex_path, test_poi), index=False)
    test_poi_pos = pd.read_csv('{}/linking/prepro/pos_{}.csv'.format(ex_path, test_poi))
    done_wifi = dict()
    done_num = 0
    for index, row in test_poi_pos.iterrows():
        if row['wifi'] not in done_wifi:
            if done_num == test_num:
                break
            else:
                done_wifi[row['wifi']] = [row['match']]
                done_num += 1
        else:
            done_wifi[row['wifi']].append(row['match'])
    print('done {} pos'.format(done_num))

    wifi_shop_score = dict()
    for wifi in done_wifi.keys():
        py, sm = pf.chinese2pyandsm(wifi, short=False)
        temp_this = list()
        for index, row in test_poi_shop.iterrows():
            if row['id'] not in done_wifi[wifi]:
                temp_score = gp.our_neg_stratagy(py, row['pinyin'])
                temp_this.append((index, temp_score))
        temp_this = sorted(temp_this, key=lambda x:x[1], reverse=True)
        temp_this = temp_this[:k]
        wifi_shop_score[wifi] = temp_this

    l_py, l_name, r_py, r_name, label = list(), list(), list(), list(), list()
    for d_wifi in done_wifi.keys():
        py, sm = pf.chinese2pyandsm(d_wifi, short=False)
        for match_shop in done_wifi[d_wifi]:
            this_shop = test_poi_shop[test_poi_shop['id'] == match_shop]
            if not this_shop.empty:
                l_name.append(d_wifi)
                l_py.append(py)
                r_name.append(this_shop.iloc[0]['name'])
                r_py.append(this_shop.iloc[0]['pinyin'])
                label.append(1)
        for i in wifi_shop_score[d_wifi]:
            l_name.append(d_wifi)
            l_py.append(py)
            this_shop = test_poi_shop.iloc[i[0]]
            r_name.append(this_shop['name'])
            r_py.append(this_shop['pinyin'])
            label.append(0)
    test_columns = {'ltable_name': l_name, 'ltable_pinyin': l_py, 'rtable_name': r_name, 'rtable_pinyin': r_py, 'label': label}
    test_data = pd.DataFrame(test_columns)
    test_data.to_csv('{}/colmatch/new_generate_data/top{}_test_data_n{}_{}.csv'.format(ex_path, k, test_num, test_poi))


def generate_data4test(pois, test_poi, test_num=5, n_p_ratio=5, topk=50, redata=False):
    print('start generating train data')
    generate_train_data(pois, test_poi, n_p_ratio, redata)

    print('start generating test data')
    # generate_test_data(test_poi, test_num)

    generate_test_data_topk(test_poi, test_num, k=topk)


if __name__ == '__main__':
    pois = ['39.92451,116.51533', '39.93483,116.45241',  '39.96333,116.45187',
            '39.86184,116.42517',  '39.90184,116.41196', '39.94735,116.35581',
            '39.98850,116.41674', '40.00034,116.46960']
    test_poi = '39.88892,116.32670' #'39.88892,116.32670'  39.96333,116.45187

    generate_data4test(pois, test_poi, test_num=10, n_p_ratio=5, topk=50, redata=False)

    # test_poi_shop = pd.read_csv('{}/linking/prepro/tables/shops_{}.csv'.format(ex_path, test_poi))
    # this_shop = test_poi_shop[test_poi_shop['id'] == '1']
    # # print(this_shop)
    # print(this_shop.empty)
