import readdata as rd
import api_amap_analyse as aaa
import api_dp_analyse as ada
import pro_func as pf
import csv
import pandas as pd
import py_entitymatching as em
import magellan as mgl
import time


def sim_detect():
    pois = rd.statistic(0)
    amap_poi_shop = aaa.read_new_poi_shop(0)
    amap_shop_list = aaa.read_new_shop_list()
    dp_poi_shop = ada.read_poi_shop(0)
    dp_shop_list = ada.read_shop_list()
    path = './src/around'
    print('Data loaded')

    all_sim = 0
    # poi = '39.89304,116.45995'
    for poi in pois:
        count = 0
        for i in amap_poi_shop[poi].keys():
            amap_shop = amap_shop_list[i]
            for j in dp_poi_shop[poi].keys():
                dp_shop = dp_shop_list[j]

                # if amap_shop['name'] == '{}({})'.format(dp_shop['name'], dp_shop['branch']) or \
                #         (dp_shop['branch'] == '' and amap_shop['name'] == dp_shop['name']):
                if amap_shop['name'] == '{}({})'.format(dp_shop['name'], dp_shop['branch']):
                    count += 1
                    continue
                    # print('{}\t{}-{}||'.format(amap_shop['name'], dp_shop['name'], dp_shop['branch']))
                else:
                    if amap_shop['name'].find('(') == -1 and dp_shop['branch'] == '' \
                            and amap_shop['name'] == dp_shop['name']:
                        count += 1
                        continue

        print('{} sim: {}'.format(poi, count))
        all_sim += count
        # return
    print('all simi {} / {} = {}'.format(all_sim, len(pois), all_sim / len(pois)))


def blocking():
    # pois = rd.statistic(0)
    amap_poi_shop = aaa.read_new_poi_shop(0)
    amap_shop_list = aaa.read_new_shop_list()
    dp_poi_shop = ada.read_poi_shop(0)
    dp_shop_list = ada.read_shop_list()
    dp_shop_addr = ada.read_poi_address(1)
    dp_shop_poi = ada.read_shop_poi(1)
    path = './src/experiment'

    # pois = ['39.89136,116.46484']
    pois = ['39.89136,116.46484', '39.89612,116.46855', '39.90062,116.47479']

    headers = ['id', 'label', 'left_Name', 'left_Name_Full', 'left_Addr', 'left_Poi', 'right_Name', 'right_Name_Full', 'right_Addr', 'right_Poi']

    for poi in pois:
        num = 0
        dp_shops = dp_poi_shop[poi]
        amap_shops = amap_poi_shop[poi]
        all_list = list()
        with open('{}/{}.csv'.format(path, poi), 'w+', encoding='utf-8') as f:
            f_csv = csv.writer(f, lineterminator='\n')
            f_csv.writerow(headers)
            for i in amap_shops.keys():
                amap_shop = amap_shop_list[i]
                temp_pairs = list()
                temp_sims = list()
                same = 0
                for j in dp_shops.keys():
                    dp_shop = dp_shop_list[j]
                    amap_shop_name = amap_shop['name'].replace('(', '').replace(')', '')
                    dp_shop_name = '{}{}'.format(dp_shop['name'], dp_shop['branch'])
                    if amap_shop_name == dp_shop_name:
                        same = 1
                        break
                    if dp_shop_addr.__contains__(j):
                        dp_shop_a = dp_shop_addr[j]['address']
                    else:
                        dp_shop_a = 'NA'
                    if dp_shop_poi.__contains__(j):
                        dp_shop_p = dp_shop_poi[j][1]
                    else:
                        dp_shop_p = 'NA.NA'

                    name_sim = pf.jaccard(amap_shop_name, dp_shop_name)
                    sim = pf.jaccard('{}{}'.format(amap_shop_name, amap_shop['address']), '{}{}'.format(dp_shop_name, dp_shop_a))
                    if name_sim >= 0.4:
                        amap_shop_p = pf.exchange_lag_lng(amap_shop['location'])
                        temp_pair = ['{}.{}'.format(i, j), 0, amap_shop_name, amap_shop_name, amap_shop['address'],
                                     amap_shop_p, dp_shop['name'], dp_shop_name, dp_shop_a, dp_shop_p]
                        print(temp_pair)
                        temp_pairs.append(temp_pair)
                        temp_sims.append(sim)
                if same == 0 and len(temp_pairs):
                    print(len(temp_pairs))
                    max_index = temp_sims.index(max(temp_sims))
                    temp_pairs[max_index][1] = 1
                    f_csv.writerows(temp_pairs)


def save_all_table_amap():
    amap_shop_list = aaa.read_new_shop_list()
    path = './src/around/all_amap.csv'

    this_amap = {'id': [], 'Name': [], 'Name_Full': [], 'Addr': [], 'Lat': [], 'Lng': []}
    for shop in amap_shop_list.keys():
        amap_shop = amap_shop_list[shop]
        name = amap_shop['name'].replace('(', '').replace(')', '')
        if amap_shop['location']:
            lat, lng = pf.exchange_lag_lng(amap_shop['location']).split(',')
        else:
            lat, lng = 0, 0
        if amap_shop['address']:
            addr = amap_shop['address']
        else:
            addr = 'NO'
        this_amap['id'].append(shop)
        this_amap['Name'].append(name)
        this_amap['Name_Full'].append(name)
        this_amap['Addr'].append(addr)
        this_amap['Lat'].append(lat)
        this_amap['Lng'].append(lng)
    a = pd.DataFrame(this_amap)
    em.set_key(a, 'id')
    em.to_csv_metadata(a, path)


def save_all_table_dp():
    dp_shop_list = ada.read_shop_list()
    dp_shop_addr = ada.read_poi_address(1)
    dp_shop_poi = ada.read_shop_poi(1)
    path = './src/around/all_dp.csv'

    this_dp = {'id': [], 'Name': [], 'Name_Full': [], 'Addr': [], 'Lat': [], 'Lng': []}
    for shop in dp_shop_list.keys():
        dp_shop = dp_shop_list[shop]
        if dp_shop_addr.__contains__(shop):
            dp_shop_a = dp_shop_addr[shop]['address']
        else:
            dp_shop_a = 'NO'
        if dp_shop_poi.__contains__(shop):
            lat, lng = dp_shop_poi[shop][1].split(',')
        else:
            lat, lng = 0, 0
        this_dp['id'].append(shop)
        this_dp['Name'].append(dp_shop['name'])
        this_dp['Name_Full'].append('{}{}'.format(dp_shop['name'], dp_shop['branch']))
        this_dp['Addr'].append(dp_shop_a)
        this_dp['Lat'].append(lat)
        this_dp['Lng'].append(lng)
    a = pd.DataFrame(this_dp)
    em.set_key(a, 'id')
    em.to_csv_metadata(a, path)


def magellan_blocking_all_data():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    path = './src/around/blocked_all_data.csv'
    A = em.read_csv_metadata('./src/around/all_amap.csv', key='id')
    D = em.read_csv_metadata('./src/around/all_dp.csv', key='id')

    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(mgl.match_black_box)
    C = bb.block_tables(A, D,
                        l_output_attrs=['id', 'Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
                        r_output_attrs=['id', 'Name', 'Name_Full', 'Addr', 'Lat', 'Lng'],
                        n_jobs=-1)

    print(len(C))
    # print(C.head())
    em.to_csv_metadata(C, path)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    # sim_detect()

    # blocking()

    # save_all_table_amap()
    # save_all_table_dp()
    magellan_blocking_all_data()