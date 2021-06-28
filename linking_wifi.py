import os
import pandas as pd
import numpy as np
import readdata as rd
import pro_func as pf


def generate_unlabeled(pois):
    poi_wifi = rd.read_ripe_data()
    path = './src/experiment/linking/instance'
    for poi in pois:
        if os.path.exists('{}/{}.csv'.format(path, poi)):
            print('{} file exists in {}  PASS!'.format(poi, path))
        else:
            wifis = poi_wifi[poi][0]
            temp_ar = np.array([[i, 0] for i in wifis])
            temp_df = pd.DataFrame(temp_ar, columns=['wifi', 'match'])
            temp_df.to_csv('{}/{}.csv'.format(path, poi), index=False)
    return


def generate_all_unlabeled_withpoi():
    poi_wifi = rd.read_ripe_data_with_wifipoi()
    path = './src/experiment/linking/all_poi'
    for poi in poi_wifi.keys():
        if os.path.exists('{}/{}.csv'.format(path, poi)):
            print('{} file exists in {}  PASS!'.format(poi, path))
        else:
            wifis = poi_wifi[poi]
            temp_ar = np.array([[i, wifis[i][0], wifis[i][1], 0] for i in wifis.keys()])
            temp_df = pd.DataFrame(temp_ar, columns=['wifi', 'lat', 'lng', 'match'])
            temp_df.to_csv('{}/{}.csv'.format(path, poi), index=False)
    return


def help_linking(target_poi):
    import re
    pattern = '^[\u4e00-\u9fa5]+$'
    r_path = './src/experiment/merged'
    path = './src/experiment/linking'
    for poi in target_poi:
        print('start {}'.format(poi))
        try:
            l = pd.read_csv('{}/merged_shop_list_{}.csv'.format(r_path, poi))
            w = pd.read_csv('{}/all_poi/{}.csv'.format(path, poi))
        except Exception as e:
            print(e)
            continue
        shengmu_l, pinyin_l = dict(), dict()
        for j, row_l in l.iterrows():
            shengmu_l[j] = (pf.chinese2shengmu(row_l['a_full_name']), pf.chinese2shengmu(row_l['d_full_name']))
            pinyin_l[j] = (pf.chinese2pinyin(row_l['a_full_name']), pf.chinese2pinyin(row_l['d_full_name']))
        print('Read data and bulid')

        for i, row_w in w.iterrows():
            candi = dict()
            wifi_name = row_w['wifi']
            zhcn = re.match(pattern, wifi_name)
            if zhcn:
                zhcn = zhcn.group()
                for j, row_l in l.iterrows():
                    score = pf.jaccard(row_l['a_full_name'], zhcn) * 0.6
                    dis_score = 1 - pf.distance_poi((row_l['a_lat'], row_l['a_lng']), (row_w['lat'], row_w['lng'])) / 2000
                    if dis_score >= 0:
                        score += dis_score * 0.4
                    if row_l['source'] == 0:
                        score2 = pf.jaccard(row_l['d_full_name'], zhcn) * 0.6
                        dis_score = 1 - pf.distance_poi((row_l['d_lat'], row_l['d_lng']),
                                                        (row_w['lat'], row_w['lng'])) / 2000
                        if dis_score >= 0:
                            score2 += dis_score * 0.4
                        score += score2
                        score /= 2
                    candi[j] = score
            else:
                for j, row_l in l.iterrows():
                    sm = pf.chinese2shengmu(wifi_name)
                    py = pf.chinese2pinyin(wifi_name)
                    score = pf.jaccard(row_l['a_full_name'], wifi_name) * 0.2 + \
                            pf.jaccard(shengmu_l[j][0], sm) * 0.3 + \
                            pf.jaccard(pinyin_l[j][0], py) * 0.3
                    dis_score = 1 - pf.distance_poi((row_l['a_lat'], row_l['a_lng']), (row_w['lat'], row_w['lng'])) / 2000
                    if dis_score >= 0:
                        score += dis_score * 0.2
                    if row_l['source'] == 0:
                        score2 = pf.jaccard(row_l['d_full_name'], wifi_name) * 0.2 + \
                                 pf.jaccard(shengmu_l[j][1], sm) * 0.3 + \
                                 pf.jaccard(pinyin_l[j][1], py) * 0.3
                        dis_score = 1 - pf.distance_poi((row_l['d_lat'], row_l['d_lng']),
                                                        (row_w['lat'], row_w['lng'])) / 2000
                        if dis_score >= 0:
                            score2 += dis_score * 0.2
                        score += score2
                        score /= 2
                    candi[j] = score
            sorted_candi = sorted(candi.items(), key=lambda item: item[1], reverse=True)

            with open('{}/candidate/{}.csv'.format(path, poi), 'a+', encoding='utf-8') as f:
                f.write('{}\n'.format(wifi_name))
                for k in range(15):
                    this = l.iloc[sorted_candi[k][0]]
                    if this['source'] == 0:
                        ids = this['a_id'] + '|' + this['d_id']
                        names = this['a_full_name'] + '|' + this['d_full_name']
                    else:
                        ids = this['a_id']
                        names = this['a_full_name']
                    f.write('{}\t{}\t{}\n'.format(ids, sorted_candi[k][1], names))
                max_score = sorted_candi[0][1]
                temp = 15
                while sorted_candi[temp][1] == max_score and temp < len(sorted_candi):
                    this = l.iloc[sorted_candi[temp][0]]
                    if this['source'] == 0:
                        ids = this['a_id'] + '|' + this['d_id']
                        names = this['a_full_name'] + '|' + this['d_full_name']
                    else:
                        ids = this['a_id']
                        names = this['a_full_name']
                    f.write('{}\t{}\t{}\n'.format(ids, max_score, names))
                    temp += 1
                f.write('\n')

            if i % 10 == 0:
                print('done {}'.format(i))
        # return


if __name__ == '__main__':
    # target_poi = ['39.96333,116.45187', '39.93483,116.45241', '39.99723,116.41830', '39.98850,116.41674',
    #               '39.92451,116.51533', '40.00038,116.41994']  # 第一批
    target_poi = ['39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
                  '39.97564,116.30627', '40.00034,116.46960']
    generate_unlabeled(target_poi)
    # generate_all_unlabeled_withpoi()

    # help_linking(target_poi)