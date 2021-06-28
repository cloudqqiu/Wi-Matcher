import sys, os
from urllib import request
import json
import time
import random
sys.path.append(os.path.abspath('..'))
import GetSearchResult.pro_wifi4search as pw4s

path = '../src/search recommendation'


def search_recom(pois):
    wifis = pw4s.get_instanced_wifi(pois)
    for poi in pois:
        print('Processing', poi)
        with open('{}/{}.txt'.format(path, poi), 'w', encoding='utf-8') as f:
            for wifi in wifis[poi]:
                try:
                    wifi_q = request.quote(wifi)
                    url = "https://www.baidu.com/sugrec?pre=1&p=3&ie=utf-8&json=1&prod=pc&from=pc_web&wd=" + wifi_q  # tn=80035161_2_dg& 疑似微软标识
                    req = request.Request(url)
                    gap = random.randint(10, 150)
                    time.sleep(gap / 100)
                    res = request.urlopen(req).read().decode('utf-8')
                    j = json.loads(res)
                except Exception as e:
                    print(e)
                if j:
                    if j.__contains__('g'):
                        temp = list()
                        for i in j['g']:
                            temp.append(i['q'])
                        print(wifi, temp)
                        f.write('{}\t{}\n'.format(wifi, temp))
    return


def search_recom_fuzzy(pois, direction, fuzzy_num=1):
    assert direction == 'l' or direction == 'r'
    assert type(fuzzy_num) == int
    wifis = pw4s.get_instanced_wifi(pois)
    f_path = path + '/fuzzy_' + direction + str(fuzzy_num)
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    for poi in pois:
        print('Processing', poi)
        with open('{}/{}.txt'.format(f_path, poi), 'w', encoding='utf-8') as f:
            for wifi in wifis[poi]:
                try:
                    if len(wifi) / 2 <= fuzzy_num:
                        continue
                    if direction == 'l':
                        wifi_f = wifi[fuzzy_num:]
                    if direction == 'r':
                        wifi_f = wifi[:-fuzzy_num]
                    wifi_q = request.quote(wifi_f)
                    url = "https://www.baidu.com/sugrec?pre=1&p=3&ie=utf-8&json=1&prod=pc&from=pc_web&wd=" + wifi_q  # tn=80035161_2_dg& 疑似微软标识
                    req = request.Request(url)
                    gap = random.randint(10, 150)
                    time.sleep(gap / 100)
                    res = request.urlopen(req).read().decode('utf-8')
                    j = json.loads(res)
                except Exception as e:
                    print(e)
                if j:
                    if j.__contains__('g'):
                        temp = list()
                        for i in j['g']:
                            temp.append(i['q'])
                        print(wifi, wifi_f, temp)
                        f.write('{}\t{}\n'.format(wifi, temp))
    return


if __name__ == '__main__':
    pois = ['39.92451,116.51533', '39.93483,116.45241',  # 这两个是第一批
            '39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
            '39.96333,116.45187', '39.98850,116.41674', '40.00034,116.46960']
    # search_baidu(pois)  # ['39.96333,116.45187']
    # search_recom(pois)

    search_recom_fuzzy(pois, direction='r', fuzzy_num=1)
