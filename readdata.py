

def read_raw_data():
    result = list()
    with open('./src/poi_wifi_didi.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        for line in f.readlines():
            line = line.strip()
            # print(line)
            poi, wifis, wnum = line.split('\t')
            wnum = int(wnum)
            wifi_list = wifis.split(' ')
            if len(wifi_list) != wnum:
                print('Parse Error!')
                continue
            result.append((poi, wifi_list, wnum))
    return result


def read_ripe_data(echo=False):
    result = dict()
    with open('./src/poi_wifi_didi.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        for line in f.readlines():
            line = line.strip()
            # print(line)
            poi, wifis, wnum = line.split('\t')
            wnum = int(wnum)
            wifi_list = wifis.split(' ')
            if len(wifi_list) != wnum:
                print('Parse Error!')
                continue
            wifi_set = set(wifi_list)
            if echo:
                print('{}\t{} / {}'.format(poi, wifi_set.__len__(), wnum))
            if result.__contains__(poi):
                new_wifi_set = wifi_set | set(result[poi][0])
                result[poi] = (list(new_wifi_set), new_wifi_set.__len__())
            else:
                result[poi] = (list(wifi_set), wifi_set.__len__())
    return result


def read_ripe_data_with_wifipoi():
    result = dict()
    with open('./src/poi_wifi_didi_2.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            temp = dict()
            poi, wifis = line.strip().split('\t')
            wifi_list = wifis.split(' ')
            count = 0
            for wifi in wifi_list:
                lat, lng, wifi_name = wifi.split(',')
                if not temp.__contains__(wifi_name):
                    temp[wifi_name] = (float(lat), float(lng))
            #     else:
            #         count += 1
            # print('{} , {}'.format(poi, count))
            if not result.__contains__(poi):
                result[poi] = temp
            else:
                print('poi exist!!!! {} %s' % poi)
    return result


def read_exist_amap():
    import os
    result = list()
    path = './src/around/amap'
    for f in os.listdir(path):
        result.append(f)
    return result


def read_exist_amap_seperate():
    import os
    result = list()
    path = './src/around/new amap'
    for f in os.listdir(path):
        result.append(f)
    return result


def statistic(echo=True):
    poi_data = read_raw_data()
    pois = set()
    for index, i in enumerate(poi_data):
        if i[0] not in pois:
            pois.add(i[0])
        else:
            if echo:
                print('{} exists! at index={}'.format(i[0], index))
    if echo:
        print('poi num = {}'.format(len(pois)))
    return pois


def statistic_1_2():
    wifi_1 = read_ripe_data()
    wifi_2 = read_ripe_data_with_wifipoi()
    import pro_func as pf
    print(pf.jaccard(wifi_1.keys(), wifi_2.keys()))

if __name__ == '__main__':
    # data1 = read_raw_data()
    # data2 = read_exist_amap()
    statistic()
    # data3 = read_ripe_data(1)
    print()
