import readdata
from urllib import request
import time


def test():
    key = 'b8fbea8f17574ff5b5e0b3fad5251081'
    types = '090000'  # '050000|060000|070000|080300|080600|090000|100000|120000|140000|160000'
    path = './src/around/amap'

    # wifi_data = readdata.statistic()
    # exist_data = readdata.read_exist_amap()
    place = '39.92432,116.51786'
    max_page = 0
    count = 0
    pois = list()
    print('{}\t{}'.format(1, place))
    url = 'https://restapi.amap.com/v3/place/around?key={}&location={}&types={}&radius=1000&page=1' \
        .format(key, place, types)
    with request.urlopen(url) as f:
        time.sleep(0.02)
        data = f.read()
        print('page: 1\tData:', data.decode('utf-8'))
        d = dict(eval(data.decode('utf-8')))
        count = int(d['count'])
        max_page = 10000  # int(count / 20) + 1
        for poi in d['pois']:
            pois.append(poi)
        for i in range(2, max_page + 1):
            url1 = 'https://restapi.amap.com/v3/place/around?key={}&location={}&types={}&radius=1500&page={}' \
                .format(key, place, types, i)
            with request.urlopen(url1) as f1:
                time.sleep(0.02)
                data1 = f1.read()
                print('page: {}\tData:{}'.format(i, data1.decode('utf-8')))
                d1 = dict(eval(data1.decode('utf-8')))
                if d1['infocode'] == '10003':
                    print('Reach Max Access Times! Try next day!\n')
                    return
            for poi in d1['pois']:
                pois.append(poi)
                # with open('{}\{}'.format(path, place), 'w+', encoding='utf-8') as pf:
                #     for poi in pois:
                #         pf.write(str(poi) + '\n')
                # break


def get_around_amap():
    key = 'b8fbea8f17574ff5b5e0b3fad5251081'
    types = '050000|060000|070000|080300|080600|090000|100000|120000|140000|160000'
    path = './src/around/amap'

    wifi_data = readdata.statistic()
    exist_data = readdata.read_exist_amap()
    for index, place in enumerate(wifi_data):

        # if place != '39.92432,116.51786':  # test
        #     continue

        max_page = 0
        count = 0
        pois = list()
        print('{}\t{}'.format(index, place))
        if place in exist_data:
            print('{} has done'.format(place))
        else:
            url = 'https://restapi.amap.com/v3/place/around?key={}&location={}&types={}&radius=1000&page=1' \
                .format(key, place, types)
            with request.urlopen(url) as f:
                time.sleep(0.02)
                data = f.read()
                print('page: 1\tData:', data.decode('utf-8'))
                d = dict(eval(data.decode('utf-8')))
                count = int(d['count'])
                max_page = int(count / 20) + 1
                for poi in d['pois']:
                    pois.append(poi)
                for i in range(2, max_page + 1):
                    url1 = 'https://restapi.amap.com/v3/place/around?key={}&location={}&types={}&radius=1000&page={}' \
                        .format(key, place, types, i)
                    with request.urlopen(url1) as f1:
                        time.sleep(0.02)
                        data1 = f1.read()
                        print('page: {}\tData:{}'.format(i, data1.decode('utf-8')))
                        d1 = dict(eval(data1.decode('utf-8')))
                        if d1['infocode'] == '10003':
                            print('Reach Max Access Times! Try next day!\n')
                            return
                    for poi in d1['pois']:
                        pois.append(poi)
                with open('{}\{}'.format(path, place), 'w+', encoding='utf-8') as pf:
                    for poi in pois:
                        pf.write(str(poi) + '\n')
            # break


def get_around_amap_seperate():
    key = 'b8fbea8f17574ff5b5e0b3fad5251081'
    types = ['050000', '060000', '070000', '080300', '080600', '090000', '100000', '120000', '140000', '160000']
    path = './src/around/new amap'

    wifi_data = readdata.statistic()
    exist_data = readdata.read_exist_amap_seperate()
    for index, place in enumerate(wifi_data):

        if place != '39.92432,116.51786':  # test
            continue

        pois = list()
        print('{}\t{}'.format(index, place))
        if place in exist_data:
            print('{} has done'.format(place))
        else:
            for ty in types:
                print('type = {}'.format(ty))
                url = 'https://restapi.amap.com/v3/place/around?key={}&location={}&types={}&radius=1000&page=1' \
                    .format(key, place, ty)
                try:
                    with request.urlopen(url) as f:
                        time.sleep(0.02)
                        data = f.read()
                        print('page: 1\tData:', data.decode('utf-8'))
                        d = dict(eval(data.decode('utf-8')))
                        count = int(d['count'])
                        max_page = int(count / 20) + 1
                        for poi in d['pois']:
                            pois.append(poi)
                        for i in range(2, max_page + 1):
                            url1 = 'https://restapi.amap.com/v3/place/around?key={}&location={}&types={}' \
                                   '&radius=1000&page={}'.format(key, place, ty, i)
                            with request.urlopen(url1) as f1:
                                time.sleep(0.02)
                                data1 = f1.read()
                                # print('page: {}\tData:{}'.format(i, data1.decode('utf-8')))
                                d1 = dict(eval(data1.decode('utf-8')))
                                if d1['infocode'] == '10003':
                                    print('Reach Max Access Times! Try next day!\n')
                                    return
                                if d1['count'] == '0' or len(d1['pois']) == 0:
                                    break
                                for poi in d1['pois']:
                                    pois.append(poi)
                except Exception as e:
                    print('{}\t{}'.format(url, e))
            with open('{}\{}'.format(path, place), 'w+', encoding='utf-8') as pf:
                for poi in pois:
                    pf.write(str(poi) + '\n')


if __name__ == "__main__":
    # test()
    # get_around_amap()
    get_around_amap_seperate()