import readdata as rd


def build_distribution():
    pois = rd.statistic(0)
    path = './src/around/amap'
    shop_list = dict()
    for poi in pois:
        lag, lng = poi.split(',')
        exist = 0
        with open('{}/{}'.format(path, poi), 'r', encoding='utf-8') as f:
            with open('{}/trans data/{}'.format(path, poi), 'w+', encoding='utf-8') as w:
                for line in f.readlines():
                    shop = eval(line.strip())
                    w.write('{}\t{}\n'.format(shop['id'], shop['distance']))
                    if shop_list.__contains__(shop['id']):
                        exist += 1
                        # print('shop = {} existed!'.format(content[2]))
                    else:
                        shop['distance'] = 0
                        shop_list[shop['id']] = shop
        print('{} redundant {}'.format(poi, exist))
        # except Exception as e:
        #     print('Error occurs! {}\n'.format(e))
    with open('{}/shoplist.txt'.format(path), 'w+', encoding='utf-8') as sl:
        for i in shop_list.keys():
            sl.write('{}\t{}\n'.format(i, shop_list[i]))
    print('Got shoplist, leng = {}'.format(len(shop_list)))


def build_new_distribution():
    pois = rd.statistic(0)
    path = './src/around/new amap'
    shop_list = dict()
    for poi in pois:
        exist = 0
        with open('{}/{}'.format(path, poi), 'r', encoding='utf-8') as f:
            with open('{}/trans data/{}'.format(path, poi), 'w+', encoding='utf-8') as w:
                for line in f.readlines():
                    shop = eval(line.strip())
                    w.write('{}\t{}\n'.format(shop['id'], shop['distance']))
                    if shop_list.__contains__(shop['id']):
                        exist += 1
                        # print('shop = {} existed!'.format(content[2]))
                    else:
                        shop['distance'] = 0
                        shop_list[shop['id']] = shop
        print('{} redundant {}'.format(poi, exist))
        # except Exception as e:
        #     print('Error occurs! {}\n'.format(e))
    with open('{}/shoplist.txt'.format(path), 'w+', encoding='utf-8') as sl:
        for i in shop_list.keys():
            sl.write('{}\t{}\n'.format(i, shop_list[i]))
    print('Got shoplist, leng = {}'.format(len(shop_list)))


def read_shop_list():
    path = './src/around/amap'
    shop_list = dict()
    with open('{}/shoplist.txt'.format(path), 'r', encoding='utf-8') as sl:
        for line in sl.readlines():
            shop_id, content = line.strip().split('\t')
            shop_list[shop_id] = eval(content)
    print('Read amap shoplist, leng = {}'.format(len(shop_list)))
    return shop_list


def read_new_shop_list():
    path = './src/around/new amap'
    shop_list = dict()
    with open('{}/shoplist.txt'.format(path), 'r', encoding='utf-8') as sl:
        for line in sl.readlines():
            shop_id, content = line.strip().split('\t')
            shop_list[shop_id] = eval(content)
    print('Read new amap shoplist, leng = {}'.format(len(shop_list)))
    return shop_list


def read_poi_shop(statistic=0):
    poi_shop = dict()
    pois = rd.statistic(0)
    path = './src/around/amap'
    total_shop = 0
    for poi in pois:
        lag, lng = poi.split(',')
        s_dict = dict()
        with open('{}/trans data/{}'.format(path, poi), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                shop_id, distance = line.strip().split('\t')
                s_dict[shop_id] = distance.strip()
        poi_shop[poi] = s_dict
        if statistic != 0:
            print('{}\t{}'.format(poi, len(s_dict)))
            total_shop += len(s_dict)
    if statistic != 0:
        print('average = {} / {} = {}'.format(total_shop, len(pois), total_shop / len(pois)))
    return poi_shop


def read_new_poi_shop(statistic=0):
    poi_shop = dict()
    pois = rd.statistic(0)
    path = './src/around/new amap'
    total_shop = 0
    for poi in pois:
        lag, lng = poi.split(',')
        s_dict = dict()
        with open('{}/trans data/{}'.format(path, poi), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                shop_id, distance = line.strip().split('\t')
                s_dict[shop_id] = distance
        poi_shop[poi] = s_dict
        if statistic != 0:
            print('{}\t{}'.format(poi, len(s_dict)))
            total_shop += len(s_dict)
    if statistic != 0:
        print('average = {} / {} = {}'.format(total_shop, len(pois), total_shop / len(pois)))
    return poi_shop

if __name__ == '__main__':
    # build_distribution()
    # shop_list = read_shop_list()
    # poi_shop = read_poi_shop(1)

    build_new_distribution()
    # shop_list = read_new_shop_list()
    # poi_shop = read_new_poi_shop(1)
    a = 1