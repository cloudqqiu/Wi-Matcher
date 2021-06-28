import readdata as rd


def build_distribution():
    pois = rd.statistic(0)
    path = './src/around/new dp'
    shop_list = dict()
    for poi in pois:
        lag, lng = poi.split(',')
        flag = str(float(lag))
        flng = str(float(lng))
        # try:
        exist = 0
        with open('{}/data/{}-{}.txt'.format(path, flag, flng), 'r', encoding='utf-8') as f:
            with open('{}/trans data/{},{}.txt'.format(path, lag, lng), 'w+', encoding='utf-8') as w:
                for line in f.readlines():
                    line = line.strip()
                    content = line.split('\t')
                    shop = dict()
                    shop['name'] = content[3]
                    shop['branch'] = content[4]
                    shop['region'] = content[5]
                    shop['cateid'] = content[0]
                    shop['cate'] = content[1]
                    w.write('{}\t{}\n'.format(content[2], content[6]))
                    if shop_list.__contains__(content[2]):
                        exist += 1
                        # print('shop = {} existed!'.format(content[2]))
                    else:
                        shop_list[content[2]] = shop
        print('{} redundant {}'.format(poi, exist))
        # except Exception as e:
        #     print('Error occurs! {}\n'.format(e))
    with open('{}/shoplist.txt'.format(path), 'w+', encoding='utf-8') as sl:
        for i in shop_list.keys():
            sl.write('{}\t{}\n'.format(i, shop_list[i]))
    print('Got shoplist, leng = {}'.format(len(shop_list)))


def read_shop_list():
    path = './src/around/new dp'
    shop_list = dict()
    with open('{}/shoplist.txt'.format(path), 'r', encoding='utf-8') as sl:
        for line in sl.readlines():
            shop_id, content = line.strip().split('\t')
            content = eval(content)
            shop_list[shop_id] = content
    print('Read dp shoplist, leng = {}'.format(len(shop_list)))
    return shop_list


def read_poi_shop(statistic=0):
    poi_shop = dict()
    pois = rd.statistic(0)
    path = './src/around/new dp'
    total_shop = 0
    for poi in pois:
        lag, lng = poi.split(',')
        s_dict = dict()
        with open('{}/trans data/{},{}.txt'.format(path, lag, lng), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                s_id, distance = line.split('\t')
                s_dict[s_id] = distance.strip()
        poi_shop[poi] = s_dict
        if statistic != 0:
            print('{}\t{}'.format(poi, len(s_dict)))
            total_shop += len(s_dict)
    if statistic != 0:
        print('average = {} / {} = {}'.format(total_shop, len(pois), total_shop / len(pois)))
    return poi_shop


def read_poi_address(statistic=0):
    poi_addr = dict()
    path = './src/around/new dp'
    with open('{}/shopinfo.txt'.format(path), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            poi, addr = line.split('\t')
            addr = eval(addr)
            poi_addr[poi] = addr
    if statistic:
        print('addr length = {}'.format(len(poi_addr)))
    return poi_addr


def read_shop_poi(statistic=0):
    shop_poi = dict()
    path = './src/around/new dp'
    with open('{}/shoppoi.txt'.format(path), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            shop, content, poi = line.split('\t')
            shop_poi[shop] = [content, poi.strip()]
    if statistic:
        print('shoppoi length = {}'.format(len(shop_poi)))
    return shop_poi


def read_poi_phone(statistic=0):
    poi_phone = dict()
    path = './src/around/new dp'
    with open('{}/shopphone.txt'.format(path), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            poi, phones = line.split('\t')
            phones = phones.split(' ')
            poi_phone[poi] = phones
    if statistic:
        print('phone length = {}'.format(len(poi_phone)))
    return poi_phone

def read_detailed_poi_info(statistic=0):
    poi_info = dict()
    path = './src/around/new dp'
    with open('{}/detailedpoiinfo.txt'.format(path), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            poi, info = line.split('\t')
            if info != 'NA':
                info = eval(info)
            poi_info[poi] = info
    if statistic:
        print('info length = {}'.format(len(poi_info)))
    return poi_info

if __name__ == '__main__':
    # build_distribution()
    shop_list = read_shop_list()
    poi_shop = read_poi_shop(1)
    a = 1