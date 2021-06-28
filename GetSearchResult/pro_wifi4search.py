
def get_instanced_wifi(pois):
    path = '../src/experiment/linking/instance'
    result = dict()
    for poi in pois:
        instanced_wifi = list()
        try:
            with open('{}/{}.csv'.format(path, poi), 'r', encoding='utf-8') as f:
                out = f.readline()
                for line in f.readlines():
                    wifi, shops = line.strip().split(',')
                    if shops == '0':
                        continue
                    else:
                        instanced_wifi.append(wifi)
        except Exception as e:
            print(e)
        result[poi] = instanced_wifi
    return result


def statistic(pois):
    total_wifi, total_pair = 0, 0
    path = '../src/experiment/linking/instance'
    for poi in pois:
        w, p = 0, 0
        try:
            with open('{}/{}.csv'.format(path, poi), 'r', encoding='utf-8') as f:
                out = f.readline()
                for line in f.readlines():
                    wifi, shops = line.strip().split(',')
                    if shops == '0':
                        continue
                    else:
                        w += 1
                        shops_s = shops.split(';')
                        p += len(shops_s)
        except Exception as e:
            print(e)
        print('{}\t{} / {}'.format(poi, p, w))
        total_wifi += w
        total_pair += p
    print('TOTAL: {} / {} = {}'.format(total_pair, total_wifi, total_pair / total_wifi))

if __name__ == '__main__':
    # a = get_instanced_wifi(['39.92451,116.51533'])
    pois = ['39.92451,116.51533', '39.93483,116.45241',  # 这两个是第一批
            '39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
            '39.96333,116.45187', '39.98850,116.41674', '40.00034,116.46960']
    statistic(pois)
