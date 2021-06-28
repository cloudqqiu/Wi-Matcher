import urllib
import urllib.request
import time
import re
import random
import json


def test():
    path = '.\src/around/dianping'
    # '14', '15', '17'
    # '10', '30', '25', '50', '45', '85', '20'
    region = ['15']  # 朝阳区14，东城区15，海淀区17
    catagory = ['45']  # 餐饮10， 休闲娱乐30， 电影演出25，丽人50，运动健身45，医疗健康85，购物20

    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
        'Cookie': '_lxsdk_cuid=165b1bc772ec8-0c0e972cdb766e-9393265-1fa400-165b1bc772ec8; _lxsdk=165b1bc772ec8-0c0e972cdb766e-9393265-1fa400-165b1bc772ec8; _hc.v=81a720a1-7fee-e551-ff0b-6ca3dbc1df5b.1536285243; ctu=b463c832c22bcce603621e975306a24fb9b5ca661ee4371dc222861cb86bc2b4; ua=%E6%AC%A7%E9%98%B3%E4%BA%91%E7%A7%8B; dper=b1b9d2eab97d37c91a56199ed43e4a2831153834fe0d6725d59fe4962a639d3e5447d86dcb6bc75fa17465d0566a5b4141fb988bbaccb70b022b99b03ad37d9b98695b2fd4d75606b5503b970545c27b793e4ef603569836fadc8492108ae90d; __utma=1.121435267.1536550328.1536550328.1536550328.1; __utmz=1.1536550328.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); switchcityflashtoast=1; m_flash2=1; default_ab=citylist%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmyinfo%3AA%3A1; cityid=10; cy=2; cye=beijing; s_ViewType=10; ll=7fd06e815b796be3df069dec7836c3df; _lx_utm=utm_source%3DBaidu%26utm_medium%3Dorganic; _lxsdk_s=165d0af9f1a-bc8-535-5e9%7C%7C112'
    }
    for j in region:
        for i in catagory:
            base_url = 'http://www.dianping.com/beijing/ch{}/r{}'.format(i, j)
            for page in range(1, 51):
                url = '{}p{}'.format(base_url, page)
                print('region={}\tcata={}\tpage={}'.format(j, i, page))
                request = urllib.request.Request(url, headers=header)
                with urllib.request.urlopen(request) as f:
                    data = f.read()
                    # print('Data:', data.decode('utf-8'))
                    # print('Data:', data)
                    utf_data = data.decode('utf-8')
                    if '没有找到符合条件的商户～' in utf_data:
                        print('blank!!! go to next cata! ')
                        time.sleep(random.randint(100, 200) / 100)
                        break

                    pattern = re.compile(r'data-shopid=\"\w+\" data-poi=\"\w+\" data-address=\".+\" '
                                         r'data-sname=\"[\u4E00-\u9FA5\w()· ]+\"', re.U)
                    result = pattern.findall(utf_data)
                    with open('{}/{}'.format(path, i), 'a+', encoding='utf-8') as w:
                        for business in result:
                            i_shopid = re.search(r'\"\w+\"', re.search(r'data-shopid=\"\w+\"', business).group()) \
                                .group().replace('\"', '')
                            i_poi = re.search(r'\"\w+\"', re.search(r'data-poi=\"\w+\"', business).group()).group() \
                                .replace('\"', '')
                            i_address = re.search(r'\".+\"', re.search(r'data-address=\".+\" ', business).group()) \
                                .group().replace('\"', '')
                            i_sname = re.search(r'\"[\u4E00-\u9FA5\w()· ]+\"',
                                                re.search(r'data-sname=\"[\u4E00-\u9FA5\w()· ]+\"', business).group()) \
                                .group().replace('\"', '')
                            # print(i_shopid)
                            print(i_shopid, i_poi, i_address, i_sname)
                            w.write('{}\t{}\t{}\t{}\n'.format(i_shopid, i_sname, i_address, i_poi))
                    time.sleep(random.randint(100, 200) / 100)


def get_around_dianping():
    path = '.\src/around/dianping'

    region = ['17']  # 朝阳区14，东城区15，海淀区17
    category = ['10', '30', '25', '50', '45', '85', '20']  # 餐饮10， 休闲娱乐30， 电影演出25，丽人50，运动健身45，医疗健康85，购物20

    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
        'Cookie': '_lxsdk_cuid=165b1bc772ec8-0c0e972cdb766e-9393265-1fa400-165b1bc772ec8; _lxsdk=165b1bc772ec8-0c0e972cdb766e-9393265-1fa400-165b1bc772ec8; _hc.v=81a720a1-7fee-e551-ff0b-6ca3dbc1df5b.1536285243; ctu=b463c832c22bcce603621e975306a24fb9b5ca661ee4371dc222861cb86bc2b4; ua=%E6%AC%A7%E9%98%B3%E4%BA%91%E7%A7%8B; dper=b1b9d2eab97d37c91a56199ed43e4a2831153834fe0d6725d59fe4962a639d3e5447d86dcb6bc75fa17465d0566a5b4141fb988bbaccb70b022b99b03ad37d9b98695b2fd4d75606b5503b970545c27b793e4ef603569836fadc8492108ae90d; __utma=1.121435267.1536550328.1536550328.1536550328.1; __utmz=1.1536550328.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); switchcityflashtoast=1; m_flash2=1; default_ab=citylist%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmyinfo%3AA%3A1; cityid=10; cy=2; cye=beijing; s_ViewType=10; ll=7fd06e815b796be3df069dec7836c3df; _lx_utm=utm_source%3DBaidu%26utm_medium%3Dorganic; _lxsdk_s=165d0af9f1a-bc8-535-5e9%7C%7C112'
    }
    for j in region:
        for i in category:
            print('region = {}\tcategory = {}'.format(j, i))
            base_url = 'http://www.dianping.com/beijing/ch{}/r{}'.format(i, j)
            for page in range(1, 51):
                url = '{}p{}'.format(base_url, page)

                request = urllib.request.Request(url, headers=header)
                with urllib.request.urlopen(request) as f:
                    data = f.read()
                    # print('Data:', data.decode('utf-8'))
                    # print('Data:', data)
                    utf_data = data.decode('utf-8')
                    if '没有找到符合条件的商户～' in utf_data:
                        print('page {} blank!!! go to next cata!'.format(page))
                        time.sleep(random.randint(100, 200) / 100)
                        break

                    pattern = re.compile(r'data-shopid=\"\w+\" data-poi=\"\w+\" data-address=\".+\" '
                                         r'data-sname=\"[\u4E00-\u9FA5\w()· ]+\"', re.U)
                    result = pattern.findall(utf_data)
                    print('page {} get {}'.format(page, result.__len__()))
                    with open('{}/r{}'.format(path, j), 'a+', encoding='utf-8') as w:
                        for business in result:
                            i_shopid = re.search(r'\"\w+\"', re.search(r'data-shopid=\"\w+\"', business).group())\
                                .group().replace('\"', '')
                            i_poi = re.search(r'\"\w+\"', re.search(r'data-poi=\"\w+\"', business).group()).group()\
                                .replace('\"', '')
                            i_address = re.search(r'\".+\"', re.search(r'data-address=\".+\" ', business).group())\
                                .group().replace('\"', '')
                            i_sname = re.search(r'\"[\u4E00-\u9FA5\w()· ]+\"',
                                                re.search(r'data-sname=\"[\u4E00-\u9FA5\w()· ]+\"', business).group())\
                                .group().replace('\"', '')
                            # print(i_shopid)
                            # print(i_shopid, i_poi, i_address, i_sname)
                            w.write('{}\t{}\t{}\t{}\n'.format(i_shopid, i_sname, i_address, i_poi))
                    time.sleep(random.randint(100, 200) / 100)


def read_dp():
    import decode_dp_poi as ddp
    result = dict()
    path = '.\src/around/dianping'
    region = ['14', '15', '17']
    for i in region:
        temp = dict()
        with open('{}/r{}'.format(path, i), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                shop_id, sname, address, poi = line.split('\t')
                try:
                    poi = ddp.lalo2str(ddp.decode(poi.strip()))
                except Exception as e:
                    print(line.strip(), e)
                temp[shop_id] = (sname, address, poi)
        result[i] = temp
    return result


def get_dp_address():
    import api_dp_analyse as ada
    dp_shops = ada.read_shop_list()
    done_poi_addr = ada.read_poi_address(1)
    path = './src/around/new dp'

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # header = {
    #     'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 11_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.0 Mobile/15E148 Safari/604.1',
    #     'Cookie': '_lxsdk_s=1667fd6aecd-ef2-842-058%7C%7C538; default_ab=index%3AA%3A1%7CshopList%3AA%3A1; msource=default; source=m_browser_test_33; m_flash2=1; pvhistory=6L+U5ZuePjo8L2dldGxvY2FsY2l0eWlkP2xhdD0zOC45ODUzODY2Nzc2MjQxOSZsbmc9MTE3LjMzNTQ4MDc5ODAwNjI2JmNvb3JkVHlwZT0xJmNhbGxiYWNrPVdoZXJlQW1JMTE1Mzk3NDM2OTM5MzE+OjwxNTM5NzQzNjk0MDAxXV9b; chwlsource=default; cityid=10; locallat=38.98538667762419; locallng=117.33548079800626; switchcityflashtoast=1; _hc.v=6033be08-e467-31ef-0f94-2cac6c7a605e.1539743068; _lxsdk=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8; _lxsdk_cuid=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8',
    #     'Referer': 'https: // m.dianping.com / tianjin / ch15 / g135d1?from=m_nav_10_KTV',
    #     'Accept - Encoding': 'br, gzip, deflate'
    # }
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
        'Cookie': '_lxsdk_cuid=1661e179052c8-0e82865351d06d-36664c08-144000-1661e179053c8; _lxsdk=1661e179052c8-0e82865351d06d-36664c08-144000-1661e179053c8; _hc.v=16008131-edcd-df42-0ddc-45a2e0be3d4c.1538103153; switchcityflashtoast=1; cy=10; cye=tianjin; _tr.u=9Vl40SLWMfUHdFxO; default_ab=citylist%3AA%3A1%7Cshop%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmap%3AA%3A1; s_ViewType=10; m_flash2=1; cityid=10; _dp.ac.v=3a9d30b6-621a-47e4-a13c-86d5243b2b9b; msource=default; chwlsource=default; logan_custom_report=; source=m_browser_test_33; _lxsdk_s=1669fa6d324-8ca-792-7d2%7C%7C8; logan_session_token=xu4h6ctf81c1pltxxdrw',
        'Referer': 'https://m.dianping.com/tianjin/ch10/d1?from=m_nav_1_meishi',
        'Accept - Encoding': 'gzip, deflate, br'
    }
    with open('{}/errlog.txt'.format(path), 'a+', encoding='utf-8') as el:
        with open('{}/shopinfo.txt'.format(path), 'a+', encoding='utf-8') as w:
            for shop in dp_shops.keys():
                if shop not in done_poi_addr.keys():
                    base_url = 'http://m.dianping.com/shop/{}/map'.format(shop)
                    try:
                        request = urllib.request.Request(base_url, headers=header)
                        with urllib.request.urlopen(request) as f:
                            data = f.read()
                            # print('Data:', data.decode('utf-8'))
                            utf_data = data.decode('utf-8')
                            if '验证中心' in utf_data:
                                print('{}\tvalidate'.format(shop))
                                el.write('{}\t{}\tvalidate\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), shop))
                                return
                            pattern = re.compile(r'"pageInitData":{.+?}')
                            m = pattern.findall(utf_data)[0]
                            m = re.search('{.+}', m)
                            print(m[0])
                            shopinfo = json.loads(m[0])
                            shopinfo.pop('shopId')
                            shopinfo.pop('userLat')
                            shopinfo.pop('userLng')
                            print(shopinfo)
                            w.write('{}\t{}\n'.format(shop, shopinfo))
                    except Exception as e:
                        el.write('{}\t{}\t{}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), shop, e))
                    time.sleep(random.randint(30, 80) / 100)
                # return


def get_dp_poi_try():
    import api_dp_analyse as ada
    import decode_dp_poi as ddp
    dp_shops = ada.read_shop_list()
    done_shop_poi = ada.read_shop_poi(1)
    path = './src/around/new dp'

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    header = {
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 11_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.0 Mobile/15E148 Safari/604.1',
        'Cookie': '_lxsdk_s=1667fd6aecd-ef2-842-058%7C%7C538; default_ab=index%3AA%3A1%7CshopList%3AA%3A1; msource=default; source=m_browser_test_33; m_flash2=1; pvhistory=6L+U5ZuePjo8L2dldGxvY2FsY2l0eWlkP2xhdD0zOC45ODUzODY2Nzc2MjQxOSZsbmc9MTE3LjMzNTQ4MDc5ODAwNjI2JmNvb3JkVHlwZT0xJmNhbGxiYWNrPVdoZXJlQW1JMTE1Mzk3NDM2OTM5MzE+OjwxNTM5NzQzNjk0MDAxXV9b; chwlsource=default; cityid=10; locallat=38.98538667762419; locallng=117.33548079800626; switchcityflashtoast=1; _hc.v=6033be08-e467-31ef-0f94-2cac6c7a605e.1539743068; _lxsdk=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8; _lxsdk_cuid=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8',
        'Referer': 'https: // m.dianping.com / tianjin / ch15 / g135d1?from=m_nav_10_KTV',
        'Accept - Encoding': 'br, gzip, deflate'
    }
    # header = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
    #     'Cookie': '_lxsdk_cuid=1661e179052c8-0e82865351d06d-36664c08-144000-1661e179053c8; _lxsdk=1661e179052c8-0e82865351d06d-36664c08-144000-1661e179053c8; _hc.v=16008131-edcd-df42-0ddc-45a2e0be3d4c.1538103153; switchcityflashtoast=1; cy=10; cye=tianjin; _tr.u=9Vl40SLWMfUHdFxO; default_ab=citylist%3AA%3A1%7Cshop%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmap%3AA%3A1; s_ViewType=10; m_flash2=1; cityid=10; _dp.ac.v=3a9d30b6-621a-47e4-a13c-86d5243b2b9b; msource=default; chwlsource=default; logan_custom_report=; source=m_browser_test_33; _lxsdk_s=1669fa6d324-8ca-792-7d2%7C%7C8; logan_session_token=xu4h6ctf81c1pltxxdrw',
    #     # 'Referer': 'https://m.dianping.com/tianjin/ch10/d1?from=m_nav_1_meishi',
    #     'Accept - Encoding': 'gzip, deflate, br'
    # }
    with open('{}/poierrlog.txt'.format(path), 'a+', encoding='utf-8') as el:
        with open('{}/shoppoi.txt'.format(path), 'a+', encoding='utf-8') as w:
            for shop in dp_shops.keys():
                if shop not in done_shop_poi.keys():
                    base_url = 'https://www.dianping.com/shop/{}/map'.format(shop)
                    print(shop, end='\t')
                    try:
                        request = urllib.request.Request(base_url, headers=header)
                        with urllib.request.urlopen(request) as f:
                            data = f.read()
                            # print('Data:', data.decode('utf-8'))
                            utf_data = data.decode('utf-8')
                            if '验证中心' in utf_data:
                                print('{}\tvalidate'.format(shop))
                                el.write('{}\t{}\tvalidate\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), shop))
                                return
                            pattern = re.compile(r'poi: \'.+\'')
                            m = pattern.findall(utf_data)[0]
                            m = re.search('\'.+\'', m)
                            content = m[0].replace('\'', '')
                            poi = ddp.decode(content)
                            print('{}\t{}\t{},{}'.format(shop, content, poi['lat'], poi['lng']))
                            w.write('{}\t{}\t{},{}\n'.format(shop, content, poi['lat'], poi['lng']))
                    except Exception as e:
                        print('{}\t{}'.format(shop, e))
                        el.write('{}\t{}\t{}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), shop, e))
                    time.sleep(random.randint(30, 80) / 100)
                # return

def get_dp_phone():
    import api_dp_analyse as ada
    dp_shops = ada.read_shop_list()
    done_poi_phone = ada.read_poi_phone(1)
    path = './src/around/new dp'

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # header = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
    #     'Cookie': '_lxsdk_cuid=1661e179052c8-0e82865351d06d-36664c08-144000-1661e179053c8; _lxsdk=1661e179052c8-0e82865351d06d-36664c08-144000-1661e179053c8; _hc.v=16008131-edcd-df42-0ddc-45a2e0be3d4c.1538103153; switchcityflashtoast=1; cy=10; cye=tianjin; _tr.u=9Vl40SLWMfUHdFxO; default_ab=citylist%3AA%3A1%7Cshop%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmap%3AA%3A1; s_ViewType=10; m_flash2=1; cityid=10; _dp.ac.v=3a9d30b6-621a-47e4-a13c-86d5243b2b9b; msource=default; chwlsource=default; logan_custom_report=; source=m_browser_test_33; _lxsdk_s=1669fa6d324-8ca-792-7d2%7C%7C8; logan_session_token=xu4h6ctf81c1pltxxdrw',
    #     'Referer': 'https://m.dianping.com/tianjin/ch10/d1?from=m_nav_1_meishi',
    #     'Accept - Encoding': 'gzip, deflate, br'
    # }
    # header = {
    #     'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 11_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.0 Mobile/15E148 Safari/604.1',
    #     'Cookie': '_lxsdk_s=1667fd6aecd-ef2-842-058%7C%7C538; default_ab=index%3AA%3A1%7CshopList%3AA%3A1; msource=default; source=m_browser_test_33; m_flash2=1; pvhistory=6L+U5ZuePjo8L2dldGxvY2FsY2l0eWlkP2xhdD0zOC45ODUzODY2Nzc2MjQxOSZsbmc9MTE3LjMzNTQ4MDc5ODAwNjI2JmNvb3JkVHlwZT0xJmNhbGxiYWNrPVdoZXJlQW1JMTE1Mzk3NDM2OTM5MzE+OjwxNTM5NzQzNjk0MDAxXV9b; chwlsource=default; cityid=10; locallat=38.98538667762419; locallng=117.33548079800626; switchcityflashtoast=1; _hc.v=6033be08-e467-31ef-0f94-2cac6c7a605e.1539743068; _lxsdk=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8; _lxsdk_cuid=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8',
    #     'Referer': 'https: // m.dianping.com / tianjin / ch15 / g135d1?from=m_nav_10_KTV',
    #     'Accept - Encoding': 'br, gzip, deflate'
    # }
    # header = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134',
    #     'Cookie': 'chwlsource=default; _hc.v=d92cb100-da92-7c51-a0fd-ffa420b71139.1540083884; _lxsdk_cuid=16694271d96c8-06a1486d7bc959-784a5037-144000-16694271d96c8; msource=default; cityid=2; _lxsdk=16694271d96c8-06a1486d7bc959-784a5037-144000-16694271d96c8; switchcityflashtoast=1; default_ab=index%3AA%3A1%7CshopList%3AA%3A1; _lxsdk_s=16694271d97-e26-133-2cc%7C%7C199; locallat=39.92839; locallng=116.45092; geoType=wgs84; pvhistory=6L+U5ZuePjo8L2dldGxvY2FsY2l0eWlkP2xhdD0zOS45MjgzOSZsbmc9MTE2LjQ1MDkyJmNvb3JkVHlwZT0xJmNhbGxiYWNrPVdoZXJlQW1JMTE1NDAwODQwNTAyODM+OjwxNTQwMDg0MDQ3MDc2XV9b; m_flash2=1; source=m_browser_test_33',
    #     'Referer': 'https://m.dianping.com/beijing/ch10/d1?from=m_nav_1_meishi',
    #     'Accept - Encoding': 'gzip, deflate, br'
    # }

    # header = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134',
    #     'Cookie': '_hc.v=7c154953-2732-9e12-c49a-49389afc0ac5.1543569737; _lxsdk_cuid=16763ed05c5c8-082b2572a95183-784a5037-1fa400-16763ed05c5c8; msource=default; cityid=10; _lxsdk=16763ed05c5c8-082b2572a95183-784a5037-1fa400-16763ed05c5c8; default_ab=shop%3AA%3A1; _lxsdk_s=16763ed05c6-d73-674-706%7C%7C24',
    #     'Accept - Encoding': 'br, gzip, deflate'
    # }

    # header = {
    #     'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 11_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.0 Mobile/15E148 Safari/604.1',
    #     'Cookie': '_lxsdk_s=167675cee19-768-8b8-431%7C%7C8; default_ab=shop%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmap%3AA%3A1; msource=default; locallat=38.98537531962521; locallng=117.33562344252648; logan_custom_report=; logan_session_token=g58xsajiyz72z2qbwvml; chwlsource=default; cityid=10; cy=10; cye=tianjin; switchcityflashtoast=1; _hc.v=6033be08-e467-31ef-0f94-2cac6c7a605e.1539743068; _lxsdk=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8; _lxsdk_cuid=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8',
    #     'Accept - Encoding': 'br, gzip, deflate'
    # }

    header = {
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 11_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15F79 MicroMessenger/6.7.3(0x16070321) NetType/WIFI Language/zh_CN',
        'Cookie': '_lxsdk_s=167675cee19-768-8b8-431%7C%7C8; default_ab=shop%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmap%3AA%3A1; msource=default; locallat=38.98537531962521; locallng=117.33562344252648; logan_custom_report=; logan_session_token=g58xsajiyz72z2qbwvml; chwlsource=default; cityid=10; cy=10; cye=tianjin; switchcityflashtoast=1; _hc.v=6033be08-e467-31ef-0f94-2cac6c7a605e.1539743068; _lxsdk=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8; _lxsdk_cuid=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8',
        'Referer': 'https://servicewechat.com/wx734c1ad7b3562129/97/page-frame.html',
        'Accept - Encoding': 'br, gzip, deflate'
    }

    with open('{}/perrlog.txt'.format(path), 'a+', encoding='utf-8') as el:
        with open('{}/shopphone.txt'.format(path), 'a+', encoding='utf-8') as w:
            for shop in dp_shops.keys():
                if shop not in done_poi_phone.keys():
                    base_url = 'http://m.dianping.com/shop/{}'.format(shop)
                    try:
                        request = urllib.request.Request(base_url, headers=header)
                        with urllib.request.urlopen(request) as f:
                            data = f.read()
                            # print('Data:', data.decode('utf-8'))
                            utf_data = data.decode('utf-8')
                            if '验证中心' in utf_data:
                                print('{}\tvalidate'.format(shop))
                                el.write('{}\t{}\tvalidate\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), shop))
                                return
                            pattern = re.compile('tel:\d+')
                            phones = pattern.findall(utf_data)
                            if phones:
                                ps = str()
                                for phone in phones:
                                    p = phone.replace('tel:', '')
                                    ps += p + ' '
                                print('{}\t{}'.format(shop, ps))
                                w.write('{}\t{}\n'.format(shop, ps.strip()))
                            else:
                                w.write('{}\tNA\n'.format(shop))
                    except Exception as e:
                        if '403' in str(e):
                            print('403 now stop!')
                            return
                        el.write('{}\t{}\t{}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), shop, e))
                    time.sleep(random.randint(70, 130) / 100)
                # return


def get_detailed_dp_info():
    import api_dp_analyse as ada
    dp_shops = ada.read_shop_list()
    done_poi_phone = ada.read_detailed_poi_info(1)
    path = './src/around/new dp'

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    header = {
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 11_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15F79 MicroMessenger/6.7.3(0x16070321) NetType/WIFI Language/zh_CN',
        #'Cookie': '_lxsdk_s=167675cee19-768-8b8-431%7C%7C8; default_ab=shop%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmap%3AA%3A1; msource=default; locallat=38.98537531962521; locallng=117.33562344252648; logan_custom_report=; logan_session_token=g58xsajiyz72z2qbwvml; chwlsource=default; cityid=10; cy=10; cye=tianjin; switchcityflashtoast=1; _hc.v=6033be08-e467-31ef-0f94-2cac6c7a605e.1539743068; _lxsdk=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8; _lxsdk_cuid=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8',
        'Referer': 'https://servicewechat.com/wx734c1ad7b3562129/97/page-frame.html',
        'Accept - Encoding': 'br, gzip, deflate'
    }

    with open('{}/derrlog.txt'.format(path), 'a+', encoding='utf-8') as el:
        with open('{}/detailedpoiinfo.txt'.format(path), 'a+', encoding='utf-8') as w:
            for shop in dp_shops.keys():
                if shop not in done_poi_phone.keys():
                    base_url = 'http://mapi.dianping.com/shopping/shopinfo?shopid={}'.format(shop)
                    try:
                        request = urllib.request.Request(base_url, headers=header)
                        with urllib.request.urlopen(request) as f:
                            data = f.read()
                            # print('Data:', data.decode('utf-8'))
                            utf_data = data.decode('utf-8')
                            if '验证中心' in utf_data:
                                print('{}\tvalidate'.format(shop))
                                el.write(
                                    '{}\t{}\tvalidate\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                                shop))
                                return
                            info = json.loads(utf_data)['msg']
                            print('{}\t{}'.format(shop, info))
                            if info:
                                w.write('{}\t{}\n'.format(shop, info))
                            else:
                                w.write('{}\tNA\n'.format(shop))
                    except Exception as e:
                        if '403' in str(e):
                            print('403 now stop!')
                            return
                        el.write('{}\t{}\t{}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), shop, e))
                    time.sleep(random.randint(30, 80) / 100)
                    # return

def api_try_dp():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    header = {
        'Host': 'mapi.dianping.com',
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 11_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15F79 MicroMessenger/6.7.3(0x16070321) NetType/WIFI Language/zh_CN',
        # 'Cookie': '_lxsdk_s=167675cee19-768-8b8-431%7C%7C8; default_ab=shop%3AA%3A1%7Cindex%3AA%3A1%7CshopList%3AA%3A1%7Cmap%3AA%3A1; msource=default; locallat=38.98537531962521; locallng=117.33562344252648; logan_custom_report=; logan_session_token=g58xsajiyz72z2qbwvml; chwlsource=default; cityid=10; cy=10; cye=tianjin; switchcityflashtoast=1; _hc.v=6033be08-e467-31ef-0f94-2cac6c7a605e.1539743068; _lxsdk=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8; _lxsdk_cuid=1667fd6aecac8-0127d9002a0a498-3b347953-c0000-1667fd6aecbc8',
        'Referer': 'https://servicewechat.com/wx734c1ad7b3562129/97/page-frame.html',
        'microMsgVersion': '6.7.3',
        'platformVersion': '11.4',
        'channel': 'weixin',
        'appVersion': '4.0.0',
        'isMicroMessenger': 'true',
        'phone-model': 'iPad Air 2 (WiFi)< iPad5,3>',
        'network-type': 'wifi',
        'dpid': 'uTKbWmq0qjqT2uwwuQDcbGm5bbPruBhR6rds-IzDus8',
        'appName': 'dianping-wxapp',
        'platform': 'iPhone',
        'token': 'e67265b783c26ba60f2074cb53b85d325c05ff08c1a2349dcf3406aa3e505ba7365bf359365f2ec578ad38226cfce8a384c843432db65523eff09489384dde6a',
        'Connection': 'keep-alive',
        'Accept - Language': 'zh-cn',
        'phone-brand': 'iPhone',
        'Accept': '*/*',
        'Content-Type': 'application/json',
        'Accept-Encoding': 'br, gzip, deflate'
    }
    # base_url = 'https://mapi.dianping.com//searchshop.json?' \
    #            'start=0&categoryid=135&parentCategoryId=15&locatecityid=10' \
    #            '&limit=20&sortid=1&cityid=10&range=-1' \
    #            '&mylat=38.98538667762419&mylng=117.33548079800626' \
    #            '&lat=38.98538667762419&lng=117.33548079800626' \
    #            '&maptype=0&callback=jsonp_1539743886989_94979'

    # base_url = 'http://mapi.dianping.com/mapi/wechat/shop.bin?shopUuid=75093457&cookieId=uTKbWmq0qjqT2uwwuQDcbGm5bbPruBhR6rds-IzDus8&lat=38.98540496826172&lng=117.3355941772461&mtsiReferrer=/pages/detail/detail?shopUuid=75093457&cookieId=uTKbWmq0qjqT2uwwuQDcbGm5bbPruBhR6rds-IzDus8&lat=38.98540496826172&lng=117.3355941772461&_token=eJxNkdtyokAQht+F2+0q5zxDqnIRFHc1StSoiUnlAjwF8RCBqHFr332HGVRuhq97+Lt7+v/rpK2Zc4cRAifPNHBGBVEEc6QkONNbznUlRZKBE6XjhnP3LoUCF5OPIjHQ8TvmVABWhH1AlQmD4uNELf2T8xVOk3A5z2rZPEynn7UvE6zjLDeHA47NxNvZ/GRPR4s3w0IsEAXBqP5JIAacuwVhDkypgigFRnlBzAWGkMkRYEwYwsA4MYQ08SvJgohW2HpEAhOmh56cCVOZaK00CqIVylRWGKi9FQSoK68KbOZDwEnRlysBXBaV9QkC4wuZW8G0gpjKWJQ9uKsV1GolcGUUSr8cqQsRViik7ktNN66nQuZt2M6nV5YUK9Pf0K7OZSD1qg7x/GimF8V8l1A7C1Sha4j0LaY21CXy6vYB3/Zv2Tpg2Xpg2bpQ5o0PJRsnSjZeVFhatn6UbBwp2XhSsnGlZPNuy9YZy9abihZDxR/AN4cA3zyCikvlW4xP5XvFtW/pVVnHuAUVv26se+ObZ1Bxrax5mV+vO4uXW+fOmbd/xuc0e1quHkbet0dbk2bQ/1bHWcP3+/hnNGn/eOOkzvNVP+mlGLceeHf4vUhqvfW5t5+GJzxNR6oZLGJ5qA3O3fN58+dtTYMglI8jVHtU+043m252/mHfP2yeVn2h0uaO13vHXbux9fw+WnpLzCj2jg88DvznVt+bnLwh6710olWPJ/5X+zQmk6E/yvM3XO8kr1FtEI7E73Aa7Sfjzm73Vn9+9Ouv3U86aZBfy8WgGdPn2TbqRufOKMjXw+DYWNBVtnCfTi9oeX/v/PsPH48ShQ==&optimus_uuid=1667f9906fac8-faaa11fbbb27b-0-0-1667f9906fac8&optimus_platform=13&optimus_partner=203&optimus_risk_level=71&optimus_code=10'
    base_url = 'http://mapi.dianping.com/shopping/shopinfo?shopid=123343496'  # 96117838

    # base_url = 'https://link.dianping.com/universal-link?shopid=102719685'
    # GET /poi/app/shop/ajax/getExtInfo?shopId=45413336 HTTP/1.1


    request = urllib.request.Request(base_url, headers=header)
    with urllib.request.urlopen(request) as f:
        data = f.read()
        print('Data:', data.decode('utf-8'))
        print('Data:', data)
        utf_data = data.decode('utf-8')
        import json
        a = json.loads(utf_data)
        # print(a['msg'])




if __name__ == "__main__":
    # test()
    # get_around_dianping()
    # dp = read_dp()

    # api_try_dp()

    # get_dp_address()
    # get_dp_phone()
    # get_detailed_dp_info()

    get_dp_poi_try()
    a = 1