import os
from urllib import request
from bs4 import BeautifulSoup
from urlextract import URLExtract
import GetSearchResult.pro_wifi4search as pw4s


def search_baidu(pois):
    path = '../src/search engine/data_baidu'
    wifis = pw4s.get_instanced_wifi(pois)
    for poi in pois:
        print('Processing', poi)
        path_poi = path + '/' + poi
        path_poi_html = path_poi + '(html)'
        if not os.path.exists(path_poi):
            os.mkdir(path_poi)
        if not os.path.exists(path_poi_html):
            os.mkdir(path_poi_html)

        for wifi in wifis[poi]:
            try:
                wifi_q = request.quote(wifi)
                url = "http://www.baidu.com/s?wd=" + wifi_q  # tn=80035161_2_dg& 疑似微软标识
                req = request.Request(url)
                web_doc = request.urlopen(req).read()
            except Exception as e:
                print(e)

            with open('{}/{}.html'.format(path_poi_html, wifi), 'wb') as f:
                f.write(web_doc)
            with open('{}/{}.txt'.format(path_poi, wifi), 'w', encoding='utf-8') as f:
                soup = BeautifulSoup(web_doc, 'lxml')
                # print(soup)
                urls = soup.find_all(class_="result c-container")
                # print(len(urls))
                for item in urls:
                    item_abs = item.find_all(class_="c-abstract")
                    if len(item_abs) == 0:
                        continue
                    for i in item_abs:
                        f.write(i.text + "$$$\n")
                        # print(i.text)

                    item_title = item.find_all(class_="t")
                    for i in item_title:
                        f.write(i.text + "$$$\n")
                        print(i.text)

                    item_url = item.find_all(class_='f13')
                    extractor = URLExtract()
                    for i in item_url:
                        urls = extractor.find_urls(i.text)
                        if urls:
                            temp = urls[0].replace('- 百度快照', '').strip()
                            f.write(temp + "$$$\n")
                            # print(temp)
                    f.write('\n')
                    # print('\n')

        return


if __name__ == '__main__':
    # pois = ['39.92451,116.51533', '39.93483,116.45241',  # 这两个是第一批
    #         '39.86184,116.42517', '39.88892,116.32670', '39.90184,116.41196', '39.94735,116.35581',
    #         '39.96333,116.45187', '39.98850,116.41674', '40.00034,116.46960']
    # search_baidu(pois)  # ['39.96333,116.45187']
    a = 'CU_ziroom'
    b = request.quote(a)
    print(b)
    url = "http://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_enter=1&tn=baiduhome_pg&wd=" + b  # tn=80035161_2_dg& 疑似微软标识

    req = request.Request(url)
    web_doc = request.urlopen(req).read()
    with open('a.html', 'wb') as f:
        f.write(web_doc)
