# dataset for magellan, yandexmaycher, hiem can be generated
# from russian and chinese datasets following the same format
# as specified by the models.

data_path = './data/zh-ssid-venue'

glove_path = '../../data/others/glove.42B.300d.txt'

ru_data_path = '../../data/SSID2ORG'
ru_dataset = f'{ru_data_path}/dataset.csv'
ru_magellan_data_path = f'{ru_data_path}/magellan'
ru_deepmatcher_data_path = f'{ru_data_path}/deepmatcher'
ru_yandexmatcher_data_path = f'{ru_data_path}/yandexmatcher'
ru_hiem_data_path = f'{ru_data_path}/hiem'
ru_transformer_path = f'{ru_data_path}/transformer'
ru_wimatcher_data_path = f'{ru_data_path}/wimatcher'

ru_query_rec_data = f'{ru_data_path}/web_info/ssid_qr.json'
ru_search_res_data = f'{ru_data_path}/web_info/ssid_sr.json'
ru_search_res_path = f'{ru_data_path}/web_info'

zh_data_version = 'use'  # _use or _paper
zh_data_path = '../../data/zh-ssid-venue'
zh_base_dataset = f'{zh_data_path}/match_{zh_data_version}.csv'
zh_magellan_data_path = f'{zh_data_path}/matching/magellan'
zh_deepmatcher_data_path = f'{zh_data_path}/matching/deepmatcher'
zh_yandexmatcher_data_path = f'{zh_data_path}/matching/yandexmatcher'
zh_hiem_data_path = f'{zh_data_path}/matching/HI_ET'
zh_transformer_path = f'{zh_data_path}/transformer'

zh_search_res_path = f'{zh_data_path}/search engine'
zh_query_rec_path = f'{zh_data_path}/search recommendation'
zh_wimatcher_data_path = f'{zh_data_path}/matching/wimatcher'
