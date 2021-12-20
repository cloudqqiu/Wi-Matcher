import numpy as np

import models.model_config as configs
import json

def load_data(data):
    if data == 'search':
        datafile = configs.ru_search_res_data
    elif data == 'recom':
        datafile = configs.ru_query_rec_data
    with open(datafile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def average_num():
    rec_data = load_data('recom')
    search_data = load_data('search')
    rec_count = len(rec_data)
    rec_entries = sum([len(v) for k,v in rec_data.items()])
    rec_avg = np.mean([len(v) for k,v in rec_data.items()] + [0 for _ in range(9683-len(rec_data))])
    search_count = len(search_data)
    search_entries = sum([len(v) for k,v in search_data.items()])
    search_avg = np.mean([len(v) for k,v in search_data.items()] + [0 for _ in range(9683-len(search_data))])

    print('Query Recommendations entries:', rec_entries)
    print('Query Recommendations entries per SSID:', rec_avg)
    print('SSID with QR entries:', rec_count)

    print('Search Results entries:', search_entries)
    print('Search Results entries per SSID:', search_avg)
    print('SSID with SR entries:', search_count)


if __name__ == '__main__':
    average_num()