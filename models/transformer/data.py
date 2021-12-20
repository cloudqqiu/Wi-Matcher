import itertools
from collections import Counter

import numpy as np
import pandas as pd
from keras.preprocessing import sequence

import models.model_config as config

def select_first_name(row):
    # SSID2ORG dataset 问题： 两个引号、一个字符串中可能有逗号、可能有\u字符（暂时没发现影响）
    return row['names'][2:-2].split('","')[0].replace('\\', '')


def extract_data_simple(data_type, max_seq_len, process_column, ngram=3, remove_char=False):  # 默认参数必须指向不变对象
    if data_type == 'zh':
        raw_data = pd.read_csv(config.zh_base_dataset, delimiter=',', header=0, low_memory=False, encoding='utf-8')
        raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin', 'label', 'ltable_name', 'rtable_name']]
        raw_data.rename(columns={'ltable_pinyin': 'ssid', 'rtable_pinyin': 'venue', 'ltable_name': 'ssid_raw',
                                 'rtable_name': 'venue_raw'}, inplace=True)
    elif data_type == 'ru':
        raw_data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
        raw_data['venue'] = raw_data.apply(select_first_name, axis=1)
        raw_data = raw_data[['ssid', 'venue', 'target']]
        raw_data['ssid_raw'] = raw_data['ssid']
        raw_data['venue_raw'] = raw_data['venue']
        raw_data.rename(columns={'target': 'label'}, inplace=True)

    def lower_case(row):
        for col in process_column:
            row[col] = row[col].lower()
        return row

    raw_data = raw_data.apply(lambda row: lower_case(row), axis=1)

    #########################################
    if remove_char:
        all_str_list = list(itertools.chain.from_iterable(raw_data[process_column].values))
        charset = set(char for charlist in all_str_list for char in list(charlist))
        print("Before remove char: ", len(charset))
        ch_count = Counter([c for str in all_str_list for c in str])
        remove_ch = [k for k,v in ch_count.items() if v <= 1]
        def remove_char(row):
            for col in process_column:
                row[col] = ''.join([i for i in row[col] if i not in remove_ch])
            return row
        raw_data = raw_data.apply(lambda row: remove_char(row), axis=1)
    #########################################

    all_str_list = list(itertools.chain.from_iterable(raw_data[process_column].values))

    charset = set(char for charlist in all_str_list for char in list(charlist))

    def get_grams(str):
        return [str[i:i + ngram] for i in range(len(str) - ngram + 1)] if len(str) >= ngram else [str]

    gramset = set(g for str in all_str_list for g in get_grams(str))

    print('extract {}-gram {}'.format(ngram, len(gramset)))
    print('extract character {}'.format(len(charset)))

    gram_len_s, char_len_s = len(gramset), len(charset)
    embedding_matrix_s = np.zeros((gram_len_s + 1, char_len_s), dtype=int)

    gram2index = {gram: index + 1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}

    for index in index2gram:
        for char in index2gram[index]:
            embedding_matrix_s[index, char2index[char]] += 1

    def encode(row, cols):
        for col in cols:
            if len(row[col]) < ngram:
                row[col] = [gram2index.get(row[col])]
            else:
                row[col] = [gram2index.get(row[col][j:j + ngram]) for j in range(len(row[col]) - ngram + 1)]
        return row

    raw_data = raw_data.apply(lambda row: encode(row, process_column), axis=1)

    np_ssid = raw_data['ssid'].to_numpy()
    np_venue = raw_data['venue'].to_numpy()
    label = raw_data['label']

    # padding
    np_ssid = sequence.pad_sequences(np_ssid, maxlen=max_seq_len, padding='post')
    np_venue = sequence.pad_sequences(np_venue, maxlen=max_seq_len, padding='post')

    return np_ssid, np_venue, label, embedding_matrix_s, gram_len_s, char_len_s