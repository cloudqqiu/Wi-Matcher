import copy
import itertools
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import random
from keras.preprocessing import sequence


import models.model_config as config
from models.common import utils


def get_search_recommendation(data_type):
    result = dict()
    if data_type == 'zh':
        path = config.zh_query_rec_path
        with open('{}/recom.txt'.format(path), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wifi, recs = line.strip().split('\t')
                recs = eval(recs)
                if result.__contains__(wifi):
                    result[wifi] = list(set(result[wifi]) | set(recs))
                else:
                    result[wifi] = recs
    elif data_type == 'ru':
        datafile = config.ru_query_rec_data
        with open(datafile, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            if len(v)!= 0:
                result[k] = v
    return result


def read_search_result(data_type, max_sr_num, clean=False, title=False):
    search_result = dict()
    assert not (title & clean)
    if data_type == 'zh':
        f_name = 'wifi_search_result.txt'
        if clean:
            print('Using clean SE')
            f_name = 'clean_' + f_name
        if title:
            print('Using title SE')
            f_name = 'title_' + f_name
        # for source in sources:
        search_docs = dict()
        source = 'baidu'
        with open('{}/data_{}/{}'.format(config.zh_search_res_path, source, f_name), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wifi, docs = line.strip().split('\t')
                docs = eval(docs)
                # docs = docs[:max_sr_num]
                search_docs[wifi] = docs
        search_result = search_docs
    elif data_type == 'ru':
        datafile = config.ru_search_res_data
        with open(datafile, 'r', encoding='utf-8') as f:
            sr_data = json.load(f)
        if title:
            print('only use search result title text')
            search_result = { ssid: [d['title'] for d in data] for ssid, data in sr_data.items()}
        elif clean:
            clean_ssid_sr_file = f'{config.ru_search_res_path}/clean_ssid_sr.json'
            if os.path.isfile(clean_ssid_sr_file):
                with open(clean_ssid_sr_file, 'r', encoding='utf-8') as f:
                    search_result = json.load(f)
            else:
                import nltk
                nltk.set_proxy('http://127.0.0.1:7890')
                nltk.download('stopwords')
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize

                for ssid, data in tqdm(sr_data.items(), desc='Cleaning search result'):
                    text = ' '.join([data['desc'], data['title']])
                    text_tokens = word_tokenize(text)
                    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
                    search_result[ssid] = ' '.join(tokens_without_sw+[data['link']])
                with open(clean_ssid_sr_file, 'w', encoding='utf-8') as save:
                    json.dump(search_result, save)
        else:
            search_result = {ssid: [' '.join([d['desc'], d['title'], d['link']]) for d in data] for ssid, data in sr_data.items()}

    return search_result


def get_sr_respectively(df, search_result,
                        min_s_seq_len, max_s_seq_len, max_sr_num,
                        gram_len_c, c_name='ssid_raw'):
    result = [list() for _ in range(max_sr_num)]
    def process(row):
        if search_result.__contains__(row[c_name]):
            if search_result[row[c_name]]:
                for j in range(max_sr_num):
                    if j < len(search_result[row[c_name]]):
                        result[j].append(search_result[row[c_name]][j])
                    else:
                        result[j].append([np.random.randint(0, gram_len_c + 1) for _ in
                                          range(np.random.randint(min_s_seq_len, max_s_seq_len + 1))])
            else:
                for j in range(max_sr_num):
                    result[j].append([np.random.randint(0, gram_len_c + 1) for _ in range(np.random.randint(max_s_seq_len, max_s_seq_len + 1))])
        else:
            for j in range(max_sr_num):
                result[j].append([np.random.randint(0, gram_len_c + 1) for _ in range(np.random.randint(max_s_seq_len, max_s_seq_len + 1))])
    df.apply(lambda row: process(row), axis=1)

    for index, i in enumerate(result):
        result[index] = np.array(sequence.pad_sequences(i, maxlen=max_s_seq_len, padding='post', truncating='post'))
    return result


def extract_data_simple_rec_v2(data_type, max_qr_num, max_seq_len, process_column, ngram=3, fuzzy_rec=False):  # 带rec进网络 允许读取模糊查询获得的rec
    if data_type == 'zh':
        raw_data = pd.read_csv(config.zh_base_dataset, delimiter=',', header=0, low_memory=False, encoding='utf-8')
        raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin', 'label', 'ltable_name', 'rtable_name']]
        raw_data.rename(columns={'ltable_pinyin': 'ssid', 'rtable_pinyin': 'venue', 'ltable_name': 'ssid_raw',
                                 'rtable_name': 'venue_raw'}, inplace=True)
    elif data_type == 'ru':
        raw_data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
        raw_data['venue'] = raw_data.apply(utils.select_first_name, axis=1)
        raw_data = raw_data[['ssid', 'venue', 'target']]
        raw_data['ssid_raw'] = raw_data['ssid']
        raw_data['venue_raw'] = raw_data['venue']
        raw_data.rename(columns={'target': 'label'}, inplace=True)

    if fuzzy_rec:
        raise Exception('Fuzzy mode is not supposed to be used.')
    else:
        rec_dict = get_search_recommendation(data_type)

    def lower_case(row):
        for col in process_column:
            row[col] = row[col].lower()
        return row

    raw_data = raw_data.apply(lambda row: lower_case(row), axis=1)


    all_str_list = list(itertools.chain.from_iterable(raw_data[process_column].values))
    charset = set(char for charlist in all_str_list for char in list(charlist))

    def get_grams(str):
        return [str[i:i + ngram] for i in range(len(str) - ngram + 1)] if len(str) >= ngram else [str]

    gramset = set(g for str in all_str_list for g in get_grams(str))

    rec_result = [list() for _ in range(max_qr_num)]
    def rec_result_process(row):
        if rec_dict.__contains__(row['ssid_raw']):
            temp_score_dict = dict()
            done = 0
            for rec in rec_dict[row['ssid_raw']]:
                temp_score = sum( utils.jaccard(utils.get_ngram(rec, k, True), utils.get_ngram(row['venue_raw'], k, True)) for k in range(1, 4)) / 3 + 1 / (utils.edit_dis(rec, row['venue_raw']) + 1)
                temp_score_dict[rec] = temp_score
            for name in sorted(temp_score_dict, key=temp_score_dict.__getitem__, reverse=True):
                if done < max_qr_num:
                    if data_type == 'zh':
                        this_pinyin = utils.chinese2pinyin(name)
                    elif data_type == 'ru':
                        this_pinyin = name
                    charset.update(set(this_pinyin))
                    if len(this_pinyin) < ngram:
                        grams = [this_pinyin]
                    else:
                        grams = [this_pinyin[i:i + ngram] for i in range(len(this_pinyin) - ngram + 1)]
                    gramset.update(set(grams))
                    rec_result[done].append(this_pinyin)
                    done += 1
                else:
                    break
            while done < max_qr_num:
                rec_result[done].append(row['ssid'])
                done += 1
        else:
            for _ in range(max_qr_num):
                rec_result[_].append(row['ssid'])

    raw_data.apply(lambda row: rec_result_process(row), axis=1)

    print('extract {}-gram {}'.format(ngram, len(gramset)))
    print('extract character {}'.format(len(charset)))

    gram_len_s, char_len_s = len(gramset), len(charset)
    embed_mat_s = np.zeros((gram_len_s + 1, char_len_s), dtype=int)

    gram2index = {gram: index + 1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}

    for index in index2gram:
        for char in index2gram[index]:
            embed_mat_s[index, char2index[char]] += 1

    def encode(row):
        for col in process_column:
            if len(row[col]) < ngram:
                row[col] = [gram2index.get(row[col])]
            else:
                row[col] = [gram2index.get(row[col][j:j + ngram]) for j in range(len(row[col]) - ngram + 1)]
        return row

    raw_data = raw_data.apply(lambda row: encode(row), axis=1)

    for index, i in enumerate(rec_result):
        temp = list()
        for rec in i:
            if len(rec) < ngram:
                temp.append([gram2index.get(rec)])
            else:
                temp.append([gram2index.get(rec[j:j + ngram]) for j in range(len(rec) - ngram + 1)])
        rec_result[index] = sequence.pad_sequences(temp, maxlen=max_seq_len, padding='post', truncating='post')

    return raw_data, rec_result, gram_len_s, char_len_s, embed_mat_s


def extract_data_complex(data_type, process_column, max_s_seq_len, max_sr_num,ngram=3, need_rec_score=False):
    if data_type == 'zh':
        raw_data = pd.read_csv(config.zh_base_dataset, delimiter=',', header=0, low_memory=False, encoding='utf-8')
        raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin', 'label', 'ltable_name', 'rtable_name']]
        raw_data.rename(columns={'ltable_pinyin': 'ssid', 'rtable_pinyin': 'venue', 'ltable_name': 'ssid_raw',
                                 'rtable_name': 'venue_raw'}, inplace=True)
    elif data_type == 'ru':
        raw_data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
        raw_data['venue'] = raw_data.apply(utils.select_first_name, axis=1)
        raw_data = raw_data[['ssid', 'venue', 'target']]
        raw_data['ssid_raw'] = raw_data['ssid']
        raw_data['venue_raw'] = raw_data['venue']
        raw_data.rename(columns={'target': 'label'}, inplace=True)

    # lowercase done; 读取去特殊字符停用词的就用clean 只要title就title
    search_result = read_search_result(data_type, max_sr_num, clean=False, title=True)

    def lower_case(row):
        for col in process_column:
            row[col] = row[col].lower()
        return row

    raw_data = raw_data.apply(lambda row: lower_case(row), axis=1)

    ws_data = raw_data[['ssid', 'venue']].copy()
    ws_data['ssid'] = ws_data.apply(lambda row: row['ssid'] + row['venue'], axis=1)
    ws_data.drop(columns=['venue'], inplace=True)

    all_str_list = list(itertools.chain.from_iterable(raw_data[process_column].values))
    charset = set(char for charlist in all_str_list for char in list(charlist))

    def get_grams(str):
        return [str[i:i + ngram] for i in range(len(str) - ngram + 1)] if len(str) >= ngram else [str]

    gramset = set(g for str in all_str_list for g in get_grams(str))

    # for index, row in ws_data.iterrows():
    def ws_data_update_set(row):
        charset.update(set(row['ssid']))
        if len(row['ssid']) < ngram:
            grams = [row['ssid']]
        else:
            grams = [row['ssid'][i:i + ngram] for i in range(len(row['ssid']) - ngram + 1)]
        gramset.update(set(grams))

    ws_data.apply(lambda row: ws_data_update_set(row), axis=1)

    # for source in sources:
    for key in search_result.keys():
        for c in search_result[key]:
            charset.update(set(c))
            if len(c) < ngram:
                grams = [c]
            else:
                grams = [c[i:i + ngram] for i in range(len(c) - ngram + 1)]
            gramset.update(set(grams))

    print('extract {}-gram {}'.format(ngram, len(gramset)))
    print('extract character {}'.format(len(charset)))

    gram_len_c, char_len_c = len(gramset), len(charset)
    embed_mat_c = np.zeros((gram_len_c + 1, char_len_c), dtype=int)

    gram2index = {gram: index + 1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}

    for index in index2gram:
        for char in index2gram[index]:
            embed_mat_c[index, char2index[char]] += 1

    if need_rec_score:
        raise Exception('Rec score not supposed to be used')
        rec_dict = get_search_recommendation()
        new_data['rec_score'] = 0.00001

    def encode(row, cols):
        for col in cols:
            if len(row[col]) < ngram:
                row[col] = [gram2index.get(row[col])]
            else:
                row[col] = [gram2index.get(row[col][j:j + ngram]) for j in range(len(row[col]) - ngram + 1)]
        return row

    raw_data = raw_data.apply(lambda row: encode(row, process_column), axis=1)
    ws_data = ws_data.apply(lambda row: encode(row, ['ssid']), axis=1)


    max_source_length = 0
    # for source in sources:
    for key in search_result.keys():
        this = search_result[key]
        for index, c in enumerate(this):
            if len(c) < ngram:
                this[index] = [gram2index.get(c)]
            else:
                this[index] = [gram2index.get(c[j:j + ngram]) for j in range(len(c) - ngram + 1)]
                if len(this[index]) > max_source_length:
                    max_source_length = len(this[index])
        search_result[key] = this
    if max_source_length < max_s_seq_len:
        max_s_seq_len = max_source_length
        print('CHANGE MAX_S_SEQ_LENGTH =', max_s_seq_len)
    return raw_data, ws_data, search_result, max_s_seq_len, gram_len_c, char_len_c, embed_mat_c


def extract_data_simple(data_type, process_column, ngram=3):  # 默认参数必须指向不变对象
    if data_type == 'zh':
        raw_data = pd.read_csv(config.zh_base_dataset, delimiter=',', header=0, low_memory=False, encoding='utf-8')
        raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin', 'label', 'ltable_name', 'rtable_name']]
        raw_data.rename(columns={'ltable_pinyin': 'ssid', 'rtable_pinyin': 'venue', 'ltable_name': 'ssid_raw',
                                 'rtable_name': 'venue_raw'}, inplace=True)
    elif data_type == 'ru':
        raw_data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
        raw_data['venue'] = raw_data.apply(utils.select_first_name, axis=1)
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

    return raw_data, embedding_matrix_s, gram_len_s, char_len_s


def load_data(data_type):
    if data_type == 'zh':
        raw_data = pd.read_csv(config.zh_base_dataset, delimiter=',', header=0, low_memory=False, encoding='utf-8')
        raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin', 'label', 'ltable_name', 'rtable_name']]
        raw_data.rename(columns={'ltable_pinyin': 'ssid', 'rtable_pinyin': 'venue', 'ltable_name': 'ssid_raw',
                                 'rtable_name': 'venue_raw'}, inplace=True)
    elif data_type == 'ru':
        raw_data = pd.read_csv(config.ru_dataset, delimiter='\t', header=0, low_memory=False, encoding='utf-8')
        raw_data['venue'] = raw_data.apply(utils.select_first_name, axis=1)
        raw_data = raw_data[['ssid', 'venue', 'target']]
        raw_data['ssid_raw'] = raw_data['ssid']
        raw_data['venue_raw'] = raw_data['venue']
        raw_data.rename(columns={'target': 'label'}, inplace=True)
    # search results and query recom results are Dict format. [SSID: TEXT]
    base_data = raw_data[['ssid', 'venue', 'ssid_raw', 'venue_raw']]
    label = raw_data['label']

    search_result = read_search_result(data_type, None, clean=False, title=True)

    rec_result = get_search_recommendation(data_type)

    return base_data, label, search_result, rec_result


def encode_data(data_type, base_data, search_result, rec_result, ngram=3, max_qr_num=3, max_sr_num =7, max_seq_len=50, min_sr_seq_len=10, max_sr_seq_len=256, qr_reorder=True, sr_reorder=True, remove_char=True):
    columns = ['ssid', 'venue']


    # convert ssid and venue string to lowercase
    for col in columns:
        base_data[col] = base_data[col].str.lower()
    base_qr_data = copy.copy(base_data)
    base_sr_data = copy.copy(base_data)

    if remove_char:
        #########################################
        all_str_list = list(itertools.chain.from_iterable(base_data[columns].values))
        charset = set(char for charlist in all_str_list for char in list(charlist))

        base_ch_count = Counter([c for str in all_str_list for c in str])
        if data_type=='zh':
            sr_ch_count = Counter([c for sr_list in search_result.values() for sr in sr_list for c in utils.chinese2pinyin(sr)])
        else:
            sr_ch_count = Counter(
                [c for sr_list in search_result.values() for sr in sr_list for c in sr])
        qr_ch_count = Counter([c for qr_list in rec_result.values() for qr in qr_list for c in qr])

        # qr data are processed together because they are encoded together
        qr_ch_count.update(base_ch_count)
        base_qr_remove_ch = [k for k,v in qr_ch_count.items() if v <= 1]

        def remove_char(row):
            for col in columns:
                row[col] = ''.join([i for i in row[col] if i not in base_qr_remove_ch])
            return row
        base_qr_data = base_data.apply(lambda row: remove_char(row), axis=1)

        for ssid in rec_result:
            for index, qr in enumerate(rec_result[ssid]):
                rec_result[ssid][index] = ''.join([i for i in qr if i not in base_qr_remove_ch])

        # sr data process
        sr_ch_count.update(base_ch_count)
        sr_remove_ch = [k for k, v in sr_ch_count.items() if v <= 1]

        def remove_char(row):
            for col in columns:
                row[col] = ''.join([i for i in row[col] if i not in sr_remove_ch])
            return row
        base_sr_data = base_data.apply(lambda row: remove_char(row), axis=1)

        for ssid in search_result:
            for index, sr in enumerate(search_result[ssid]):
                search_result[ssid][index] = ''.join([i for i in sr if i not in sr_remove_ch])

        #########################################

    # process QR module data
    #  - choose top N rec result
    #  - fill the qr results to N
    #  - update charset and gramset of rec text

    # get char-set and ngram-set of base data (ssid and venue strings)
    qr_all_str_list = list(itertools.chain.from_iterable(base_qr_data[columns].values))
    qr_char_set = set(char for charlist in qr_all_str_list for char in list(charlist))

    def get_grams(str):
        return [str[i:i + ngram] for i in range(len(str) - ngram + 1)] if len(str) >= ngram else [str]
    qr_gram_set = set(g for str in qr_all_str_list for g in get_grams(str))

    qr_data = []

    # Expand QR with copies of the original SSID for QR entries
    def rec_result_process(row):
        cur_qr = []
        if row['ssid_raw'] in rec_result:
            if qr_reorder:
                rec_str_score = dict()
                for rec in rec_result[row['ssid_raw']]:
                    score = sum( utils.jaccard(utils.get_ngram(rec, k, True), utils.get_ngram(row['venue_raw'], k, True)) for k in range(1, 4)) / 3 + 1 / (utils.edit_dis(rec, row['venue_raw']) + 1)
                    rec_str_score[rec] = score
                rec_strings = sorted(rec_str_score, key=lambda k: rec_str_score[k], reverse=True)
            else:
                rec_strings = rec_result[row['ssid_raw']]

            for name in rec_strings[:max_qr_num]:
                if data_type == 'zh':
                    str_letter = utils.chinese2pinyin(name)
                elif data_type == 'ru':
                    str_letter = name
                if str_letter == '':
                    print('str letter is empty, skip')
                    continue
                qr_char_set.update(set(str_letter))
                if len(str_letter) < ngram:
                    grams = [str_letter]
                else:
                    grams = [str_letter[i:i + ngram] for i in range(len(str_letter) - ngram + 1)]
                qr_gram_set.update(set(grams))
                cur_qr.append(str_letter)

        while len(cur_qr) < max_qr_num:
            cur_qr.append(row['ssid'])
        qr_data.append(cur_qr)

    base_qr_data.apply(lambda row: rec_result_process(row), axis=1)
    print('From QR module data:')
    print(' - extract {}-gram {}'.format(ngram, len(qr_gram_set)))
    print(' - extract character {}'.format(len(qr_char_set)))

    # process SR module data
    # get char-set and ngram-set of base data (ssid and venue strings)
    sr_all_str_list = list(itertools.chain.from_iterable(base_sr_data[columns].values))
    sr_char_set = set(char for charlist in sr_all_str_list for char in list(charlist))

    def get_grams(str):
        return [str[i:i + ngram] for i in range(len(str) - ngram + 1)] if len(str) >= ngram else [str]
    sr_gram_set = set(g for str in sr_all_str_list for g in get_grams(str))

    # generate ssid-venue concat strings for sr module
    sv_data = pd.DataFrame(base_sr_data.apply(lambda row: row['ssid']+str(row['venue']), axis=1), columns=['ssid-venue'])

    def sv_data_update_set(row):
        sr_char_set.update(set(row['ssid-venue']))
        if len(row['ssid-venue']) < ngram:
            grams = [row['ssid-venue']]
        else:
            grams = [row['ssid-venue'][i:i + ngram] for i in range(len(row['ssid-venue']) - ngram + 1)]
        sr_gram_set.update(set(grams))

    sv_data.apply(lambda row: sv_data_update_set(row), axis=1)

    for key in search_result.keys():
        for c in search_result[key]:
            sr_char_set.update(set(c))
            if len(c) < ngram:
                grams = [c]
            else:
                grams = [c[i:i + ngram] for i in range(len(c) - ngram + 1)]
            sr_gram_set.update(set(grams))


    print('From SR module data:')
    print(' - extract {}-gram {}'.format(ngram, len(sr_gram_set)))
    print(' - extract character {}'.format(len(sr_char_set)))



    def convert_to_index(char_set, gram_set):
        gram_len, char_len = len(gram_set), len(char_set)
        embed_mat = np.zeros((gram_len + 1, char_len), dtype=int)

        gram2index = {gram: index + 1 for index, gram in enumerate(gram_set)}
        index2gram = {v: k for k, v in gram2index.items()}
        char2index = {char: index for index, char in enumerate(char_set)}
        for index in index2gram:
            for char in index2gram[index]:
                embed_mat[index, char2index[char]] += 1
        return gram_len, char_len, gram2index, embed_mat

    # char and ngram to index

    def encode(row, gram_to_index, selected_cols=columns):
        for col in selected_cols:
            if len(row[col]) < ngram:
                row[col] = [gram_to_index.get(row[col])]
            else:
                row[col] = [gram_to_index.get(row[col][j:j + ngram]) for j in range(len(row[col]) - ngram + 1)]
        return row

    print('Convert QR module data to index')
    # ssid, venue, query recommendation
    qr_gram_len, qr_char_len, qr_gram2index, qr_embed_mat = convert_to_index(qr_char_set, qr_gram_set)

    qr_base_data = copy.copy(base_qr_data)
    qr_base_data.drop(labels=['ssid_raw', 'venue_raw'],axis=1, inplace=True)
    qr_base_data = qr_base_data.apply(lambda row: encode(row, qr_gram2index), axis=1)

    qr_encoded_data = []
    for qr_list in qr_data:
        temp = []
        for qr in qr_list:
            if len(qr) < ngram:
                temp.append([qr_gram2index.get(qr)])
            else:
                temp.append([qr_gram2index.get(qr[j:j + ngram]) for j in range(len(qr) - ngram + 1)])
        qr_encoded_data.append(sequence.pad_sequences(temp, maxlen=max_seq_len, padding='post', truncating='post'))

    print('Convert SR module data to index')
    # ssid, venue, ssid-venue concat, search results
    sr_gram_len, sr_char_len, sr_gram2index, sr_embed_mat = convert_to_index(sr_char_set, sr_gram_set)

    sr_base_data = copy.copy(base_sr_data)
    sr_base_data.drop(labels=['ssid_raw', 'venue_raw'],axis=1, inplace=True)
    sr_base_data = sr_base_data.apply(lambda row: encode(row, sr_gram2index), axis=1)
    sv_data = sv_data.apply(lambda row: encode(row, sr_gram2index, selected_cols=['ssid-venue']), axis=1)

    # Expand SR with random trigrams of random length for SR entries
    sr_encoded_data = []

    sr_len_max = 0
    for _, sr_list in search_result.items():
        for sr in sr_list:
            if len(sr) < ngram:
                temp = [sr_gram2index.get(sr)]
            else:
                temp = [sr_gram2index.get(sr[j:j + ngram]) for j in range(len(sr) - ngram + 1)]
            sr_len_max = max(len(temp), sr_len_max)
    max_sr_seq_len = min(sr_len_max, max_sr_seq_len)

    def sr_process(row):
        cur_sr_list = []
        if row['ssid_raw'] in search_result:
            if sr_reorder:
                search_str_score = dict()
                for s in search_result[row['ssid_raw']]:
                    score = sum(
                        utils.jaccard(utils.get_ngram(s, k, True), utils.get_ngram(row['venue_raw'], k, True)) for k
                        in range(1, 4)) / 3 + 1 / (utils.edit_dis(s, row['venue_raw']) + 1)
                    search_str_score[s] = score
                search_contents = sorted(search_str_score, key=lambda k: search_str_score[k], reverse=True)
            else:
                search_contents = search_result[row['ssid_raw']]
            for sr in search_contents[:max_sr_num]:
                if len(sr) < ngram:
                    cur_sr_list.append([sr_gram2index.get(sr)])
                else:
                    cur_sr_list.append([sr_gram2index.get(sr[j:j + ngram]) for j in range(len(sr) - ngram + 1)])
        all_sr_gram_index = list(sr_gram2index.values())
        while len(cur_sr_list) < max_sr_num:
            cur_sr_list.append([random.choice(all_sr_gram_index) for _ in
                                range(random.randint(min_sr_seq_len, max_sr_seq_len))])
        sr_encoded_data.append(sequence.pad_sequences(cur_sr_list, maxlen=max_sr_seq_len, padding='post', truncating='post'))

    base_sr_data.apply(lambda row: sr_process(row), axis=1)

    np_ssid = qr_base_data['ssid'].to_numpy()
    np_venue = qr_base_data['venue'].to_numpy()
    np_qr_data = np.stack(qr_encoded_data) # padded
    np_sr_sv = sv_data['ssid-venue']
    np_sr_venue = sr_base_data['venue'].to_numpy()
    np_sr_data = np.stack(sr_encoded_data) # padded

    # padding
    np_ssid = sequence.pad_sequences(np_ssid, maxlen=max_seq_len, padding='post')
    np_venue = sequence.pad_sequences(np_venue, maxlen=max_seq_len, padding='post')
    np_sr_sv = sequence.pad_sequences(np_sr_sv, maxlen=max_sr_seq_len, padding='post')
    np_sr_venue = sequence.pad_sequences(np_sr_venue, maxlen=max_sr_seq_len, padding='post')

    return np_ssid, np_venue, np_qr_data, np_sr_venue, np_sr_sv, np_sr_data, \
           qr_gram_len, qr_char_len, qr_embed_mat, \
           sr_gram_len, sr_char_len, max_sr_seq_len, sr_embed_mat