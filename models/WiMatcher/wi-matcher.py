import argparse
import itertools
import json

import tensorflow as tf
from keras.layers import *  # Input, SimpleRNN, LSTM, Dense, Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import gc
import numpy as np
import pandas as pd
import math
import time
import sys, os
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

sys.path.append(os.path.abspath('../..'))
from layers import *
import models.common.utils as utils

import models.model_config as config

se_path = f'{config.zh_data_path}/search engine'
MAX_SEQ_LENGTH = 50
MAX_S_SEQ_LENGTH = 256
MIN_S_SEQ_LENGTH = 10
MAX_SR_NUM = 7
MAX_REC_NUM = 3
NN_DIM = 300  # 300 128
NUM_DENSE = 128
NUM_SEC_DENSE = 32
GRAM_LEN_s, CHAR_LEN_s = 0, 0
GRAM_LEN_c, CHAR_LEN_c = 0, 0
embedding_matrix_s, embedding_matrix_c = None, None
BATCH_SIZE = 128
MAX_EPOCHS = 20
MODEL_NAME = None
dropout = 0.2
REC_REPEAT = 128
time_str = None
pois_global = ['39.92451,116.51533', '39.93483,116.45241', '39.86184,116.42517', '39.88892,116.32670',
               '39.90184,116.41196', '39.94735,116.35581', '39.96333,116.45187', '39.98850,116.41674',
               '40.00034,116.46960']


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


def read_search_result(data_type, clean=False, title=False):
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
                docs = docs[:MAX_SR_NUM]
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


def get_sr_respectively(df, search_result, c_name='ssid_raw'):
    result = [list() for _ in range(MAX_SR_NUM)]
    def process(row):
        if search_result.__contains__(row[c_name]):
            if search_result[row[c_name]]:
                for j in range(MAX_SR_NUM):
                    if j < len(search_result[row[c_name]]):
                        result[j].append(search_result[row[c_name]][j])
                    else:
                        result[j].append([np.random.randint(0, GRAM_LEN_c + 1) for _ in
                                          range(np.random.randint(MIN_S_SEQ_LENGTH, MAX_S_SEQ_LENGTH + 1))])
            else:
                for j in range(MAX_SR_NUM):
                    result[j].append([np.random.randint(0, GRAM_LEN_c + 1) for _ in
                                      range(np.random.randint(MIN_S_SEQ_LENGTH, MAX_S_SEQ_LENGTH + 1))])
        else:
            for j in range(MAX_SR_NUM):
                result[j].append([np.random.randint(0, GRAM_LEN_c + 1) for _ in
                                  range(np.random.randint(MIN_S_SEQ_LENGTH, MAX_S_SEQ_LENGTH + 1))])
    df.apply(lambda row: process(row), axis=1)

    for index, i in enumerate(result):
        result[index] = np.array(sequence.pad_sequences(i, maxlen=MAX_S_SEQ_LENGTH, padding='post', truncating='post'))
    return result

def combine_model_v1():
    global MODEL_NAME
    MODEL_NAME = 'combine_v1' + time_str
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    s_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='s_input')
    ws_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_input')

    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_2')
    sr_input_3 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_3')
    sr_input_4 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_4')
    sr_input_5 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_5')
    sr_input_6 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_6')
    # sr_input_7 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_7')
    # sr_input_8 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_8')
    # sr_input_9 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_9')

    rec_input_0 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_0')
    rec_input_1 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_1')
    rec_input_2 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_2')
    # rec_input_3 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_3')
    # rec_input_4 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_4')
    # rec_input_5 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_5')
    # rec_input_6 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_6')
    # rec_input_7 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_7')
    # rec_input_8 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_8')
    # rec_input_9 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_9')

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    wifi_e, shop_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_input)
    r_e_0 = embedding_layer_s(rec_input_0)
    r_e_1 = embedding_layer_s(rec_input_1)
    r_e_2 = embedding_layer_s(rec_input_2)
    # r_e_3 = embedding_layer_s(rec_input_3)
    # r_e_4 = embedding_layer_s(rec_input_4)
    # r_e_5 = embedding_layer_s(rec_input_5)
    # r_e_6 = embedding_layer_s(rec_input_6)
    # r_e_7 = embedding_layer_s(rec_input_7)
    # r_e_8 = embedding_layer_s(rec_input_8)
    # r_e_9 = embedding_layer_s(rec_input_9)

    s_e, ws_e = embedding_layer(s_input), embedding_layer(ws_input)
    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)
    sr_e_3 = embedding_layer(sr_input_3)
    sr_e_4 = embedding_layer(sr_input_4)
    sr_e_5 = embedding_layer(sr_input_5)
    sr_e_6 = embedding_layer(sr_input_6)
    # sr_e_7 = embedding_layer(sr_input_7)
    # sr_e_8 = embedding_layer(sr_input_8)
    # sr_e_9 = embedding_layer(sr_input_9)

    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')  # w-s模块
    wifi, shop = rs(wifi_e), rs(shop_e)
    sim_wifi_shop = subtract([wifi, shop])
    print(K.int_shape(sim_wifi_shop))

    bigru_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    s_r = bigru_r(shop_e)
    r_0 = bigru_r(r_e_0)
    r_1 = bigru_r(r_e_1)
    r_2 = bigru_r(r_e_2)
    # r_3 = bigru_r(r_e_3)
    # r_4 = bigru_r(r_e_4)
    # r_5 = bigru_r(r_e_5)
    # r_6 = bigru_r(r_e_6)
    # r_7 = bigru_r(r_e_7)
    # r_8 = bigru_r(r_e_8)
    # r_9 = bigru_r(r_e_9)

    bigruagg = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    bialignagg = BiAlignAggLayer(nn_dim=NN_DIM, agg_nn=bigruagg)
    r_0 = bialignagg([s_r, r_0])
    r_1 = bialignagg([s_r, r_1])
    r_2 = bialignagg([s_r, r_2])
    # r_3 = bialignagg([s_r, r_3])
    # r_4 = bialignagg([s_r, r_4])
    # r_5 = bialignagg([s_r, r_5])
    # r_6 = bialignagg([s_r, r_6])
    # r_7 = bialignagg([s_r, r_7])
    # r_8 = bialignagg([s_r, r_8])
    # r_9 = bialignagg([s_r, r_9])

    rs = Lambda(utils.get_stack)([r_0, r_1, r_2])  # , r_3, r_4, r_5, r_6, r_7, r_8, r_9])
    # print(K.int_shape(rs))

    s_agg = bigruagg(s_r)  # 使用shop过网络的结果做RECs的聚合
    smatt_sagg = SoftmaxAttLayer(main_tensor=s_agg)
    sim_s_rec = smatt_sagg(rs)

    # weight = dot([s_r, r_0], axes=2)  # 单一rec时层处理
    # weight_j = Lambda(get_softmax_row)(weight)
    # weight_i = Lambda(get_softmax)(weight)
    # weighted_i = dot([weight_i, s_r], axes=1)
    # weighted_j = dot([weight_j, r_0], axes=[2, 1])
    # output_i = subtract([s_r, weighted_j])
    # output_j = subtract([r_0, weighted_i])
    # output_i, output_j = Lambda(get_abs)(output_i), Lambda(get_abs)(output_j)  # 可选 sub加abs
    # output_i, output_j = bigruagg(output_i), bigruagg(output_j)
    # sim_s_rec = average([output_i, output_j])

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')  # ws-s-sr模块
    s = r(s_e)
    ws = r(ws_e)
    ws = Lambda(lambda a: a[:, -1, :])(ws)

    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)
    sr_3 = r(sr_e_3)
    sr_4 = r(sr_e_4)
    sr_5 = r(sr_e_5)
    sr_6 = r(sr_e_6)
    # sr_7 = r(sr_e_7)
    # sr_8 = r(sr_e_8)
    # sr_9 = r(sr_e_9)

    align = AlignSubLayer()
    sr_0 = align([s, sr_0])
    sr_1 = align([s, sr_1])
    sr_2 = align([s, sr_2])
    sr_3 = align([s, sr_3])
    sr_4 = align([s, sr_4])
    sr_5 = align([s, sr_5])
    sr_6 = align([s, sr_6])
    # sr_7 = align([s, sr_7])
    # sr_8 = align([s, sr_8])
    # sr_9 = align([s, sr_9])

    # print(K.int_shape(sr_0))
    ts = Lambda(utils.get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6 ])#, sr_7, sr_8, sr_9])
    # print(K.int_shape(ts))
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)

    # weights = dot([s, sr_0], axes=[2, 2])  # 本段为单一sr时层处理
    # weights = Lambda(get_softmax)(weights)
    # outputs = dot([weights, s], axes=1)
    # outputs = subtract([sr_0, outputs])
    # wt = TimeDistributed(Dense(1, activation=None, use_bias=True))
    # outputs_w = wt(outputs)
    # print(K.int_shape(outputs_w))
    # outputs_w = Lambda(get_softmax)(outputs_w)
    # sr = dot([outputs_w, outputs], axes=1)
    # sr = Lambda(squeeze)(sr)
    # print(K.int_shape(sr))

    sim_con = concatenate([sim_wifi_shop, sim_s_rec, sr])  # 连接三模型
    print(K.int_shape(sim_con))

    sim_con = Dropout(rate=0.4)(sim_con)  # 试把dropout放到后面

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)  # NUM_DENSE * 2
    # score = Dense(NUM_DENSE, activation='relu')(score)
    score = Dense(NUM_SEC_DENSE, activation='relu')(score)
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
                          sr_input_4, sr_input_5, sr_input_6,  #sr_input_7, sr_input_8, sr_input_9,
                          rec_input_0, rec_input_1, rec_input_2],
                  # , rec_input_3, rec_input_4, rec_input_5, rec_input_6,
                  # rec_input_7, rec_input_8, rec_input_9],
                  outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'
    model.summary()
    return model


def extract_data_simple_rec_v2(data_type, process_column, ngram=3, fuzzy_rec=False):  # 带rec进网络 允许读取模糊查询获得的rec
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

    # for i in range(len(raw_data)):  # 转小写
    #     temp = raw_data.iloc[i].copy()
    #     for column in process_column:
    #         temp[column] = temp[column].lower()
    #     raw_data.iloc[i] = temp
    def lower_case(row):
        for col in process_column:
            row[col] = row[col].lower()
        return row

    raw_data = raw_data.apply(lambda row: lower_case(row), axis=1)

    rec_result = [list() for _ in range(MAX_REC_NUM)]

    all_str_list = list(itertools.chain.from_iterable(raw_data[process_column].values))
    charset = set(char for charlist in all_str_list for char in list(charlist))

    def get_grams(str):
        return [str[i:i + ngram] for i in range(len(str) - ngram + 1)] if len(str) >= ngram else [str]

    gramset = set(g for str in all_str_list for g in get_grams(str))

    def rec_result_process(row):
        if rec_dict.__contains__(row['ssid_raw']):
            temp_score_dict = dict()
            done = 0
            for rec in rec_dict[row['ssid_raw']]:
                temp_score = sum( utils.jaccard(utils.get_ngram(rec, k, True), utils.get_ngram(row['venue_raw'], k, True)) for k in range(1, 4)) / 3 + 1 / (utils.edit_dis(rec, row['venue_raw']) + 1)
                temp_score_dict[row['venue_raw']] = temp_score
            for name in sorted(temp_score_dict, key=temp_score_dict.__getitem__, reverse=True):
                if done < MAX_REC_NUM:
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
            while done < MAX_REC_NUM:
                rec_result[done].append(row['ssid'])
                done += 1
        else:
            for _ in range(MAX_REC_NUM):
                rec_result[_].append(row['ssid'])

    raw_data.apply(lambda row: rec_result_process(row), axis=1)

    # for index, row in raw_data.iterrows():
    #     for column in process_column:
    #         charset = charset | set(row[column])
    #         if len(row[column]) < ngram:
    #             grams = [row[column]]
    #         else:
    #             grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
    #         gramset = gramset | set(grams)

    print('extract {}-gram {}'.format(ngram, len(gramset)))
    print('extract character {}'.format(len(charset)))
    global GRAM_LEN_s, CHAR_LEN_s, embedding_matrix_s
    GRAM_LEN_s, CHAR_LEN_s = len(gramset), len(charset)
    embedding_matrix_s = np.zeros((GRAM_LEN_s + 1, CHAR_LEN_s), dtype=int)

    gram2index = {gram: index + 1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}

    for index in index2gram:
        for char in index2gram[index]:
            embedding_matrix_s[index, char2index[char]] += 1

    # new_data = raw_data.copy()
    # for i in range(len(new_data)):
    #     temp = new_data.iloc[i].copy()
    #     for column in process_column:
    #         if len(temp[column]) < ngram:
    #             temp[column] = [gram2index.get(temp[column])]
    #         else:
    #             temp[column] = [gram2index.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
    #     new_data.iloc[i] = temp

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
        rec_result[index] = sequence.pad_sequences(temp, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')

    return raw_data, rec_result


def extract_data_complex(data_type, process_column, ngram=3, need_rec_score=False):
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
    search_result = read_search_result(data_type, clean=False, title=True)

    # for i in range(len(raw_data)):
    #     temp = raw_data.iloc[i].copy()
    #     for column in process_column:
    #         temp[column] = temp[column].lower()
    #     raw_data.iloc[i] = temp

    def lower_case(row):
        for col in process_column:
            row[col] = row[col].lower()
        return row

    raw_data = raw_data.apply(lambda row: lower_case(row), axis=1)

    ws_data = raw_data[['ssid', 'venue']].copy()
    # for i in range(len(ws_data)):
    #     ws_data.iloc[i]['ssid'] = ws_data.iloc[i]['ssid'] + ws_data.iloc[i]['venue']
    ws_data['ssid'] = ws_data.apply(lambda row: row['ssid'] + row['venue'], axis=1)
    ws_data.drop(columns=['venue'], inplace=True)

    # gramset, charset = set(), set()
    # for index, row in raw_data.iterrows():
    #     for column in process_column:
    #         charset = charset | set(row[column])
    #         if len(row[column]) < ngram:
    #             grams = [row[column]]
    #         else:
    #             grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
    #         gramset = gramset | set(grams)
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
    global GRAM_LEN_c, CHAR_LEN_c, embedding_matrix_c
    GRAM_LEN_c, CHAR_LEN_c = len(gramset), len(charset)
    embedding_matrix_c = np.zeros((GRAM_LEN_c + 1, CHAR_LEN_c), dtype=int)
    # embedding_matrix = np.delete(embedding_matrix, [0], axis=1)

    gram2index = {gram: index + 1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}
    # index2char = {char2index[char]: char for char in char2index}

    for index in index2gram:
        for char in index2gram[index]:
            embedding_matrix_c[index, char2index[char]] += 1

    if need_rec_score:
        raise Exception('Rec score not supposed to be used')
        rec_dict = get_search_recommendation()
        new_data['rec_score'] = 0.00001

    # for i in range(len(new_data)):
    #     temp = new_data.iloc[i].copy()
    #     for column in process_column:
    #         if len(temp[column]) < ngram:
    #             temp[column] = [gram2index.get(temp[column])]
    #         else:
    #             temp[column] = [gram2index.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
    #     if need_rec_score:
    #         if rec_dict.__contains__(temp['ssid_raw']):
    #             temp_score_list = list()
    #             for rec in rec_dict[temp['ssid_raw']]:
    #                 temp_score = sum(utils.jaccard(utils.get_ngram(rec, k, True), utils.get_ngram(temp['venue_name'], k, True))
    #                                  for k in range(1, 4)) / 3 + 1 / (utils.edit_dis(rec, temp['venue_name']) + 1)
    #                 temp_score_list.append(temp_score)
    #             sort_score_list = sorted(temp_score_list, reverse=True)
    #             # if len(sort_score_list) > 3:
    #             #     score = sum(sort_score_list[0:3]) / 3
    #             # else:
    #             #     score = sum(sort_score_list) / len(sort_score_list)
    #             score = sort_score_list[0]
    #             temp['rec_score'] += score
    #     new_data.iloc[i] = temp
    def encode(row, cols):
        for col in cols:
            if len(row[col]) < ngram:
                row[col] = [gram2index.get(row[col])]
            else:
                row[col] = [gram2index.get(row[col][j:j + ngram]) for j in range(len(row[col]) - ngram + 1)]
        return row

    raw_data = raw_data.apply(lambda row: encode(row, process_column), axis=1)
    ws_data = ws_data.apply(lambda row: encode(row, ['ssid']), axis=1)

    # for i in range(len(ws_data)):
    #     temp = ws_data.iloc[i].copy()
    #     if len(temp['ssid']) < ngram:
    #         temp['ssid'] = [gram2index.get(temp['ssid'])]
    #     else:
    #         temp['ssid'] = [gram2index.get(temp['ssid'][j:j+ngram]) for j in range(len(temp['ssid']) - ngram + 1)]
    #     ws_data.iloc[i] = temp
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
    global MAX_S_SEQ_LENGTH
    if max_source_length < MAX_S_SEQ_LENGTH:
        MAX_S_SEQ_LENGTH = max_source_length
        print('CHANGE MAX_S_SEQ_LENGTH =', MAX_S_SEQ_LENGTH)
    return raw_data, ws_data, search_result


def our_combine_model_v1(data_type, folds, save_log=False):
    # w-s ws-s-sr s-rec 三个模块结合 方法使用complexv2 和simplev5
    if data_type == 'zh':
        model_save_path = config.zh_wimatcher_data_path
    elif data_type == 'ru':
        model_save_path = config.ru_wimatcher_data_path

    columns = ['ssid', 'venue']
    new_data_s, rec_result = extract_data_simple_rec_v2(data_type, process_column=columns, ngram=5, fuzzy_rec=False)
    new_data, ws_data, search_result = extract_data_complex(data_type, process_column=columns, ngram=5, need_rec_score=False)
    pre_result = list()

    print('Data loaded')

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # tf_config = tf.ConfigProto(device_count={'GPU': 0})
    # sess = tf.Session(config=tf_config)
    # K.set_session(sess)


    k_fold = StratifiedKFold(n_splits=folds, shuffle=True)
    for fold_num, (train_index, test_index) in enumerate(k_fold.split(new_data, new_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))
        new_data_train = new_data.iloc[train_index]
        new_data_s_train = new_data_s.iloc[train_index]

        val_folder = StratifiedKFold(n_splits=10, shuffle=True)
        for t_index, val_index in val_folder.split(new_data_train, new_data_train['label']):
            # print(t_index, val_index)
            train, test, val = dict(), dict(), dict()
            for c in columns:
                tra = np.array(new_data_s_train.iloc[t_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
                tes = np.array(new_data_s.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
                va = np.array(new_data_s_train.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va

            train_ws = np.array(ws_data.iloc[train_index].iloc[t_index]['ssid'])
            train_ws = sequence.pad_sequences(train_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_ws = np.array(ws_data.iloc[test_index]['ssid'])
            test_ws = sequence.pad_sequences(test_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_ws = np.array(ws_data.iloc[train_index].iloc[val_index]['ssid'])
            val_ws = sequence.pad_sequences(val_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_s = np.array(new_data_train.iloc[t_index]['venue'])
            train_s = sequence.pad_sequences(train_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_s = np.array(new_data.iloc[test_index]['venue'])
            test_s = sequence.pad_sequences(test_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_s = np.array(new_data_train.iloc[val_index]['venue'])
            val_s = sequence.pad_sequences(val_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']
            train_sr = get_sr_respectively(new_data_train.iloc[t_index].copy(), search_result)
            test_sr = get_sr_respectively(new_data.iloc[test_index].copy(), search_result)
            val_sr = get_sr_respectively(new_data_train.iloc[val_index].copy(), search_result)
            train_rec, test_rec, val_rec = [0 for _ in range(MAX_REC_NUM)], [0 for _ in range(MAX_REC_NUM)], \
                                           [0 for _ in range(MAX_REC_NUM)]
            for i in range(MAX_REC_NUM):
                train_rec[i] = rec_result[i][train_index][t_index]
                test_rec[i] = rec_result[i][test_index]
                val_rec[i] = rec_result[i][train_index][val_index]

            model = combine_model_v1()  # combine_model_v1  combine_dm_hy
            model_checkpoint_path = f'{model_save_path}/model_checkpoint_{MODEL_NAME}.h5'
            model.fit([train[c] for c in columns] + [train_s, train_ws] + train_sr + train_rec, train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=2,  # 2 1
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + [val_s, val_ws] + val_sr + val_rec, val_label),
                      callbacks=[
                          EarlyStopping(
                              monitor='val_loss',
                              min_delta=0.0001,
                              patience=3,
                              verbose=2,
                              mode='auto',
                          ),
                          # Save the weights of the best epoch.
                          ModelCheckpoint(
                              model_checkpoint_path,
                              monitor='val_loss',
                              save_best_only=True,
                              verbose=2,
                          ),
                      ])
            model.load_weights(model_checkpoint_path)
            test_predict = model.predict([test[c] for c in columns] + [test_s, test_ws] + test_sr + test_rec,
                                         batch_size=BATCH_SIZE, verbose=1)
            t_label = test_label.values
            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if t_label[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                        if save_log:
                            with open(f'{config.zh_data_path}/matching/wimatcher/log/FP-{MODEL_NAME}.log', 'a+',
                                      encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ssid_raw'],
                                                                  new_data.iloc[test_index].iloc[index]['venue_raw']))
                else:
                    if t_label[index] == 1:
                        fn += 1
                        if save_log:
                            with open(f'{config.zh_data_path}/matching/wimatcher/log/FN-{MODEL_NAME}.log', 'a+',
                                      encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ssid_raw'],
                                                                  new_data.iloc[test_index].iloc[index]['venue_raw']))
            print(tp, fp, fn)
            try:
                print(tp / (tp + fp), tp / (tp + fn))
            except Exception as e:
                print(e)
            pre_result.append([tp, fp, fn])

            # return
            K.clear_session()
            del train, test, train_label, test_label
            del model
            gc.collect()
            break
    tp, fp, fn = 0, 0, 0
    for result in pre_result:
        tp += result[0]
        fp += result[1]
        fn += result[2]
    tp /= len(pre_result)
    fp /= len(pre_result)
    fn /= len(pre_result)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print('micro P:', precision)
    print('micro R:', recall)
    print('micro F1:', f1)
    return precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hi-EM model")
    parser.add_argument('--dataset', '-d', required=True, choices=['ru', 'zh'], help='Dataset zh or ru')

    args = parser.parse_args()

    # tf.config.experimental.list_physical_devices('GPU')

    start_time = time.ctime()
    # if sys.platform == 'win32':
    #     start_time = start_time.replace(':', ' ')
    time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
    _p, _r, times = 0, 0, 1
    for i in range(times):
        temp_p, temp_r = our_combine_model_v1(data_type=args.dataset, folds=5, save_log=False)
        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
