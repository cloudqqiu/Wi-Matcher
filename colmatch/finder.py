import tensorflow as tf
import keras
from keras.layers import *  # Input, SimpleRNN, LSTM, Dense, Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.activations import relu
from keras import backend as K
import gc
import numpy as np
import pandas as pd
import math
import time
import sys, os
from sklearn.model_selection import StratifiedKFold, KFold
# from sklearn.utils import shuffle
sys.path.append(os.path.abspath('..'))
from WifiShop.utilsLayer import SoftmaxAttLayer, AlignLayer, AlignSubLayer, HighwayLayer, AlignOnlySubLayer, BiAlignAggLayer, BiAlignLayer, InterAttLayer, S2SMLayer
import WifiShop.data_process as dp
import pro_func as pf

ex_path = '../src/experiment'
se_path = '../src/search engine'
SOURCES = ['baidu']
MAX_SEQ_LENGTH = 50
MAX_S_SEQ_LENGTH = 256
MIN_S_SEQ_LENGTH = 10
MAX_SR_NUM = 7
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
MAX_REC_NUM = 3
start_time = None


def extract_data_simple(process_column, test_poi, nother, test_num, ngram=3, need_rec_score=False):  # 默认参数必须指向不变对象
    select_column = ['ltable_pinyin', 'rtable_pinyin'] + ['label'] + ['ltable_name', 'rtable_name']
    raw_train_data = pd.read_csv('{}/colmatch/new_generate_data/train_data_{}other_{}.csv'.format(ex_path, nother, test_poi))
    # raw_test_data = pd.read_csv('{}/colmatch/new_generate_data/test_data_n{}_{}.csv'.format(ex_path, test_num, test_poi))
    # topk选test法
    raw_test_data = pd.read_csv('{}/colmatch/new_generate_data/top50_test_data_n{}_{}.csv'.format(ex_path, test_num, test_poi))
    raw_train_data, raw_test_data = raw_train_data[select_column], raw_test_data[select_column]

    for i in range(len(raw_train_data)):  # 转小写
        temp = raw_train_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        raw_train_data.iloc[i] = temp
    for i in range(len(raw_test_data)):  # 转小写
        temp = raw_test_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        raw_test_data.iloc[i] = temp

    gramset, charset = set(), set()  # 获取ngram
    for index, row in raw_train_data.iterrows():
        for column in process_column:
            charset = charset | set(row[column])
            if len(row[column]) < ngram:
                grams = [row[column]]
            else:
                grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
            gramset = gramset | set(grams)
    for index, row in raw_test_data.iterrows():
        for column in process_column:
            charset = charset | set(row[column])
            if len(row[column]) < ngram:
                grams = [row[column]]
            else:
                grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
            gramset = gramset | set(grams)

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

    new_train_data = raw_train_data.copy()
    if need_rec_score:
        rec_dict = dp.get_search_recommendation()
        new_train_data['rec_score'] = 0
    for i in range(len(new_train_data)):
        temp = new_train_data.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index.get(temp[column])]
            else:
                temp[column] = [gram2index.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        if need_rec_score:
            if rec_dict.__contains__(temp['ltable_name']):
                temp_score_list = list()
                for rec in rec_dict[temp['ltable_name']]:
                    temp_score = sum(pf.jaccard(pf.get_ngram(rec, k, True), pf.get_ngram(temp['rtable_name'], k, True))
                                     for k in range(1, 4)) / 3 + 1 / (pf.edit_dis(rec, temp['rtable_name']) + 1)
                    temp_score_list.append(temp_score)
                sort_score_list = sorted(temp_score_list, reverse=True)
                score = sort_score_list[0]
                temp['rec_score'] = score
        new_train_data.iloc[i] = temp

    new_test_data = raw_test_data.copy()
    if need_rec_score:
        rec_dict = dp.get_search_recommendation()
        new_test_data['rec_score'] = 0
    for i in range(len(new_test_data)):
        temp = new_test_data.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index.get(temp[column])]
            else:
                temp[column] = [gram2index.get(temp[column][j:j + ngram]) for j in range(len(temp[column]) - ngram + 1)]
        if need_rec_score:
            if rec_dict.__contains__(temp['ltable_name']):
                temp_score_list = list()
                for rec in rec_dict[temp['ltable_name']]:
                    temp_score = sum(pf.jaccard(pf.get_ngram(rec, k, True), pf.get_ngram(temp['rtable_name'], k, True))
                                     for k in range(1, 4)) / 3 + 1 / (pf.edit_dis(rec, temp['rtable_name']) + 1)
                    temp_score_list.append(temp_score)
                sort_score_list = sorted(temp_score_list, reverse=True)
                score = sort_score_list[0]
                temp['rec_score'] = score
        new_test_data.iloc[i] = temp

    return new_train_data, new_test_data


def simple_model(m='rnn', bidirection=False):
    global MODEL_NAME
    if bidirection:
        MODEL_NAME = 'simple_bi_' + m + start_time
    else:
        MODEL_NAME = 'simple_' + m + start_time

    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    # embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1, input_length=MAX_SEQ_LENGTH,
    #                             weights=[embedding_matrix_c],
    #                             mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    if m == 'lstm':
        r = LSTM(units=NN_DIM)
    elif m == 'gru':
        if bidirection:
            r = Bidirectional(GRU(units=NN_DIM, dropout=dropout, recurrent_dropout=dropout), merge_mode='concat')
        else:
            r = GRU(units=NN_DIM)
    else:
        r = SimpleRNN(units=NN_DIM)
    w, s = r(w_e), r(s_e)
    # avg_lambda = Lambda(lambda a: K.mean(a, axis=1))
    # w, s = avg_lambda(w), avg_lambda(s)

    sim_w_s = Subtract()([w, s])
    # sim_w_s = Lambda(lambda a: K.abs(a))(sim_w_s)

    score = Dense(NUM_DENSE, activation='relu')(sim_w_s)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def our_simple_model(folds, test_poi, m='rnn', nother=8, test_num=5, bid=False):
    columns = ['ltable_pinyin', 'rtable_pinyin']
    train_data, test_data = extract_data_simple(process_column=columns, test_poi=test_poi, nother=nother, test_num=test_num, ngram=3)

    test = dict()
    for c in columns:
        tes = np.array(test_data[c])
        tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
        test[c] = tes
    test_label = test_data['label']

    k_fold = StratifiedKFold(n_splits=folds, shuffle=True)
    for fold_num, (train_index, val_index) in enumerate(k_fold.split(train_data, train_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))

        train, val = dict(), dict()
        for c in columns:
            tra = np.array(train_data.iloc[train_index][c])
            tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
            va = np.array(train_data.iloc[val_index][c])
            va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
            train[c] = tra
            val[c] = va
        train_label = train_data.iloc[train_index]['label']
        val_label = train_data.iloc[val_index]['label']

        model = simple_model(m, bid) #simple_model(m, bid)  # 'lstm' 'rnn' 'gru'
        model_checkpoint_path = '{}/colmatch/matching/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
        model.fit([train[c] for c in columns], train_label,
                  batch_size=BATCH_SIZE,
                  epochs=MAX_EPOCHS,
                  verbose=2,
                  # validation_split=0.1,
                  validation_data=([val[c] for c in columns], val_label),
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
        print('training is over.')

        model.load_weights(model_checkpoint_path)
        # test_result = model.evaluate([test[c] for c in columns], test_label, batch_size=BATCH_SIZE, verbose=1)
        # print(test_result)
        # pre_result.append(test_result)
        test_predict = model.predict([test[c] for c in columns], batch_size=BATCH_SIZE, verbose=1)
        t_label = test_label.values
        for index, i in enumerate(test_predict):
            if i > 0.5:
                temp_score = 1
            else:
                temp_score = 0
            with open('{}/colmatch/result/{}_{}_{}.log'.format(ex_path, test_poi, test_num, MODEL_NAME), 'a+', encoding='utf-8') as f:
                f.write('{}/{}\t{}\t{}\t{}\n'.format(t_label[index], temp_score, test_data.iloc[index]['ltable_name'],
                                                     test_data.iloc[index]['rtable_name'], i))
        return

if __name__ == '__main__':
    start_time = time.ctime()
    if sys.platform == 'win32':
        start_time = start_time.replace(':', ' ')

    pois = ['39.92451,116.51533', '39.93483,116.45241', '39.96333,116.45187',
            '39.86184,116.42517', '39.90184,116.41196', '39.94735,116.35581',
            '39.98850,116.41674', '40.00034,116.46960']
    test_poi = '39.88892,116.32670'  # '39.88892,116.32670' 39.96333,116.45187

    our_simple_model(folds=10, test_poi=test_poi, m='gru', nother=8, test_num=10, bid=True)  # 记得更改testdata选法
