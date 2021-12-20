# 复现sigmod18 deepmatcher 的方法
import sys
import os
sys.path.append(os.path.abspath('..'))
import tensorflow as tf
import keras
from keras.layers import *  # Input, SimpleRNN, LSTM, Dense, Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import gc
import pandas as pd
import numpy as np
import math
import time
from sklearn.model_selection import StratifiedKFold


ex_path = '../../data/zh-ssid-venue/'
se_path = '../src/search engine'
SOURCES = ['baidu']
MAX_SEQ_LENGTH = 50
MAX_S_SEQ_LENGTH = 256
MIN_S_SEQ_LENGTH = 10
MAX_SR_NUM = 3
NN_DIM = 300
NUM_DENSE = 128
GRAM_LEN = 0
CHAR_LEN = 0
embedding_matrix = None
BATCH_SIZE = 128
MAX_EPOCHS = 20
MODEL_NAME = None


class HighwayLayer(Layer):
    def __init__(self, **kwargs):
        super(HighwayLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.supports_masking = True
        self.W = self.add_weight(name='weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        # print(K.int_shape(self.W))
        self.b = self.add_weight(name='bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        # print(K.int_shape(self.b))
        super(HighwayLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # print(K.int_shape(x))
        # x.shape = (batch_size, seq_len, time_steps)
        # self.b = K.expand_dims(self.b, axis=-1)
        linear_x = K.dot(x, self.W) + self.b
        # print('linearx', K.int_shape(linear_x))
        T_x = K.sigmoid(linear_x)
        relu_x = K.relu(linear_x)
        outputs = T_x * relu_x + (1.0 - T_x) * x
        outputs = K.permute_dimensions(outputs, (0, 2, 1))
        # print(K.int_shape(outputs))
        return outputs


def extract_data_simple(process_column, ngram=3):  # 默认参数必须指向不变对象
    raw_data = pd.read_csv('{}/match_use.csv'.format(ex_path))
    raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin'] + ['label']]

    for i in range(len(raw_data)):  # 转小写
        temp = raw_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        raw_data.iloc[i] = temp

    gramset, charset = set(), set()  # 获取ngram
    for index, row in raw_data.iterrows():
        for column in process_column:
            charset = charset | set(row[column])
            if len(row[column]) < ngram:
                grams = [row[column]]
            else:
                grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
            gramset = gramset | set(grams)
    # for i in gramset:
    #     charset = charset | set(i)
    print('extract {}-gram {}'.format(ngram, len(gramset)))
    print('extract character {}'.format(len(charset)))
    global GRAM_LEN, CHAR_LEN, embedding_matrix
    GRAM_LEN, CHAR_LEN = len(gramset), len(charset)
    embedding_matrix = np.zeros((GRAM_LEN + 1, CHAR_LEN), dtype=int)
    # embedding_matrix = np.delete(embedding_matrix, [0], axis=1)

    gram2index = {gram: index + 1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}
    # index2char = {char2index[char]: char for char in char2index}

    for index in index2gram:
        for char in index2gram[index]:
            embedding_matrix[index, char2index[char]] += 1
    new_data = raw_data.copy()
    for i in range(len(new_data)):
        temp = new_data.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index.get(temp[column])]
            else:
                temp[column] = [gram2index.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        new_data.iloc[i] = temp

    return new_data, raw_data


def read_search_result(sources):
    search_result = dict()
    for source in sources:
        search_docs = dict()
        with open('{}/data_{}/wifi_search_result.txt'.format(se_path, source), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wifi, docs = line.strip().split('\t')
                docs = eval(docs)
                docs = docs[:MAX_SR_NUM]
                search_docs[wifi] = docs
        search_result[source] = search_docs
    return search_result


def extract_data_complex(process_column, sources, ngram=3):
    raw_data = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin'] + ['label'] + ['ltable_name']]
    search_result = read_search_result(sources)  # lowercase done
    for i in range(len(raw_data)):
        temp = raw_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        raw_data.iloc[i] = temp

    ws_data = raw_data[['ltable_pinyin', 'rtable_pinyin']].copy()
    for i in range(len(ws_data)):
        ws_data.iloc[i]['ltable_pinyin'] = ws_data.iloc[i]['ltable_pinyin'] + ws_data.iloc[i]['rtable_pinyin']
    ws_data = ws_data.drop(columns=['rtable_pinyin'])

    gramset, charset = set(), set()
    for index, row in raw_data.iterrows():
        for column in process_column:
            charset = charset | set(row[column])
            if len(row[column]) < ngram:
                grams = [row[column]]
            else:
                grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
            gramset = gramset | set(grams)
    for index, row in ws_data.iterrows():
        charset = charset | set(row['ltable_pinyin'])
        if len(row['ltable_pinyin']) < ngram:
            grams = [row['ltable_pinyin']]
        else:
            grams = [row['ltable_pinyin'][i:i+ngram] for i in range(len(row['ltable_pinyin']) - ngram + 1)]
        gramset = gramset | set(grams)
    for source in sources:
        for key in search_result[source].keys():
            for c in search_result[source][key]:
                charset = charset | set(c)
                if len(c) < ngram:
                    grams = [c]
                else:
                    grams = [c[i:i + ngram] for i in range(len(c) - ngram + 1)]
                gramset = gramset | set(grams)
    # for i in gramset:
    #     charset = charset | set(i)

    print('extract {}-gram {}'.format(ngram, len(gramset)))
    print('extract character {}'.format(len(charset)))
    global GRAM_LEN, CHAR_LEN, embedding_matrix
    GRAM_LEN, CHAR_LEN = len(gramset), len(charset)
    embedding_matrix = np.zeros((GRAM_LEN + 1, CHAR_LEN), dtype=int)
    # embedding_matrix = np.delete(embedding_matrix, [0], axis=1)

    gram2index = {gram: index+1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}
    # index2char = {char2index[char]: char for char in char2index}

    for index in index2gram:
        for char in index2gram[index]:
            embedding_matrix[index, char2index[char]] += 1

    new_data = raw_data.copy()
    for i in range(len(new_data)):
        temp = new_data.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index.get(temp[column])]
            else:
                temp[column] = [gram2index.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        new_data.iloc[i] = temp
    for i in range(len(ws_data)):
        temp = ws_data.iloc[i].copy()
        if len(temp['ltable_pinyin']) < ngram:
            temp['ltable_pinyin'] = [gram2index.get(temp['ltable_pinyin'])]
        else:
            temp['ltable_pinyin'] = [gram2index.get(temp['ltable_pinyin'][j:j+ngram]) for j in range(len(temp['ltable_pinyin']) - ngram + 1)]
        ws_data.iloc[i] = temp
    max_source_length = 0
    for source in sources:
        for key in search_result[source].keys():
            this = search_result[source][key]
            for index, c in enumerate(this):
                if len(c) < ngram:
                    this[index] = [gram2index.get(c)]
                else:
                    this[index] = [gram2index.get(c[j:j + ngram]) for j in range(len(c) - ngram + 1)]
                    if len(this[index]) > max_source_length:
                        max_source_length = len(this[index])
            search_result[source][key] = this
    global MAX_S_SEQ_LENGTH
    if max_source_length < MAX_S_SEQ_LENGTH:
        MAX_S_SEQ_LENGTH = max_source_length
        print('CHANGE MAX_S_SEQ_LENGTH =', MAX_S_SEQ_LENGTH)
    return new_data, ws_data, search_result


def get_transpose(tensor):
    return K.permute_dimensions(tensor, (0, 2, 1))


def get_softmax(tensor):
    return K.softmax(tensor, axis=1)


def get_softalign_m(tensor):
    print(K.int_shape(tensor[1]))
    k_transpose = K.permute_dimensions(tensor[1], (0, 2, 1))
    # k_transpose = tensor[1]
    print(K.int_shape(k_transpose))
    softalign_m = K.dot(tensor[0], k_transpose)
    softalign_m = K.softmax(softalign_m, axis=1)
    return softalign_m


def get_average_k(tensor):
    # k_transpose = K.permute_dimensions(tensor[1], (0, 2, 1))
    ave_k = K.batch_dot(tensor[0], tensor[1])
    return ave_k


def get_div_dim_seq(tensor):
    assert K.ndim(tensor) == 2
    temp = np.array([1 / math.sqrt(MAX_SEQ_LENGTH) for _ in range(K.int_shape(tensor)[1])])
    return tensor * temp


def att_model():
    global MODEL_NAME
    MODEL_NAME = 'deepmatcher_ATT' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')

    embedding_layer = Embedding(output_dim=CHAR_LEN, input_dim=GRAM_LEN + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix],
                                mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    # align_h1, align_h2 = Highway(activation='relu', init='glorot_uniform'), Highway(activation='relu', init='glorot_uniform')
    # td = TimeDistributed(align_h1)
    # w_a, s_a = td(w_e), td(s_e)
    # def single_highway(tensor):
    #     return K.map_fn(lambda x: align_h1(x), elems=tensor[0])
    # w_a, s_a = Lambda(lambda a: K.squeeze(a, axis=0))(w_e), Lambda(lambda a: K.squeeze(a, axis=0))(s_e)
    # w_a, s_a = align_h1(w_a), align_h1(s_a)
    # print(K.int_shape(w_a), K.int_shape(s_a))
    # highway, h2 = HighwayLayer(), HighwayLayer()
    # d1, d2 = Dense(units=CHAR_LEN, activation='relu'), Dense(units=CHAR_LEN, activation='relu')
    # w_a, s_a = d1(d2(w_e)), d1(d2(s_e))
    h1, h2 = HighwayLayer(), HighwayLayer()
    w_a, s_a = h1(h2(w_e)), h1(h2(s_e))
    # w_a, s_a = w_e, s_e
    # print(K.int_shape(w_a), K.int_shape(s_a))

    # w_a, s_a = w_e, s_e
    # d3, d4 = Dense(units=CHAR_LEN, activation='relu'), Dense(units=CHAR_LEN, activation='tanh')
    h3, h4 = HighwayLayer(), HighwayLayer()
    # D = Dropout(0.5)  # 加dropout会变差
    softalign_w_s = dot(inputs=[w_a, Lambda(get_transpose)(s_a)], axes=(2, 1))
    softalign_w_s = Lambda(get_softmax)(softalign_w_s)
    # print(K.int_shape(softalign_w_s))
    s_a_avg = dot(inputs=[softalign_w_s, s_a], axes=1)
    # print('s_a_avg', K.int_shape(s_a_avg))
    w_comparison = concatenate(inputs=[w_a, s_a_avg], axis=-1)
    # print(K.int_shape(w_comparison))
    w_comparison = h3(h4(w_comparison))
    w_aggregation = Lambda(lambda a: K.mean(a, axis=1))(w_comparison)
    # print(K.int_shape(w_aggregation))
    w_aggregation = Lambda(get_div_dim_seq)(w_aggregation)

    softalign_s_w = dot(inputs=[s_a, Lambda(get_transpose)(w_a)], axes=(2, 1))
    softalign_s_w = Lambda(get_softmax)(softalign_s_w)
    w_a_avg = dot(inputs=[softalign_s_w, w_a], axes=1)
    s_comparison = concatenate(inputs=[s_a, w_a_avg], axis=-1)
    s_comparison = h3(h4(s_comparison))
    s_aggregation = Lambda(lambda a: K.mean(a, axis=1))(s_comparison)  # seq长度相同 不均一化了
    s_aggregation = Lambda(get_div_dim_seq)(s_aggregation)

    sim_w_s = subtract(inputs=[w_aggregation, s_aggregation])
    # print(K.int_shape(sim_w_s))

    score = Dense(NUM_DENSE, activation='relu')(sim_w_s)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def hy_model():
    global MODEL_NAME
    MODEL_NAME = 'deepmatcher_HY' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')

    embedding_layer = Embedding(output_dim=CHAR_LEN, input_dim=GRAM_LEN + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix],
                                mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)
    rnn1, rnn2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat'), \
                 Bidirectional(GRU(units=NN_DIM), merge_mode='concat')

    # d1, d2 = Dense(units=CHAR_LEN, activation='relu'), Dense(units=CHAR_LEN, activation='tanh')
    # w_a, s_a = d1(d2(w_e)), d1(d2(s_e))

    h1, h2 = HighwayLayer(), HighwayLayer()
    w_a, s_a = h1(h2(w_e)), h1(h2(s_e))

    # w_a, s_a = w_e, s_e
    # print(K.int_shape(w_a), K.int_shape(s_a))

    w_rnn1, s_rnn1 = rnn1(w_e), rnn1(s_e)
    w_rnn2, s_rnn2 = rnn2(w_e), rnn2(s_e)
    d3, d4 = Dense(units=NN_DIM * 2, activation='relu'), Dense(units=NN_DIM * 2, activation='tanh')
    d5, d6 = Dense(units=1, activation='relu'), Dense(units=NN_DIM * 2, activation='tanh')
    h3, h4, h5, h6 = HighwayLayer(), HighwayLayer(), HighwayLayer(), HighwayLayer()
    # D1, D2 = Dropout(0.3), Dropout(0.3)

    softalign_w_s = dot(inputs=[w_a, Lambda(get_transpose)(s_a)], axes=(2, 1))
    softalign_w_s = Lambda(get_softmax)(softalign_w_s)
    # print(K.int_shape(softalign_w_s))
    s_a_avg = dot(inputs=[softalign_w_s, s_rnn1], axes=1)
    # print('s_a_avg', K.int_shape(s_a_avg))
    w_comparison = concatenate(inputs=[w_rnn1, s_a_avg], axis=-1)
    # print(K.int_shape(w_comparison))
    # w_comparison = d3(d4(w_comparison))
    w_comparison = h3(h4(w_comparison))
    s_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(s_rnn2)
    # print(K.int_shape(s_rnn2_rp))
    w_comparison_weight = concatenate(inputs=[w_comparison, s_rnn2_rp], axis=-1)
    w_comparison_weight = d5(h5(w_comparison_weight))
    # print(K.int_shape(w_comparison_weight))
    w_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(w_comparison_weight)
    w_comparison_weight = Lambda(get_softmax)(w_comparison_weight)
    # print(K.int_shape(w_comparison_weight))
    w_aggregation = dot(inputs=[w_comparison_weight, w_comparison], axes=1)
    # print(K.int_shape(w_aggregation))

    softalign_s_w = dot(inputs=[s_a, Lambda(get_transpose)(w_a)], axes=(2, 1))
    softalign_s_w = Lambda(get_softmax)(softalign_s_w)
    w_a_avg = dot(inputs=[softalign_s_w, w_rnn1], axes=1)
    s_comparison = concatenate(inputs=[s_rnn1, w_a_avg], axis=-1)
    # s_comparison = d3(d4(s_comparison))
    s_comparison = h3(h4(s_comparison))
    w_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(w_rnn2)
    s_comparison_weight = concatenate(inputs=[s_comparison, w_rnn2_rp], axis=-1)
    s_comparison_weight = d5(h5(s_comparison_weight))
    s_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(s_comparison_weight)
    s_comparison_weight = Lambda(get_softmax)(s_comparison_weight)
    s_aggregation = dot(inputs=[s_comparison_weight, s_comparison], axes=1)

    sim_w_s = subtract(inputs=[w_aggregation, s_aggregation])
    # print(K.int_shape(sim_w_s))
    sim_w_s = Lambda(lambda a: K.abs(a))(sim_w_s)

    score = Dense(NUM_DENSE, activation='relu')(sim_w_s)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def dm_model(folds=5):
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data, raw_data = extract_data_simple(process_column=columns, ngram=3)  # 使用w-s词汇表 或者包括se的词汇表
    # new_data, ws_data, search_result = extract_data_complex(process_column=columns,
    #                                                                                 sources=SOURCES, ngram=3)
    pre_result = list()

    k_fold = StratifiedKFold(n_splits=folds, shuffle=True)
    for fold_num, (train_index, test_index) in enumerate(k_fold.split(new_data, new_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))
        new_data_train = new_data.iloc[train_index]

        val_folder = StratifiedKFold(n_splits=10, shuffle=True)
        for t_index, val_index in val_folder.split(new_data_train, new_data_train['label']):
            # print(t_index, val_index)
            train, test, val = dict(), dict(), dict()
            for c in columns:
                # tra = np.array(new_data.iloc[train_index][c])
                tra = np.array(new_data_train.iloc[t_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
                tes = np.array(new_data.iloc[test_index][c])
                # tes = np.array(new_data.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
                va = np.array(new_data_train.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va
            # train_label = new_data.iloc[train_index]['label']
            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']

            model = hy_model()  # att hy
            model_checkpoint_path = '{}/linking/matching/deepmatcher/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
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
            model.load_weights(model_checkpoint_path)
            # test_result = model.evaluate([test[c] for c in columns], test_label, batch_size=BATCH_SIZE, verbose=1)
            # print(test_result)
            # pre_result.append(test_result)
            test_predict = model.predict([test[c] for c in columns], batch_size=BATCH_SIZE, verbose=1)
            t_label = test_label.values
            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if t_label[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if t_label[index] == 1:
                        fn += 1
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
    start_time = time.ctime()

    p, r, times = 0, 0, 10
    for i in range(times):
        temp_p, temp_r = dm_model(folds=5)
        p += temp_p
        r += temp_r
    p /= times
    r /= times
    f = 2 * p * r / (p + r)
    print('P: {}\tR: {}\tF: {}\n'.format(p, r, f))

    # dm_model(folds=5)
