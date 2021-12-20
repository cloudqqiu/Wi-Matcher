# WWW '19 method HI_ET
import argparse
import itertools
import sys
import os
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
import time
from sklearn.model_selection import StratifiedKFold
sys.path.append(os.getcwd()+"/../..")

import models.common.utils as utils
import models.model_config as config
from timeit import default_timer as timer
sys.path.append(os.path.abspath('../..'))



MAX_SEQ_LENGTH = 50
NN_DIM = 300
NUM_DENSE = 128
GRAM_LEN = 0
CHAR_LEN = 0
BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 128
MAX_EPOCHS = 15
MODEL_NAME = None

EMBED_DIM = 300


class WWW19Layer(Layer):
    def __init__(self, main_tensor, **kwargs):
        self.main_tensor = main_tensor
        # self.softlambda = Lambda(get_softmax_vertical)
        # self.meanlambda = Lambda(get_mean)
        super(WWW19Layer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        self.W = self.add_weight(name='bilinear_weight',
                                 shape=(K.int_shape(self.main_tensor)[2], input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)
        self.w = self.add_weight(name='self_weight',
                                 shape=(K.int_shape(self.main_tensor)[2] * 2, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(WWW19Layer, self).build(input_shape)

    def call(self, inputs):
        # print(K.int_shape(self.main_tensor), K.int_shape(self.W))
        weights = K.dot(self.main_tensor, self.W)
        # print(K.int_shape(weights))
        weights = dot([weights, inputs], axes=2)
        # print(K.int_shape(weights))
        A = dot([weights, inputs], axes=[2, 1])
        # print(K.int_shape(A))
        # print('input', K.int_shape(inputs))
        s = subtract([self.main_tensor, A])
        m = multiply([self.main_tensor, A])
        outputs = concatenate([s, m], axis=-1)
        # print(K.int_shape(outputs))
        # outputs = self.meanlambda(outputs)
        # print(K.int_shape(self.W))
        beta = K.dot(outputs, self.w)
        # print(K.int_shape(beta))
        beta = K.squeeze(beta, axis=-1)
        # print(K.int_shape(beta))
        outputs = dot([beta, outputs], axes=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0],  K.int_shape(self.main_tensor)[2] * 2


class HIEMLayer(Layer):
    def __init__(self, **kwargs):
        self.rsoftlambda = Lambda(utils.get_softmax_row)
        self.softlambda = Lambda(utils.get_softmax)
        super(HIEMLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        self.W = self.add_weight(name='bilinear_weight',
                                 shape=(input_shape[0][2], input_shape[0][2]),
                                 initializer='uniform',
                                 trainable=True)
        self.w = self.add_weight(name='self_weight',
                                 shape=(input_shape[0][2] * 2, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(HIEMLayer, self).build(input_shape)

    def call(self, inputs):
        i, j = inputs
        # print(K.int_shape(i), K.int_shape(j))
        alpha_i = K.dot(i, self.W)
        # print(K.int_shape(alpha))
        alpha_i = dot([alpha_i, j], axes=2)
        # print(K.int_shape(alpha))
        # alpha_i = dot([i, j], axes=2)
        alpha_i = self.rsoftlambda(alpha_i)
        a_i = dot([alpha_i, j], axes=[2, 1])
        # print(K.int_shape(A))
        s_i = subtract([i, a_i])
        m_i = multiply([i, a_i])
        output_i = concatenate([s_i, m_i], axis=-1)
        print(K.int_shape(output_i))
        # outputs = self.meanlambda(outputs)
        # print(K.int_shape(self.W))
        beta_i = K.dot(output_i, self.w)
        print(K.int_shape(beta_i))
        beta_i = K.squeeze(beta_i, axis=-1)
        print(K.int_shape(beta_i))
        beta_i = self.softlambda(beta_i)
        output_i = dot([beta_i, output_i], axes=1)

        alpha_j = K.dot(j, self.W)
        # print(K.int_shape(alpha))
        alpha_j = dot([alpha_j, i], axes=2)
        # print(K.int_shape(alpha))
        # alpha_j = dot([j, i], axes=2)
        alpha_j = self.rsoftlambda(alpha_j)
        a_j = dot([alpha_j, i], axes=[2, 1])
        # print(K.int_shape(A))
        s_j = subtract([j, a_j])
        m_j = multiply([j, a_j])
        output_j = concatenate([s_j, m_j], axis=-1)
        print(K.int_shape(output_j))
        # outputs = self.meanlambda(outputs)
        # print(K.int_shape(self.W))
        beta_j = K.dot(output_j, self.w)
        print(K.int_shape(beta_j))
        beta_j = K.squeeze(beta_j, axis=-1)
        print(K.int_shape(beta_j))
        beta_j = self.softlambda(beta_j)
        output_j = dot([beta_j, output_j], axes=1)
        return [output_i, output_j]
        # return output_i

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][2] * 2), (input_shape[1][0], input_shape[1][2] * 2)]
        # return (input_shape[0][0], input_shape[0][2] * 2)


class InterAttLayer(Layer):
    def __init__(self, **kwargs):
        self.rsoftlambda = Lambda(utils.get_softmax_row)
        super(InterAttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        self.W = self.add_weight(name='bilinear_weight',
                                 shape=(input_shape[0][2], input_shape[0][2]),
                                 initializer='uniform',
                                 trainable=True)
        super(InterAttLayer, self).build(input_shape)

    def call(self, inputs):
        i, j = inputs
        # print(K.int_shape(i), K.int_shape(j))
        alpha_i = K.dot(i, self.W)
        # print(K.int_shape(alpha))
        alpha_i = dot([alpha_i, j], axes=2)
        # print(K.int_shape(alpha))
        # alpha_i = dot([i, j], axes=2)
        alpha_i = self.rsoftlambda(alpha_i)
        a_i = dot([alpha_i, j], axes=[2, 1])
        # print(K.int_shape(a_i))

        alpha_j = K.dot(j, self.W)
        # print(K.int_shape(alpha))
        alpha_j = dot([alpha_j, i], axes=2)
        # print(K.int_shape(alpha))
        # alpha_j = dot([j, i], axes=2)
        alpha_j = self.rsoftlambda(alpha_j)
        a_j = dot([alpha_j, i], axes=[2, 1])
        # print(K.int_shape(a_j))

        return [a_i, a_j]

    def compute_output_shape(self, input_shape):
        return input_shape


class IntraAttLayer(Layer):
    def __init__(self, **kwargs):
        self.softlambda = Lambda(utils.get_softmax)
        self.timedisdense = TimeDistributed(Dense(1))
        super(IntraAttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 # shape=(input_shape[2], 1),
                                 shape=(1, input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)
        super(IntraAttLayer, self).build(input_shape)

    def call(self, inputs):
        # weight = K.dot(inputs, self.W)
        # weight = K.squeeze(weight, axis=-1)
        # weight = self.softlambda(weight)
        # outputs = dot([weight, inputs], axes=1)

        # a = K.permute_dimensions(inputs, (0, 2, 1))
        # weight = K.dot(self.W, a)
        # print(K.int_shape(weight))
        # weight = K.squeeze(weight, axis=0)
        # print(K.int_shape(weight))
        # outputs = dot([weight, inputs], axes=1)

        weight = self.timedisdense(inputs)
        print(K.int_shape(weight))
        weight = K.squeeze(weight, axis=-1)
        print(K.int_shape(weight))
        outputs = dot([weight, inputs], axes=1)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]




def extract_data_simple(data, process_column, ngram=3):  # 默认参数必须指向不变对象
    raw_data = utils.load_data(data)

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

    return new_data


def extract_data_simple_char(data, process_column):
    raw_data = utils.load_data(data)

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

    # charset = set()
    # for index, row in raw_data.iterrows():
    #     for column in process_column:
    #         charset = charset | set(row[column])

    all_str_list = list(itertools.chain.from_iterable(raw_data[process_column].values))
    charset = set(char for charlist in all_str_list for char in list(charlist))

    print('extract character {}'.format(len(charset)))
    global CHAR_LEN
    CHAR_LEN = len(charset)
    char2index = {char: index + 1 for index, char in enumerate(charset)}

    def encode(row):
        for col in process_column:
            row[col] = [char2index.get(row[col][j]) for j in range(len(row[col]))]
        return row

    # new_data = raw_data.copy()
    # for i in range(len(new_data)):
    #     temp = new_data.iloc[i].copy()
    #     for column in process_column:
    #         temp[column] = [char2index.get(temp[column][j]) for j in range(len(temp[column]))]
    #     new_data.iloc[i] = temp
    raw_data = raw_data.apply(lambda row: encode(row), axis=1)

    return raw_data


def hi_em_model(char=False):
    global MODEL_NAME
    MODEL_NAME = 'www19 HI_EM' + time_str
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')

    if char:
        embedding_layer = Embedding(output_dim=300, input_dim=CHAR_LEN + 1, input_length=MAX_SEQ_LENGTH,  # output_dim=CHAR_LEN
                                    embeddings_initializer='uniform',
                                    mask_zero=True, trainable=True)
    else:
        embedding_layer = Embedding(output_dim=CHAR_LEN, input_dim=GRAM_LEN + 1, input_length=MAX_SEQ_LENGTH,
                                    embeddings_initializer='uniform',
                                    mask_zero=True, trainable=True)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')  # 这两个r层 dropout=0.4, recurrent_dropout=0.4
    r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # w, s = r(w_e), r(w_e)
    w, s = r2(r(w_e)), r2(r(s_e))
    # w = Lambda(lambda a: a[:, -1, :])(w)
    # s = Lambda(lambda a: a[:, -1, :])(s)

    # h = HIETLayer()
    # w_h, s_h = h([w, s])
    # w_h = h([w, s])
    # print(K.int_shape(w_h))
    # s_h = h([s, w])

    inter = InterAttLayer()
    w_inter, s_inter = inter([w, s])
    sub_i = subtract([w, w_inter])
    mul_i = multiply([w, w_inter])
    w_inter = concatenate([sub_i, mul_i], axis=-1)
    sub_j = subtract([s, s_inter])
    mul_j = multiply([s, s_inter])
    s_inter = concatenate([sub_j, mul_j], axis=-1)
    # print(K.int_shape(s_inter))

    # intra = IntraAttLayer()
    # w_intra, s_intra = intra(w_inter), intra(s_inter)
    # print(K.int_shape(s_intra))
    # m = Lambda(get_mean)
    # w_intra, s_intra = m(w_inter), m(s_inter)

    intra_attention = TimeDistributed(Dense(1))
    squ, soft = Lambda(utils.squeeze), Lambda(utils.get_softmax)
    beta_w = intra_attention(w_inter)
    beta_s = intra_attention(s_inter)
    beta_w, beta_s = soft(beta_w), soft(beta_s)
    print(K.int_shape(beta_w))
    w_intra = dot([beta_w, w_inter], axes=1)
    s_intra = dot([beta_s, s_inter], axes=1)
    # print(K.int_shape(w_intra))
    w_intra, s_intra = squ(w_intra), squ(s_intra)
    # print(K.int_shape(w_intra))

    # www = WWW19Layer(main_tensor=w)
    # w_h = www(s)
    # wws = WWW19Layer(main_tensor=s)
    # s_h = wws(w)

    sim_w_s = concatenate([w_intra, s_intra])
    print(K.int_shape(sim_w_s))

    # sim_w_s = w_h
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_w_s)  # NUM_DENSE * 2
    # score = Dense(NN_DIM, activation='relu')(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def hiem_model(data, folds=5, char=False):
    timer_log = utils.Timer()
    timer_log.start()
    columns = ['ssid', 'venue']

    start_time_data_preprocess = timer()

    if char:
        new_data = extract_data_simple_char(data, process_column=columns)  # 使用w-s词汇表
    else:
        new_data = extract_data_simple(data, process_column=columns, ngram=3)  # 使用w-s词汇表
    pre_result = list()

    end_time_data_preprocess = timer()
    print(f"# Timer - Data Preprocess 1 - {end_time_data_preprocess - start_time_data_preprocess}")

    timer_log.log("Data Load")

    k_fold = StratifiedKFold(n_splits=folds, shuffle=True)
    for fold_num, (train_index, test_index) in enumerate(k_fold.split(new_data, new_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))

        start_time_data_preprocess = timer()

        new_data_train = new_data.iloc[train_index]

        val_folder = StratifiedKFold(n_splits=10, shuffle=True)
        for t_index, val_index in val_folder.split(new_data_train, new_data_train['label']):
            # print(t_index, val_index)
            train, test, val = dict(), dict(), dict()
            for c in columns:
                # tra = np.array(new_data.iloc[train_index][c])
                tra = np.array(new_data_train.iloc[t_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
                # tes = np.array(new_data.iloc[test_index][c])
                tes = np.array(new_data.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
                va = np.array(new_data_train.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va
            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']

            end_time_data_preprocess = timer()
            print(f"# Timer - Data Preprocess 2 - {end_time_data_preprocess - start_time_data_preprocess}")

            timer_log.log(f'Fold {fold_num + 1} Data Preprocess')

            model = hi_em_model(char=char)
            if data=='zh':
                model_checkpoint_path = f'{config.zh_hiem_data_path}/{data}_model_checkpoint_{MODEL_NAME}.h5'
            elif data=='ru':
                model_checkpoint_path = f'{config.ru_hiem_data_path}/{data}_model_checkpoint_{MODEL_NAME}.h5'

            timer_log.log(f'Fold {fold_num + 1} Model Initialization')

            start_time_model_train = timer()

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

            end_time_model_train = timer()
            print(f"# Timer - Model Train - {end_time_model_train - start_time_model_train}")

            timer_log.log(f'Fold {fold_num + 1} Model Train')

            model.load_weights(model_checkpoint_path)
            # test_result = model.evaluate([test[c] for c in columns], test_label, batch_size=BATCH_SIZE, verbose=1)
            # print(test_result)
            # pre_result.append(test_result)

            start_time_model_predict = timer()

            test_predict = model.predict([test[c] for c in columns], batch_size=BATCH_SIZE, verbose=1)

            end_time_model_predict = timer()
            print(f"# Timer - Model Predict - Samples: {len(test)} Batch Size: {BATCH_SIZE} - {end_time_model_predict - start_time_model_predict}")

            timer_log.log(f'Fold {fold_num + 1} Prediction')

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

    timer_log.stop(f'Hi-EM {folds} Folds')
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
    parser.add_argument('--repeat_run', type=int, default=5)

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    start_time = time.ctime()
    # if sys.platform == 'win32':
    #     start_time = start_time.replace(':', ' ')
    time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    times = args.repeat_run
    _p, _r = 0, 0
    for i in range(times):
        temp_p, temp_r = hiem_model(data=args.dataset, folds=5, char=True)
        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    f = 2 * _p * _r / (_p + _r)
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, f))