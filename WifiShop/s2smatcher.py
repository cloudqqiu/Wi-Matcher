import time
import sys, os
import tensorflow as tf
from keras.layers import *  # Input, SimpleRNN, LSTM, Dense, Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.activations import relu
from keras import backend as K
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
sys.path.append(os.path.abspath('..'))
from WifiShop.utilsLayer import S2SMLayer, SoftmaxAttLayer, AlignSubLayer, KmaxLayer
import WifiShop.data_process as dp
import pro_func as pf

ex_path = '../src/experiment'
MAX_SEQ_LENGTH = 50
NN_DIM = 300  # 300 128
NUM_DENSE = 128
NUM_SEC_DENSE = 32
GRAM_LEN_s, CHAR_LEN_s = 0, 0
GRAM_LEN_c, CHAR_LEN_c = 0, 0
embedding_matrix_s, embedding_matrix_c = None, None
BATCH_SIZE = 128
MAX_EPOCHS = 20
MODEL_NAME = None
start_time = None


def get_abs(tensor):
    return K.abs(tensor)

def get_std(tensor, axis):
    return K.std(tensor, axis=axis)


def get_repeat(tensor, nums):
    return K.repeat(tensor, n=nums)


def get_max(tensor, axis):
    return K.max(tensor, axis=axis)


def get_transpose(tensor):
    return tf.transpose(tensor, [0, 2, 1])


def extract_data_simple(process_column, ngram=3, need_rec_score=False):  # 默认参数必须指向不变对象
    raw_data = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin'] + ['label'] + ['ltable_name', 'rtable_name']]

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
    new_data = raw_data.copy()
    if need_rec_score:
        rec_dict = dp.get_search_recommendation()
        new_data['rec_score'] = 0
    for i in range(len(new_data)):
        temp = new_data.iloc[i].copy()
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
                # if len(sort_score_list) > 3:
                #     score = sum(sort_score_list[0:3]) / 3
                # else:
                #     score = sum(sort_score_list) / len(sort_score_list)
                score = sort_score_list[0]
                temp['rec_score'] = score
        new_data.iloc[i] = temp

    return new_data, raw_data


def s2s_model():
    global MODEL_NAME
    MODEL_NAME = 's2s_model' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=False, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    sim = S2SMLayer(filter_num=100, filter_size=3)([w_e, s_e])
    # sim = AlignSubLayer()([w_e, s_e])
    #




    # weights = dot([w_e, s_e], axes=[2, 2])
    #
    # transpose_layer = Lambda(get_tranpose)
    # abs_layer = Lambda(get_abs)
    # weights_kmax_c = KmaxLayer(20)(weights)
    # weights_kmax_m = KmaxLayer(20)(transpose_layer(weights))
    #
    # att_merge_c = dot([weights_kmax_c, w_e], axes=1)
    # print(K.int_shape(att_merge_c))
    # outputs_m = abs_layer(subtract([s_e, att_merge_c]))
    # print(K.int_shape(outputs_m))
    # att_merge_m = dot([weights_kmax_m, s_e], axes=[2, 1])
    # outputs_c = abs_layer(subtract([w_e, att_merge_m]))
    #
    # std_1_lambda = Lambda(get_std, arguments={'axis': 1})
    # std_2_lambda = Lambda(get_std, arguments={'axis': 2})
    # std_c = std_2_lambda(weights)
    # print(K.int_shape(std_c))
    # std_m = std_1_lambda(weights)
    # repeat_c_lambda = Lambda(get_repeat, arguments={'nums': K.int_shape(w_e)[2]})
    # repeat_m_lambda = Lambda(get_repeat, arguments={'nums': K.int_shape(s_e)[2]})
    # std_c = repeat_c_lambda(std_c)
    # std_c = transpose_layer(std_c)
    # print(K.int_shape(std_c))
    # std_m = repeat_m_lambda(std_m)
    # std_m = transpose_layer(std_m)
    #
    # outputs_m = multiply([std_m, outputs_m])
    # print(K.int_shape(outputs_m))
    # outputs_c = multiply([std_c, outputs_c])
    #
    # cnn = Conv1D(100, kernel_size=3, activation='relu', padding='same')
    # cnn_m = cnn(outputs_m)
    # print(K.int_shape(cnn_m))
    # # maxpooling_m = K.max(cnn_m, axis=1)
    #
    # max_1_lambda = Lambda(get_max, arguments={'axis': 1})
    # maxpooling_m = max_1_lambda(cnn_m)
    # print(K.int_shape(maxpooling_m))
    #
    # cnn_c = cnn(outputs_c)
    # # maxpooling_c = K.max(cnn_c, axis=1)
    # maxpooling_c = max_1_lambda(cnn_c)
    #
    # sim = concatenate([maxpooling_c, maxpooling_m])

    print(K.int_shape(sim))

    score = Dense(NUM_DENSE, activation='relu')(sim)
    score = Dense(1, activation='sigmoid')(score)
    print(K.int_shape(score))

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model

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

    r = LSTM(units=NN_DIM)
    w, s = r(w_e), r(s_e)
    sim_w_s = Subtract()([w, s])

    score = Dense(NUM_DENSE, activation='relu')(sim_w_s)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def s2sm_cikm19(folds=5):
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data, raw_data = extract_data_simple(process_column=columns, ngram=3)

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

            model = s2s_model() #simple_model()#
            model_checkpoint_path = '{}/linking/matching/s2s/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
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
            print(tp / (tp + fp), tp / (tp + fn))
            pre_result.append([tp, fp, fn])

            # return
            K.clear_session()
            del train, test, train_label, test_label
            del model
            gc.collect()
            # break
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
    if sys.platform == 'win32':
        start_time = start_time.replace(':', ' ')
    _p, _r, times = 0, 0, 10
    for i in range(times):
        temp_p, temp_r = s2sm_cikm19(folds=5)
        _p += temp_p
        _r += temp_r
        # break
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))