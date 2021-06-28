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
sys.path.append(os.path.abspath('..'))


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


def extract_data_simple(process_column, ngram=3):  # 默认参数必须指向不变对象
    raw_data = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))  # 整个训练数据文件
    raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin'] + ['label'] + ['ltable_name', 'rtable_name']]

    for i in range(len(raw_data)):  # 转小写
        temp = raw_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        raw_data.iloc[i] = temp

    gramset, charset = set(), set()  # 获取ngram和字符集合
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
    embedding_matrix_s = np.zeros((GRAM_LEN_s + 1, CHAR_LEN_s), dtype=int)  # 矩阵多1维 为神经网络mask操作保留一个默认零

    gram2index = {gram: index + 1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}

    for index in index2gram:
        for char in index2gram[index]:
            embedding_matrix_s[index, char2index[char]] += 1  # 生成embedding矩阵 类one-hot
    new_data = raw_data.copy()
    for i in range(len(new_data)):  # ngram换成embedding矩阵中对应的索引号
        temp = new_data.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index.get(temp[column])]
            else:
                temp[column] = [gram2index.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        new_data.iloc[i] = temp

    return new_data, raw_data


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

    sim_w_s = Subtract()([w, s])
    # sim_w_s = Lambda(lambda a: K.abs(a))(sim_w_s)

    score = Dense(NUM_DENSE, activation='relu')(sim_w_s)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def our_simple_model(folds, m='rnn', bid=False, save_log=False):
    columns = ['ltable_pinyin', 'rtable_pinyin']  # 这个列表表示需要处理的数据的属性列，我这里是比较两个实体的pinyin，当做参数输入extract_data_simple函数。
    new_data, raw_datam = extract_data_simple(process_column=columns, ngram=3)
    pre_result = list()

    k_fold = StratifiedKFold(n_splits=folds, shuffle=True)
    for fold_num, (train_index, test_index) in enumerate(k_fold.split(new_data, new_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))
        new_data_train = new_data.iloc[train_index]

        val_folder = StratifiedKFold(n_splits=10, shuffle=True)
        for t_index, val_index in val_folder.split(new_data_train, new_data_train['label']):
            # print(t_index, val_index)
            train, test, val = dict(), dict(), dict()
            for c in columns:  # 为每个属性列生成一个序列化的dict
                tra = np.array(new_data_train.iloc[t_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
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

            model = simple_model(m, bid)
            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
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
                        if save_log:
                            with open('{}/linking/log/FP-{}.log'.format(ex_path, MODEL_NAME), 'a+', encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ltable_name'],
                                                                  new_data.iloc[test_index].iloc[index]['rtable_name']))
                else:
                    if t_label[index] == 1:
                        fn += 1
                        if save_log:
                            with open('{}/linking/log/FN-{}.log'.format(ex_path, MODEL_NAME), 'a+', encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ltable_name'],
                                                                  new_data.iloc[test_index].iloc[index]['rtable_name']))
            print(tp, fp, fn)
            print(tp / (tp + fp), tp / (tp + fn))
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
    if sys.platform == 'win32':
        start_time = start_time.replace(':', ' ')
    _p, _r, times = 0, 0, 10
    for i in range(times):
        temp_p, temp_r = our_simple_model(folds=5, m='gru', bid=True, save_log=True)
        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))