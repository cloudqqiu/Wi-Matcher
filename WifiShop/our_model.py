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


def softmax10avg(tensors):
    # print(len(tensors))
    # ts = K.concatenate(tensors[1:], axis=-1)
    # print(K.int_shape(ts))
    # ts = K.reshape(ts, (MAX_SR_NUM, NN_DIM))
    ts = K.stack(tensors[1:], axis=1)
    # print(K.int_shape(ts))
    # ts_transpose = K.reshape(ts, (NN_DIM, MAX_SR_NUM))
    ts_transpose = K.permute_dimensions(ts, (0, 2, 1))
    # print(K.int_shape(ts_transpose))
    weights = K.dot(tensors[0], ts_transpose)
    # print(K.int_shape(weights))
    weights = K.squeeze(weights, axis=1)
    # print(K.int_shape(weights))
    sfm_weights = K.softmax(weights)
    print(K.int_shape(sfm_weights))
    # r = K.dot(sfm_weights, ts)
    # print(K.int_shape(r))
    # r = K.squeeze(r, axis=1)
    # print(K.int_shape(r))
    return sfm_weights


def get_stack(tensors):
    return K.stack(tensors, axis=1)


def get_stack0(tensors):
    return K.stack(tensors, axis=0)


def get_transpose(tensor):
    return K.permute_dimensions(tensor, (0, 2, 1))


def get_softmax(tensor):
    return K.softmax(tensor)


def get_softmax_row(tensor):
    return K.softmax(tensor, axis=1)


def get_div_dim(tensor):
    assert K.shape(tensor) == 2
    temp = np.array([math.sqrt(K.int_shape(tensor)[1]) for _ in range(K.int_shape(tensor)[1])])
    return tensor * temp


def get_repeat(tensor):
    return K.repeat_elements(tensor, 2, axis=1)


def get_mean(tensor):
    return K.mean(tensor, axis=1)


def get_sigmoid(tensor):
    return K.sigmoid(tensor)


def squeeze(tensor):
    return K.squeeze(tensor, axis=1)


def get_abs(tensor):
    return K.abs(tensor)


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


def simple_model_pairwise():
    def pairwise_loss(input_tensors):
        def custom_loss(y_true, y_pred):
            y_t, y_f = input_tensors
            sub = y_t - y_f
            sub = K.sigmoid(sub)
            print(K.int_shape(sub))
            return K.mean(K.binary_crossentropy(y_true, sub), axis=-1)
        return custom_loss

    global MODEL_NAME
    MODEL_NAME = 'simple_pairwise' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_true_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_true_input')
    shop_false_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_false_input')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    w_e, s_t_e, s_f_e = embedding_layer(wifi_input), embedding_layer(shop_true_input), embedding_layer(shop_false_input)

    r = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')

    w, s_t, s_f = r(w_e), r(s_t_e), r(s_f_e)
    sim_w_s_t, sim_w_s_f = subtract([w, s_t]), subtract([w, s_f])

    d1, d2 = Dense(NUM_DENSE, activation='relu'), Dense(1, activation='sigmoid')  # , activation='sigmoid'
    score_t, score_f = d1(sim_w_s_t), d1(sim_w_s_f)
    score_t, score_f = d2(score_t), d2(score_f)

    score = score_f
    # score = Lambda(get_sigmoid)(score_f)

    model = Model(inputs=[wifi_input, shop_true_input, shop_false_input], outputs=score)
    model.compile(loss=pairwise_loss([score_t, score_f]), optimizer='nadam')

    return model


def simple_model_v2():
    global MODEL_NAME
    MODEL_NAME = 'simple_bi_v2' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    rec_input = Input(shape=(1,), dtype='float32', name='rec_input')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    r = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    w, s = r(w_e), r(s_e)
    # avg_lambda = Lambda(lambda a: K.mean(a, axis=1))
    # w, s = avg_lambda(w), avg_lambda(s)

    sim_w_s = Subtract()([w, s])
    rec = RepeatVector(REC_REPEAT)(rec_input)
    rec = Flatten()(rec)
    sim_con = concatenate([sim_w_s, rec])

    score = Dense(NUM_DENSE, activation='relu')(sim_con)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, rec_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def simple_model_v3():
    # rec 取分最高的过网络
    global MODEL_NAME
    MODEL_NAME = 'simple_bi_v3' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    rec_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    w_e, s_e, r_e = embedding_layer(wifi_input), embedding_layer(shop_input), embedding_layer(rec_input)

    r = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    w, s, rec = r(w_e), r(s_e), r(r_e)
    # avg_lambda = Lambda(lambda a: K.mean(a, axis=1))
    # w, s = avg_lambda(w), avg_lambda(s)

    sim_w_s = subtract([w, s])
    sim_r_s = subtract([s, rec])
    sim_con = concatenate([sim_w_s, sim_r_s])

    score = Dense(NUM_DENSE, activation='relu')(sim_con)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, rec_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def simple_model_v4():
    # rec取分最高的n个像se一样过网络att
    global MODEL_NAME
    MODEL_NAME = 'simple_bi_v4' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    rec_input_0 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_0')
    rec_input_1 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_1')
    rec_input_2 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_2')
    rec_input_3 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_3')
    rec_input_4 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_4')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)
    r_e_0 = embedding_layer(rec_input_0)
    r_e_1 = embedding_layer(rec_input_1)
    r_e_2 = embedding_layer(rec_input_2)
    r_e_3 = embedding_layer(rec_input_3)
    r_e_4 = embedding_layer(rec_input_4)

    bigru = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    bigru_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    w, s = bigru(w_e), bigru(s_e)
    s_r = bigru_r(s_e)
    r_0 = bigru_r(r_e_0)
    r_1 = bigru_r(r_e_1)
    r_2 = bigru_r(r_e_2)
    r_3 = bigru_r(r_e_3)
    r_4 = bigru_r(r_e_4)

    align = AlignSubLayer()
    r_0 = align([s_r, r_0])
    r_1 = align([s_r, r_1])
    r_2 = align([s_r, r_2])
    r_3 = align([s_r, r_3])
    r_4 = align([s_r, r_4])

    # print(K.int_shape(r_0))
    rs = Lambda(get_stack)([r_0, r_1, r_2, r_3, r_4])
    print(K.int_shape(rs))
    smatt = SoftmaxAttLayer(main_tensor=s)
    sim_s_rec = smatt(rs)
    sim_w_s = subtract([w, s])
    # sim_con = concatenate([sim_w_s, r_0])
    sim_con = concatenate([sim_w_s, sim_s_rec])

    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, rec_input_0, rec_input_1, rec_input_2, rec_input_3, rec_input_4], outputs=score)  # , rec_input_1, rec_input_2, rec_input_3, rec_input_4
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def simple_model_v5():
    # rec取分最高的n个过网络att  网络新想法 双对齐sub gruaggregation
    global MODEL_NAME
    MODEL_NAME = 'simple_bi_v5' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    rec_input_0 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_0')
    rec_input_1 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_1')
    rec_input_2 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_2')
    # rec_input_3 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_3')
    # rec_input_4 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_4')
    # rec_input_5 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_5')
    # rec_input_6 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_6')
    # rec_input_7 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_7')
    # rec_input_8 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_8')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)
    r_e_0 = embedding_layer(rec_input_0)
    r_e_1 = embedding_layer(rec_input_1)
    r_e_2 = embedding_layer(rec_input_2)
    # r_e_3 = embedding_layer(rec_input_3)
    # r_e_4 = embedding_layer(rec_input_4)
    # r_e_5 = embedding_layer(rec_input_5)
    # r_e_6 = embedding_layer(rec_input_6)
    # r_e_7 = embedding_layer(rec_input_7)
    # r_e_8 = embedding_layer(rec_input_8)

    bigru = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    bigru_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    w, s = bigru(w_e), bigru(s_e)
    s_r = bigru_r(s_e)
    r_0 = bigru_r(r_e_0)
    r_1 = bigru_r(r_e_1)
    r_2 = bigru_r(r_e_2)
    # r_3 = bigru_r(r_e_3)
    # r_4 = bigru_r(r_e_4)
    # r_5 = bigru_r(r_e_5)
    # r_6 = bigru_r(r_e_6)
    # r_7 = bigru_r(r_e_7)
    # r_8 = bigru_r(r_e_8)

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
    rs = Lambda(get_stack)([r_0, r_1, r_2])#, r_3])#, r_4, r_5, r_6])#, r_7, r_8])
    # print(K.int_shape(rs))

    s_agg = bigruagg(s_r)  # 使用shop过网络的结果做RECs的聚合
    smatt = SoftmaxAttLayer(main_tensor=s_agg)
    sim_s_rec = smatt(rs)
    # sim_s_rec = r_0

    # sim_s_rec = Lambda(get_mean)(rs)
    sim_w_s = subtract([w, s])
    sim_con = concatenate([sim_w_s, sim_s_rec])

    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, rec_input_0, rec_input_1, rec_input_2],#, rec_input_3],#, rec_input_4,
                          # rec_input_5, rec_input_6],#, rec_input_7, rec_input_8],
                  outputs=score)  # , rec_input_1, rec_input_2, rec_input_3, rec_input_4
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def simple_model_v5_single():
    # rec取分最高的n个过网络att  网络新想法 双对齐sub gruaggregation
    # 1rec时v5训练过拟合 试试把1rec写成固定网络 毕竟之前自己选最大的过网络都没有出现过拟合
    global MODEL_NAME
    MODEL_NAME = 'simple_bi_v5single' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    rec_input_0 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_0')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)
    r_e_0 = embedding_layer(rec_input_0)

    bigru = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    bigru_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    w, s = bigru(w_e), bigru(s_e)
    s_r = bigru_r(s_e)
    r_0 = bigru_r(r_e_0)

    bigruagg = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    weight = dot([s_r, r_0], axes=2)
    weight_j = Lambda(get_softmax_row)(weight)
    weight_i = Lambda(get_softmax)(weight)
    weighted_i = dot([weight_i, s_r], axes=1)
    weighted_j = dot([weight_j, r_0], axes=[2, 1])
    output_i = subtract([s_r, weighted_j])
    output_j = subtract([r_0, weighted_i])
    output_i, output_j = bigruagg(output_i), bigruagg(output_j)
    con = Lambda(get_stack)([output_i, output_j])
    output = Lambda(get_mean)(con)
    print(K.int_shape(output))
    sim_w_s = subtract([w, s])
    sim_con = concatenate([sim_w_s, output])

    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, rec_input_0], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model

def s2s_model():
    global MODEL_NAME
    MODEL_NAME = 's2s_model'
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    sim = S2SMLayer(filter_num=100, filter_size=3)([w_e, s_e])
    # sim = AlignSubLayer()([w_e, s_e])
    print(K.int_shape(sim))

    score = Dense(NUM_DENSE, activation='relu')(sim)
    score = Dense(1, activation='sigmoid')(score)
    print(K.int_shape(score))

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def baseline_rec(name):
    global MODEL_NAME
    assert name == 'hiem' or name == 'dm_hy'
    MODEL_NAME = name + '_rec' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    rec_input_0 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_0')
    rec_input_1 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_1')
    rec_input_2 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_2')
    # rec_input_3 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_3')
    # rec_input_4 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_4')
    # rec_input_5 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_5')
    # rec_input_6 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_6')
    # rec_input_7 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_7')
    # rec_input_8 = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input_8')

    embedding_layer = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1, input_length=MAX_SEQ_LENGTH,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    r_e_0 = embedding_layer(rec_input_0)
    r_e_1 = embedding_layer(rec_input_1)
    r_e_2 = embedding_layer(rec_input_2)
    # r_e_3 = embedding_layer(rec_input_3)
    # r_e_4 = embedding_layer(rec_input_4)
    # r_e_5 = embedding_layer(rec_input_5)
    # r_e_6 = embedding_layer(rec_input_6)
    # r_e_7 = embedding_layer(rec_input_7)
    # r_e_8 = embedding_layer(rec_input_8)

    if name == 'hiem':
        # w-s hiem model
        embedding_layer_hiem = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                         embeddings_initializer='uniform', mask_zero=True, trainable=True)
        wifi_e, shop_e = embedding_layer_hiem(wifi_input), embedding_layer_hiem(shop_input)
        ws_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
        ws_r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
        w, s = ws_r2(ws_r(wifi_e)), ws_r2(ws_r(shop_e))

        inter = InterAttLayer()
        w_inter, s_inter = inter([w, s])
        sub_i = subtract([w, w_inter])
        mul_i = multiply([w, w_inter])
        w_inter = concatenate([sub_i, mul_i], axis=-1)
        sub_j = subtract([s, s_inter])
        mul_j = multiply([s, s_inter])
        s_inter = concatenate([sub_j, mul_j], axis=-1)

        intra_attention = TimeDistributed(Dense(1))
        squ, soft = Lambda(squeeze), Lambda(get_softmax)
        beta_w = intra_attention(w_inter)
        beta_s = intra_attention(s_inter)
        beta_w, beta_s = soft(beta_w), soft(beta_s)
        print(K.int_shape(beta_w))
        w_intra = dot([beta_w, w_inter], axes=1)
        s_intra = dot([beta_s, s_inter], axes=1)
        w_intra, s_intra = squ(w_intra), squ(s_intra)
        sim_wifi_shop = concatenate([w_intra, s_intra])
    elif name == 'dm_hy':
        # w-s dm_hy model
        wifi_e, shop_e = embedding_layer(wifi_input), embedding_layer(shop_input)
        rnn1, rnn2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat'), \
                     Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
        h1, h2 = HighwayLayer(), HighwayLayer()
        w_a, s_a = h1(h2(wifi_e)), h1(h2(shop_e))
        w_rnn1, s_rnn1 = rnn1(wifi_e), rnn1(shop_e)
        w_rnn2, s_rnn2 = rnn2(wifi_e), rnn2(shop_e)
        d5, d6 = Dense(units=1, activation='relu'), Dense(units=NN_DIM * 2, activation='tanh')
        h3, h4, h5 = HighwayLayer(), HighwayLayer(), HighwayLayer()

        softalign_w_s = dot(inputs=[w_a, Lambda(get_transpose)(s_a)], axes=(2, 1))
        softalign_w_s = Lambda(get_softmax)(softalign_w_s)
        s_a_avg = dot(inputs=[softalign_w_s, s_rnn1], axes=1)
        w_comparison = concatenate(inputs=[w_rnn1, s_a_avg], axis=-1)
        w_comparison = h3(h4(w_comparison))
        s_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(s_rnn2)
        w_comparison_weight = concatenate(inputs=[w_comparison, s_rnn2_rp], axis=-1)
        w_comparison_weight = d5(h5(w_comparison_weight))
        w_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(w_comparison_weight)
        w_comparison_weight = Lambda(get_softmax)(w_comparison_weight)
        w_aggregation = dot(inputs=[w_comparison_weight, w_comparison], axes=1)

        softalign_s_w = dot(inputs=[s_a, Lambda(get_transpose)(w_a)], axes=(2, 1))
        softalign_s_w = Lambda(get_softmax)(softalign_s_w)
        w_a_avg = dot(inputs=[softalign_s_w, w_rnn1], axes=1)
        s_comparison = concatenate(inputs=[s_rnn1, w_a_avg], axis=-1)
        s_comparison = h3(h4(s_comparison))
        w_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(w_rnn2)
        s_comparison_weight = concatenate(inputs=[s_comparison, w_rnn2_rp], axis=-1)
        s_comparison_weight = d5(h5(s_comparison_weight))
        s_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(s_comparison_weight)
        s_comparison_weight = Lambda(get_softmax)(s_comparison_weight)
        s_aggregation = dot(inputs=[s_comparison_weight, s_comparison], axes=1)

        sim_w_s = subtract(inputs=[w_aggregation, s_aggregation])
        sim_wifi_shop = Lambda(lambda a: K.abs(a))(sim_w_s)

    bigru_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    if name == 'hiem':
        s_e = embedding_layer(shop_input)
        s_r = bigru_r(s_e)
    else:
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
    rs = Lambda(get_stack)([r_0, r_1, r_2])#, r_3])#, r_4, r_5, r_6])#, r_7, r_8])
    # print(K.int_shape(rs))

    s_agg = bigruagg(s_r)  # 使用shop过网络的结果做RECs的聚合
    smatt = SoftmaxAttLayer(main_tensor=s_agg)
    sim_s_rec = smatt(rs)

    sim_con = concatenate([sim_wifi_shop, sim_s_rec])

    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, rec_input_0, rec_input_1, rec_input_2],#, rec_input_3],#, rec_input_4,
                          # rec_input_5, rec_input_6],#, rec_input_7, rec_input_8],
                  outputs=score)  # , rec_input_1, rec_input_2, rec_input_3, rec_input_4
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def baseline_sr(name):
    global MODEL_NAME
    assert name == 'hiem' or name == 'dm_hy'
    MODEL_NAME = name + '_sr' + start_time
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

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
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

    if name == 'hiem':
        # w-s hiem model
        embedding_layer_hiem = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                         embeddings_initializer='uniform', mask_zero=True, trainable=True)
        wifi_e, shop_e = embedding_layer_hiem(wifi_input), embedding_layer_hiem(shop_input)
        ws_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
        ws_r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
        w, s = ws_r2(ws_r(wifi_e)), ws_r2(ws_r(shop_e))

        inter = InterAttLayer()
        w_inter, s_inter = inter([w, s])
        sub_i = subtract([w, w_inter])
        mul_i = multiply([w, w_inter])
        w_inter = concatenate([sub_i, mul_i], axis=-1)
        sub_j = subtract([s, s_inter])
        mul_j = multiply([s, s_inter])
        s_inter = concatenate([sub_j, mul_j], axis=-1)

        intra_attention = TimeDistributed(Dense(1))
        squ, soft = Lambda(squeeze), Lambda(get_softmax)
        beta_w = intra_attention(w_inter)
        beta_s = intra_attention(s_inter)
        beta_w, beta_s = soft(beta_w), soft(beta_s)
        print(K.int_shape(beta_w))
        w_intra = dot([beta_w, w_inter], axes=1)
        s_intra = dot([beta_s, s_inter], axes=1)
        w_intra, s_intra = squ(w_intra), squ(s_intra)
        sim_wifi_shop = concatenate([w_intra, s_intra])
    elif name == 'dm_hy':
        # w-s dm_hy model
        wifi_e, shop_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_input)
        rnn1, rnn2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat'), \
                     Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
        h1, h2 = HighwayLayer(), HighwayLayer()
        w_a, s_a = h1(h2(wifi_e)), h1(h2(shop_e))
        w_rnn1, s_rnn1 = rnn1(wifi_e), rnn1(shop_e)
        w_rnn2, s_rnn2 = rnn2(wifi_e), rnn2(shop_e)
        d5, d6 = Dense(units=1, activation='relu'), Dense(units=NN_DIM * 2, activation='tanh')
        h3, h4, h5 = HighwayLayer(), HighwayLayer(), HighwayLayer()

        softalign_w_s = dot(inputs=[w_a, Lambda(get_transpose)(s_a)], axes=(2, 1))
        softalign_w_s = Lambda(get_softmax)(softalign_w_s)
        s_a_avg = dot(inputs=[softalign_w_s, s_rnn1], axes=1)
        w_comparison = concatenate(inputs=[w_rnn1, s_a_avg], axis=-1)
        w_comparison = h3(h4(w_comparison))
        s_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(s_rnn2)
        w_comparison_weight = concatenate(inputs=[w_comparison, s_rnn2_rp], axis=-1)
        w_comparison_weight = d5(h5(w_comparison_weight))
        w_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(w_comparison_weight)
        w_comparison_weight = Lambda(get_softmax)(w_comparison_weight)
        w_aggregation = dot(inputs=[w_comparison_weight, w_comparison], axes=1)

        softalign_s_w = dot(inputs=[s_a, Lambda(get_transpose)(w_a)], axes=(2, 1))
        softalign_s_w = Lambda(get_softmax)(softalign_s_w)
        w_a_avg = dot(inputs=[softalign_s_w, w_rnn1], axes=1)
        s_comparison = concatenate(inputs=[s_rnn1, w_a_avg], axis=-1)
        s_comparison = h3(h4(s_comparison))
        w_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(w_rnn2)
        s_comparison_weight = concatenate(inputs=[s_comparison, w_rnn2_rp], axis=-1)
        s_comparison_weight = d5(h5(s_comparison_weight))
        s_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(s_comparison_weight)
        s_comparison_weight = Lambda(get_softmax)(s_comparison_weight)
        s_aggregation = dot(inputs=[s_comparison_weight, s_comparison], axes=1)

        sim_w_s = subtract(inputs=[w_aggregation, s_aggregation])
        sim_wifi_shop = Lambda(lambda a: K.abs(a))(sim_w_s)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
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

    print(K.int_shape(sr_0))
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6])#, sr_7, sr_8, sr_9])
    print(K.int_shape(ts))
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)

    sim_con = concatenate([sim_wifi_shop, sr])
    # sim_con = concatenate([sim_w_s, sr])
    # print(K.int_shape(sim_con))

    # sim_con = Dropout(rate=0.2)(sim_con)  # 试把dropout放到后面

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)  # NUM_DENSE * 2
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
                          sr_input_4, sr_input_5, sr_input_6],#, sr_input_7, sr_input_8, sr_input_9],
                  outputs=score)
    # model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'

    return model


def bigru_complex_model():
    global MODEL_NAME
    MODEL_NAME = 'our_bigru_complex_' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    ws_input = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='ws_input')

    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_2')
    sr_input_3 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_3')
    sr_input_4 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_4')
    sr_input_5 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_5')
    sr_input_6 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_6')
    sr_input_7 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_7')
    sr_input_8 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_8')
    sr_input_9 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_9')

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_input)
    ws_e = embedding_layer(ws_input)

    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)
    sr_e_3 = embedding_layer(sr_input_3)
    sr_e_4 = embedding_layer(sr_input_4)
    sr_e_5 = embedding_layer(sr_input_5)
    sr_e_6 = embedding_layer(sr_input_6)
    sr_e_7 = embedding_layer(sr_input_7)
    sr_e_8 = embedding_layer(sr_input_8)
    sr_e_9 = embedding_layer(sr_input_9)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r3 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r = GRU(units=NN_DIM, return_sequences=True)
    # w, s, ws = r(w_e), r(s_e), r(ws_e)
    # avg_lambda = Lambda(lambda a: K.mean(a, axis=1))
    # w, s, ws = avg_lambda(w), avg_lambda(s), avg_lambda(ws)
    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    w, s = rs(w_e), rs(s_e)
    ws = r(ws_e)
    # ws = r2(ws)
    # ws = r3(ws)
    ws = Lambda(lambda a: a[:, -1, :])(ws)

    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)
    # sr_0 = r2(sr_0)
    # sr_1 = r2(sr_1)
    # sr_2 = r2(sr_2)
    # sr_0 = r3(sr_0)
    # sr_1 = r3(sr_1)
    # sr_2 = r3(sr_2)
    sr_3 = r(sr_e_3)
    sr_4 = r(sr_e_4)
    sr_5 = r(sr_e_5)
    sr_6 = r(sr_e_6)
    sr_7 = r(sr_e_7)
    sr_8 = r(sr_e_8)
    sr_9 = r(sr_e_9)

    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr_0 = smatt(sr_0)
    sr_1 = smatt(sr_1)
    sr_2 = smatt(sr_2)
    sr_3 = smatt(sr_3)
    sr_4 = smatt(sr_4)
    sr_5 = smatt(sr_5)
    sr_6 = smatt(sr_6)
    sr_7 = smatt(sr_7)
    sr_8 = smatt(sr_8)
    sr_9 = smatt(sr_9)

    # print(K.int_shape(sr_0))
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6, sr_7, sr_8, sr_9])
    # print(K.int_shape(ts))
    sr = smatt(ts)

    # # print(K.int_shape(ts))
    # ts_transpose = Lambda(get_transpose)(ts)
    # # ts_transpose = Permute((0, 2, 1))(ts)
    # # print(K.int_shape(ts_transpose))
    # weights = dot([ws, ts_transpose], axes=1)
    # # print(K.int_shape(weights))
    # weights = Lambda(get_softmax)(weights)
    # # a = Lambda(d)(weights)
    # # print(K.int_shape(weights))
    # # w_0 = dot([ws, sr_0], axes=1)
    # # print(K.int_shape(w_0))
    # sr = dot([weights, ts], axes=1)
    # # print(K.int_shape(sr))
    # # print(ws.shape, sr.shape)

    sim_w_s = subtract([w, s])
    sim_ws_sr = subtract([ws, sr])
    # print(sim_ws_sr.shape)

    sim_con = concatenate([sim_w_s, sim_ws_sr])
    # print(K.int_shape(sim_con))

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)
    # score = Dropout(0.5)(score)
    # score = Dense(NUM_SEC_DENSE * 2, activation='relu')(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
                          sr_input_4, sr_input_5, sr_input_6, sr_input_7, sr_input_8, sr_input_9], outputs=score)
    # model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def bigru_complex_model_pairwise():
    def pairwise_loss(input_tensors):
        def custom_loss(y_true, y_pred):
            y_t, y_f = input_tensors
            sub = y_t - y_f
            sub = K.sigmoid(sub)
            print(K.int_shape(sub))
            return K.mean(K.binary_crossentropy(y_true, sub), axis=-1)
        return custom_loss
    global MODEL_NAME
    MODEL_NAME = 'bigru_complex_pairwise' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_true_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_true_input')
    shop_false_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_false_input')
    ws_true_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_true_input')
    ws_false_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_false_input')
    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_2')

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    wifi_e, shop_true_e, shop_false_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_true_input), \
                                        embedding_layer_s(shop_false_input)
    ws_true_e, ws_false_e = embedding_layer(ws_true_input), embedding_layer(ws_false_input)
    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    wifi, shop_t, shop_f = rs(wifi_e), rs(shop_true_e), rs(shop_false_e)
    ws_t = r(ws_true_e)
    ws_f = r(ws_false_e)
    ws_t = Lambda(lambda a: a[:, -1, :])(ws_t)
    ws_f = Lambda(lambda a: a[:, -1, :])(ws_f)
    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)

    smatt_t, smatt_f = SoftmaxAttLayer(main_tensor=ws_t), SoftmaxAttLayer(main_tensor=ws_f)
    sr_0_t, sr_1_t, sr_2_t = smatt_t(sr_0), smatt_t(sr_1), smatt_t(sr_2)
    sr_0_f, sr_1_f, sr_2_f = smatt_f(sr_0), smatt_f(sr_1), smatt_f(sr_2)
    ts_t = Lambda(get_stack)([sr_0_t, sr_1_t, sr_2_t])
    ts_f = Lambda(get_stack)([sr_0_f, sr_1_f, sr_2_f])
    sr_t, sr_f = smatt_t(ts_t), smatt_f(ts_f)

    sim_wifi_shop_t, sim_wifi_shop_f = subtract([wifi, shop_t]), subtract([wifi, shop_f])
    sim_ws_sr_t, sim_ws_sr_f = subtract([ws_t, sr_t]), subtract([ws_f, sr_f])
    sim_con_t, sim_con_f = concatenate([sim_wifi_shop_t, sim_ws_sr_t]), concatenate([sim_wifi_shop_f, sim_ws_sr_f])

    d1, d2 = Dense(NUM_DENSE * 2, activation='relu'), Dense(1, activation='sigmoid')  # , activation='sigmoid'
    score_t, score_f = d1(sim_con_t), d1(sim_con_f)
    score_t, score_f = d2(score_t), d2(score_f)
    score = score_f
    model = Model(inputs=[wifi_input, shop_true_input, shop_false_input, ws_true_input,
                          ws_false_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss=pairwise_loss([score_t, score_f]), optimizer='nadam')  # 'binary_crossentropy' , metrics=['acc']

    return model


def align_complex_model():
    global MODEL_NAME
    MODEL_NAME = 'align_complex_' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    ws_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_input')

    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_2')
    sr_input_3 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_3')
    sr_input_4 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_4')
    sr_input_5 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_5')
    sr_input_6 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_6')
    sr_input_7 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_7')
    sr_input_8 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_8')
    sr_input_9 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_9')

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_input)
    ws_e = embedding_layer(ws_input)

    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)
    sr_e_3 = embedding_layer(sr_input_3)
    sr_e_4 = embedding_layer(sr_input_4)
    sr_e_5 = embedding_layer(sr_input_5)
    sr_e_6 = embedding_layer(sr_input_6)
    sr_e_7 = embedding_layer(sr_input_7)
    sr_e_8 = embedding_layer(sr_input_8)
    sr_e_9 = embedding_layer(sr_input_9)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r = GRU(units=NN_DIM, return_sequences=True)
    # w, s, ws = r(w_e), r(s_e), r(ws_e)
    # avg_lambda = Lambda(lambda a: K.mean(a, axis=1))
    # w, s, ws = avg_lambda(w), avg_lambda(s), avg_lambda(ws)
    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    w, s = rs(w_e), rs(s_e)
    ws = r(ws_e)
    # ws = r2(ws)
    print(K.int_shape(ws))

    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)
    # sr_0 = r2(sr_0)
    # sr_1 = r2(sr_1)
    # sr_2 = r2(sr_2)
    sr_3 = r(sr_e_3)
    sr_4 = r(sr_e_4)
    sr_5 = r(sr_e_5)
    sr_6 = r(sr_e_6)
    sr_7 = r(sr_e_7)
    sr_8 = r(sr_e_8)
    sr_9 = r(sr_e_9)

    # align = AlignLayer(context_tensor=ws)
    # sr_0 = align(sr_0)
    # sr_1 = align(sr_1)
    # sr_2 = align(sr_2)
    # # sr_3 = align(sr_3)
    # # sr_4 = align(sr_4)
    # # sr_5 = align(sr_5)
    # # sr_6 = align(sr_6)
    # # sr_7 = align(sr_7)
    # # sr_8 = align(sr_8)
    # # sr_9 = align(sr_9)
    #
    # print(K.int_shape(sr_0))
    # ts = Lambda(get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6, sr_7, sr_8, sr_9])
    # print(K.int_shape(ts))
    # ws = Lambda(lambda a: a[:, -1, :])(ws)
    # # ws = Lambda(get_mean)(ws)
    # ws = Lambda(get_repeat)(ws)
    # smatt = SoftmaxAttLayer(main_tensor=ws)
    # sr = smatt(ts)
    #
    # sim_w_s = subtract([w, s])
    # sim_w_s = Lambda(get_repeat)(sim_w_s)
    # print(K.int_shape(sim_w_s))
    # sim_ws_sr = subtract([ws, sr])
    # print(sim_ws_sr.shape)
    # sim_con = concatenate([sim_w_s, sim_ws_sr])

    # bn = BatchNormalization()
    # sim_w_s = bn(sim_w_s)
    # sr = bn(sr)

    align = AlignSubLayer()
    sr_0 = align([ws, sr_0])
    sr_1 = align([ws, sr_1])
    sr_2 = align([ws, sr_2])
    sr_3 = align([ws, sr_3])
    sr_4 = align([ws, sr_4])
    sr_5 = align([ws, sr_5])
    sr_6 = align([ws, sr_6])
    sr_7 = align([ws, sr_7])
    sr_8 = align([ws, sr_8])
    sr_9 = align([ws, sr_9])

    print(K.int_shape(sr_0))
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6, sr_7, sr_8, sr_9])
    print(K.int_shape(ts))
    ws = Lambda(lambda a: a[:, -1, :])(ws)
    # ws = Lambda(get_mean)(ws)
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)
    sim_w_s = subtract([w, s])
    # print(K.int_shape(sim_w_s))

    sim_con = concatenate([sim_w_s, sr])
    print(K.int_shape(sim_con))

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)
    # score = Dropout(0.5)(score)
    # score = Dense(NUM_SEC_DENSE, activation='relu')(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
                          sr_input_4, sr_input_5, sr_input_6, sr_input_7, sr_input_8, sr_input_9], outputs=score)
    # model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def align_complex_model_pairwise():
    def pairwise_loss(input_tensors):
        def custom_loss(y_true, y_pred):
            y_t, y_f = input_tensors
            sub = y_t - y_f
            sub = K.sigmoid(sub)
            print(K.int_shape(sub))
            return K.mean(K.binary_crossentropy(y_true, sub), axis=-1)
        return custom_loss
    global MODEL_NAME
    MODEL_NAME = 'align_complex_pairwise' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_true_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_true_input')
    shop_false_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_false_input')
    ws_true_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_true_input')
    ws_false_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_false_input')
    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_2')

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    wifi_e, shop_true_e, shop_false_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_true_input), \
                                        embedding_layer_s(shop_false_input)
    ws_true_e, ws_false_e = embedding_layer(ws_true_input), embedding_layer(ws_false_input)
    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    wifi, shop_t, shop_f = rs(wifi_e), rs(shop_true_e), rs(shop_false_e)
    ws_t = r(ws_true_e)
    ws_f = r(ws_false_e)
    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)

    align = AlignSubLayer()
    sr_0_t = align([ws_t, sr_0])
    sr_1_t = align([ws_t, sr_1])
    sr_2_t = align([ws_t, sr_2])
    sr_0_f = align([ws_f, sr_0])
    sr_1_f = align([ws_f, sr_1])
    sr_2_f = align([ws_f, sr_2])

    ws_t = Lambda(lambda a: a[:, -1, :])(ws_t)
    ws_f = Lambda(lambda a: a[:, -1, :])(ws_f)
    print(K.int_shape(sr_0))
    ts_t = Lambda(get_stack)([sr_0_t, sr_1_t, sr_2_t])
    ts_f = Lambda(get_stack)([sr_0_f, sr_1_f, sr_2_f])
    smatt_t, smatt_f = SoftmaxAttLayer(main_tensor=ws_t), SoftmaxAttLayer(main_tensor=ws_f)
    sr_t, sr_f = smatt_t(ts_t), smatt_f(ts_f)

    sim_wifi_shop_t, sim_wifi_shop_f = subtract([wifi, shop_t]), subtract([wifi, shop_f])
    sim_con_t, sim_con_f = concatenate([sim_wifi_shop_t, sr_t]), concatenate([sim_wifi_shop_f, sr_f])

    d1, d2 = Dense(NUM_DENSE * 2, activation='relu'), Dense(1, activation='sigmoid')  # , activation='sigmoid'
    score_t, score_f = d1(sim_con_t), d1(sim_con_f)
    score_t, score_f = d2(score_t), d2(score_f)
    score = score_f
    model = Model(inputs=[wifi_input, shop_true_input, shop_false_input, ws_true_input,
                          ws_false_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss=pairwise_loss([score_t, score_f]), optimizer='nadam')  # 'binary_crossentropy' , metrics=['acc']

    return model


def www19_complex_model():
    global MODEL_NAME
    MODEL_NAME = 'www19' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    ws_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_input')

    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_2')
    # sr_input_3 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_3')
    # sr_input_4 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_4')
    # sr_input_5 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_5')
    # sr_input_6 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_6')
    # sr_input_7 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_7')
    # sr_input_8 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_8')
    # sr_input_9 = Input(shape=(MAX_S_SEQ_LENGTH, ), dtype='int32', name='sr_input_9')

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_input)
    ws_e = embedding_layer(ws_input)

    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)
    # sr_e_3 = embedding_layer(sr_input_3)
    # sr_e_4 = embedding_layer(sr_input_4)
    # sr_e_5 = embedding_layer(sr_input_5)
    # sr_e_6 = embedding_layer(sr_input_6)
    # sr_e_7 = embedding_layer(sr_input_7)
    # sr_e_8 = embedding_layer(sr_input_8)
    # sr_e_9 = embedding_layer(sr_input_9)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r = GRU(units=NN_DIM, return_sequences=True)
    # w, s, ws = r(w_e), r(s_e), r(ws_e)
    # avg_lambda = Lambda(lambda a: K.mean(a, axis=1))
    # w, s, ws = avg_lambda(w), avg_lambda(s), avg_lambda(ws)
    rs = Bidirectional(GRU(units=NN_DIM, dropout=dropout), merge_mode='concat')
    w, s = rs(w_e), rs(s_e)
    ws = r(ws_e)
    ws = r2(ws)
    print(K.int_shape(ws))

    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)
    sr_0 = r2(sr_0)
    sr_1 = r2(sr_1)
    sr_2 = r2(sr_2)
    # sr_3 = r(sr_e_3)
    # sr_4 = r(sr_e_4)
    # sr_5 = r(sr_e_5)
    # sr_6 = r(sr_e_6)
    # sr_7 = r(sr_e_7)
    # sr_8 = r(sr_e_8)
    # sr_9 = r(sr_e_9)

    from WifiShop.utilsLayer import WWW19Layer
    www = WWW19Layer(main_tensor=ws)
    sr_0 = www(sr_0)
    sr_1 = www(sr_1)
    sr_2 = www(sr_2)
    # sr_3 = smatt(sr_3)
    # sr_4 = smatt(sr_4)
    # sr_5 = smatt(sr_5)
    # sr_6 = smatt(sr_6)
    # sr_7 = smatt(sr_7)
    # sr_8 = smatt(sr_8)
    # sr_9 = smatt(sr_9)

    # ts = Lambda(get_stack)([sr_0, sr_1, sr_2])  # , sr_3, sr_4, sr_5, sr_6, sr_7, sr_8, sr_9])
    # sr = smatt(ts)

    print(K.int_shape(sr_0))
    sr = average([sr_0, sr_1, sr_2])
    print(K.int_shape(sr))
    # # print(K.int_shape(ts))
    # ts_transpose = Lambda(get_transpose)(ts)
    # # ts_transpose = Permute((0, 2, 1))(ts)
    # # print(K.int_shape(ts_transpose))
    # weights = dot([ws, ts_transpose], axes=1)
    # # print(K.int_shape(weights))
    # weights = Lambda(get_softmax)(weights)
    # # a = Lambda(d)(weights)
    # # print(K.int_shape(weights))
    # # w_0 = dot([ws, sr_0], axes=1)
    # # print(K.int_shape(w_0))
    # sr = dot([weights, ts], axes=1)
    # # print(K.int_shape(sr))
    # # print(ws.shape, sr.shape)

    sim_w_s = subtract([w, s])
    sim_w_s = Lambda(get_repeat)(sim_w_s)
    print(K.int_shape(sim_w_s))
    # sim_ws_sr = subtract([ws, sr])
    # print(sim_ws_sr.shape)

    # bn = BatchNormalization()
    # sim_w_s = bn(sim_w_s)
    # sr = bn(sr)

    sim_con = concatenate([sim_w_s, sr])
    print(K.int_shape(sim_con))

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    # model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
    #                       sr_input_4, sr_input_5, sr_input_6, sr_input_7, sr_input_8, sr_input_9], outputs=score)
    model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

    return model


def bigru_complex_model_v2():
    global MODEL_NAME
    MODEL_NAME = 'our_bigru_complex_v2' + start_time
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

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    wifi_e, shop_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_input)

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

    # rnn1, rnn2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat'), \
    #              Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    #
    # h1, h2 = HighwayLayer(), HighwayLayer()
    # w_a, s_a = h1(h2(wifi_e)), h1(h2(shop_e))
    #
    # # w_a, s_a = w_e, s_e
    # # print(K.int_shape(w_a), K.int_shape(s_a))
    #
    # w_rnn1, s_rnn1 = rnn1(wifi_e), rnn1(shop_e)
    # w_rnn2, s_rnn2 = rnn2(wifi_e), rnn2(shop_e)
    # d5, d6 = Dense(units=1, activation='relu'), Dense(units=NN_DIM * 2, activation='tanh')
    # h3, h4, h5 = HighwayLayer(), HighwayLayer(), HighwayLayer()
    # # D1, D2 = Dropout(0.3), Dropout(0.3)
    #
    # softalign_w_s = dot(inputs=[w_a, Lambda(get_transpose)(s_a)], axes=(2, 1))
    # softalign_w_s = Lambda(get_softmax)(softalign_w_s)
    # s_a_avg = dot(inputs=[softalign_w_s, s_rnn1], axes=1)
    # w_comparison = concatenate(inputs=[w_rnn1, s_a_avg], axis=-1)
    # w_comparison = h3(h4(w_comparison))
    # s_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(s_rnn2)
    # w_comparison_weight = concatenate(inputs=[w_comparison, s_rnn2_rp], axis=-1)
    # w_comparison_weight = d5(h5(w_comparison_weight))
    # w_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(w_comparison_weight)
    # w_comparison_weight = Lambda(get_softmax)(w_comparison_weight)
    # w_aggregation = dot(inputs=[w_comparison_weight, w_comparison], axes=1)
    #
    # softalign_s_w = dot(inputs=[s_a, Lambda(get_transpose)(w_a)], axes=(2, 1))
    # softalign_s_w = Lambda(get_softmax)(softalign_s_w)
    # w_a_avg = dot(inputs=[softalign_s_w, w_rnn1], axes=1)
    # s_comparison = concatenate(inputs=[s_rnn1, w_a_avg], axis=-1)
    # s_comparison = h3(h4(s_comparison))
    # w_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(w_rnn2)
    # s_comparison_weight = concatenate(inputs=[s_comparison, w_rnn2_rp], axis=-1)
    # s_comparison_weight = d5(h5(s_comparison_weight))
    # s_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(s_comparison_weight)
    # s_comparison_weight = Lambda(get_softmax)(s_comparison_weight)
    # s_aggregation = dot(inputs=[s_comparison_weight, s_comparison], axes=1)
    #
    # sim_w_s = subtract(inputs=[w_aggregation, s_aggregation])
    # sim_w_s = Lambda(lambda a: K.abs(a))(sim_w_s)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')

    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    wifi, shop = rs(wifi_e), rs(shop_e)
    s = r(s_e)
    ws = r(ws_e)
    # s = r2(s)
    # ws = r2(ws)
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
    # sr_0 = r2(sr_0)
    # sr_1 = r2(sr_1)
    # sr_2 = r2(sr_2)

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

    # alignonly = AlignOnlySubLayer()  # 把intraatt换成了bigru
    # alignr = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    # sr_0 = alignr(alignonly([s, sr_0]))
    # sr_1 = alignr(alignonly([s, sr_1]))
    # sr_2 = alignr(alignonly([s, sr_2]))
    # sr_3 = alignr(alignonly([s, sr_3]))
    # sr_4 = alignr(alignonly([s, sr_4]))
    # sr_5 = alignr(alignonly([s, sr_5]))
    # sr_6 = alignr(alignonly([s, sr_6]))
    # sr_7 = alignr(alignonly([s, sr_7]))
    # sr_8 = alignr(alignonly([s, sr_8]))
    # sr_9 = alignr(alignonly([s, sr_9]))

    print(K.int_shape(sr_0))
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6])#, sr_7, sr_8, sr_9])
    print(K.int_shape(ts))
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)

    sim_wifi_shop = subtract([wifi, shop])
    # print(K.int_shape(sim_wifi_shop))

    sim_con = concatenate([sim_wifi_shop, sr])
    # sim_con = concatenate([sim_w_s, sr])
    # print(K.int_shape(sim_con))

    sim_con = Dropout(rate=0.2)(sim_con)  # 试把dropout放到后面

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)  # NUM_DENSE * 2
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
                          sr_input_4, sr_input_5, sr_input_6],#, sr_input_7, sr_input_8, sr_input_9],
                  outputs=score)
    # model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'

    return model


def bigru_complex_model_v2_pairwise():
    def pairwise_loss(input_tensors):
        def custom_loss(y_true, y_pred):
            y_t, y_f = input_tensors
            sub = y_t - y_f
            sub = K.sigmoid(sub)
            print(K.int_shape(sub))
            return K.mean(K.binary_crossentropy(y_true, sub), axis=-1)
        return custom_loss

    global MODEL_NAME
    MODEL_NAME = 'our_bigru_complex_v3_pairwise' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_true_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_true_input')
    shop_false_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_false_input')
    s_true_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='s_true_input')
    s_false_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='s_false_input')
    ws_true_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_true_input')
    ws_false_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_false_input')

    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_2')

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    wifi_e, shop_true_e, shop_false_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_true_input), \
                                        embedding_layer_s(shop_false_input)
    s_true_e, s_false_e, ws_true_e, ws_false_e = embedding_layer(s_true_input), embedding_layer(s_false_input), \
                                                 embedding_layer(ws_true_input), embedding_layer(ws_false_input)
    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    wifi, shop_t, shop_f = rs(wifi_e), rs(shop_true_e), rs(shop_false_e)
    s_t, s_f = r(s_true_e), r(s_false_e)
    ws_t = r(ws_true_e)
    ws_t = Lambda(lambda a: a[:, -1, :])(ws_t)
    ws_f = r(ws_false_e)
    ws_f = Lambda(lambda a: a[:, -1, :])(ws_f)
    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)

    align = AlignSubLayer()
    sr_0_t = align([s_t, sr_0])
    sr_1_t = align([s_t, sr_1])
    sr_2_t = align([s_t, sr_2])
    sr_0_f = align([s_f, sr_0])
    sr_1_f = align([s_f, sr_1])
    sr_2_f = align([s_f, sr_2])

    print(K.int_shape(sr_0))
    ts_t = Lambda(get_stack)([sr_0_t, sr_1_t, sr_2_t])
    ts_f = Lambda(get_stack)([sr_0_f, sr_1_f, sr_2_f])
    smatt_t, smatt_f = SoftmaxAttLayer(main_tensor=ws_t), SoftmaxAttLayer(main_tensor=ws_f)
    sr_t, sr_f = smatt_t(ts_t), smatt_f(ts_f)

    sim_wifi_shop_t, sim_wifi_shop_f = subtract([wifi, shop_t]), subtract([wifi, shop_f])
    sim_con_t, sim_con_f = concatenate([sim_wifi_shop_t, sr_t]), concatenate([sim_wifi_shop_f, sr_f])

    d1, d2 = Dense(NUM_DENSE * 2, activation='relu'), Dense(1, activation='sigmoid')  # , activation='sigmoid'
    # bn = BatchNormalization()
    # sim_con_t, sim_con_f = bn(sim_con_t), bn(sim_con_f)
    score_t, score_f = d1(sim_con_t), d1(sim_con_f)
    score_t, score_f = d2(score_t), d2(score_f)

    score = score_f
    # score = Lambda(get_sigmoid)(score_f)

    # model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
    #                       sr_input_4, sr_input_5, sr_input_6, sr_input_7, sr_input_8, sr_input_9], outputs=score)
    model = Model(inputs=[wifi_input, shop_true_input, shop_false_input, s_true_input, s_false_input, ws_true_input,
                          ws_false_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss=pairwise_loss([score_t, score_f]), optimizer='nadam')  # 'binary_crossentropy' , metrics=['acc']

    return model


def bigru_complex_model_v3():
    global MODEL_NAME
    MODEL_NAME = 'our_bigru_complex_v3' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    s_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='s_input')
    ws_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_input')

    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_2')
    # sr_input_3 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_3')
    # sr_input_4 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_4')

    rec_input = Input(shape=(1,), dtype='float32', name='rec_input')
    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    wifi_e, shop_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_input)

    s_e, ws_e = embedding_layer(s_input), embedding_layer(ws_input)
    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)
    # sr_e_3 = embedding_layer(sr_input_3)
    # sr_e_4 = embedding_layer(sr_input_4)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')

    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    wifi, shop = rs(wifi_e), rs(shop_e)
    s = r(s_e)
    ws = r(ws_e)
    # s = r2(s)
    # ws = r2(ws)
    ws = Lambda(lambda a: a[:, -1, :])(ws)

    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)
    # sr_3 = r(sr_e_3)
    # sr_4 = r(sr_e_4)
    # sr_0 = r2(sr_0)
    # sr_1 = r2(sr_1)
    # sr_2 = r2(sr_2)

    align = AlignSubLayer()
    sr_0 = align([s, sr_0])
    sr_1 = align([s, sr_1])
    sr_2 = align([s, sr_2])
    # sr_3 = align(sr_3)
    # sr_4 = align(sr_4)

    print(K.int_shape(sr_0))
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2])#, sr_3, sr_4])#, sr_5, sr_6, sr_7, sr_8, sr_9])
    print(K.int_shape(ts))
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)

    sim_wifi_shop = subtract([wifi, shop])
    print(K.int_shape(sim_wifi_shop))

    rec = RepeatVector(REC_REPEAT)(rec_input)
    rec = Flatten()(rec)
    sim_con = concatenate([sim_wifi_shop, sr, rec])
    print(K.int_shape(sim_con))

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)  # NUM_DENSE * 2
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    # model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
    #                       sr_input_4, sr_input_5, sr_input_6, sr_input_7, sr_input_8, sr_input_9], outputs=score)
    model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2, rec_input], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'

    return model


def bigru_complex_model_v4():
    global MODEL_NAME
    MODEL_NAME = 'our_bigru_complex_v4' + start_time
    print('Create model =', MODEL_NAME)

    wifi_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='shop_input')
    rec_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name='rec_input')
    s_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='s_input')
    ws_input = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='ws_input')

    sr_input_0 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_2')
    # sr_input_3 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_3')
    # sr_input_4 = Input(shape=(MAX_S_SEQ_LENGTH,), dtype='int32', name='sr_input_4')

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    wifi_e, shop_e, rec_e = embedding_layer_s(wifi_input), embedding_layer_s(shop_input), embedding_layer_s(rec_input)

    s_e, ws_e = embedding_layer(s_input), embedding_layer(ws_input)
    sr_e_0 = embedding_layer(sr_input_0)
    sr_e_1 = embedding_layer(sr_input_1)
    sr_e_2 = embedding_layer(sr_input_2)
    # sr_e_3 = embedding_layer(sr_input_3)
    # sr_e_4 = embedding_layer(sr_input_4)

    r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    # r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')

    rs = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    wifi, shop, rec = rs(wifi_e), rs(shop_e), rs(rec_e)
    s = r(s_e)
    ws = r(ws_e)
    # s = r2(s)
    # ws = r2(ws)
    ws = Lambda(lambda a: a[:, -1, :])(ws)

    sr_0 = r(sr_e_0)
    sr_1 = r(sr_e_1)
    sr_2 = r(sr_e_2)
    # sr_3 = r(sr_e_3)
    # sr_4 = r(sr_e_4)
    # sr_0 = r2(sr_0)
    # sr_1 = r2(sr_1)
    # sr_2 = r2(sr_2)

    align = AlignSubLayer()
    sr_0 = align([s, sr_0])
    sr_1 = align([s, sr_1])
    sr_2 = align([s, sr_2])
    # sr_3 = align(sr_3)
    # sr_4 = align(sr_4)

    print(K.int_shape(sr_0))
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2])#, sr_3, sr_4])#, sr_5, sr_6, sr_7, sr_8, sr_9])
    print(K.int_shape(ts))
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)

    sim_wifi_shop = subtract([wifi, shop])
    print(K.int_shape(sim_wifi_shop))
    sim_shop_rec = subtract([shop, rec])

    sim_con = concatenate([sim_wifi_shop, sim_shop_rec, sr])
    print(K.int_shape(sim_con))

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)  # NUM_DENSE * 2
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    # model = Model(inputs=[wifi_input, shop_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
    #                       sr_input_4, sr_input_5, sr_input_6, sr_input_7, sr_input_8, sr_input_9], outputs=score)
    model = Model(inputs=[wifi_input, shop_input, rec_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'

    return model


def combine_model_v1():
    global MODEL_NAME
    MODEL_NAME = 'combine_v1' + start_time
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

    rs = Lambda(get_stack)([r_0, r_1, r_2])#, r_3, r_4, r_5, r_6, r_7, r_8, r_9])
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
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6])#, sr_7])#, sr_8])#, sr_9])
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
                          sr_input_4, sr_input_5, sr_input_6,# sr_input_7,# sr_input_8,# sr_input_9,
                          rec_input_0, rec_input_1, rec_input_2],#, rec_input_3, rec_input_4, rec_input_5, rec_input_6,
                          #rec_input_7, rec_input_8, rec_input_9],
                  outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'

    return model


def combine_dm_hy():
    global MODEL_NAME
    MODEL_NAME = 'combine_rec_sr_dm_hy' + start_time
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

    # w-s dm_hy model
    rnn1, rnn2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat'), \
                 Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    h1, h2 = HighwayLayer(), HighwayLayer()
    w_a, s_a = h1(h2(wifi_e)), h1(h2(shop_e))
    w_rnn1, s_rnn1 = rnn1(wifi_e), rnn1(shop_e)
    w_rnn2, s_rnn2 = rnn2(wifi_e), rnn2(shop_e)
    d5, d6 = Dense(units=1, activation='relu'), Dense(units=NN_DIM * 2, activation='tanh')
    h3, h4, h5 = HighwayLayer(), HighwayLayer(), HighwayLayer()

    softalign_w_s = dot(inputs=[w_a, Lambda(get_transpose)(s_a)], axes=(2, 1))
    softalign_w_s = Lambda(get_softmax)(softalign_w_s)
    s_a_avg = dot(inputs=[softalign_w_s, s_rnn1], axes=1)
    w_comparison = concatenate(inputs=[w_rnn1, s_a_avg], axis=-1)
    w_comparison = h3(h4(w_comparison))
    s_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(s_rnn2)
    w_comparison_weight = concatenate(inputs=[w_comparison, s_rnn2_rp], axis=-1)
    w_comparison_weight = d5(h5(w_comparison_weight))
    w_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(w_comparison_weight)
    w_comparison_weight = Lambda(get_softmax)(w_comparison_weight)
    w_aggregation = dot(inputs=[w_comparison_weight, w_comparison], axes=1)

    softalign_s_w = dot(inputs=[s_a, Lambda(get_transpose)(w_a)], axes=(2, 1))
    softalign_s_w = Lambda(get_softmax)(softalign_s_w)
    w_a_avg = dot(inputs=[softalign_s_w, w_rnn1], axes=1)
    s_comparison = concatenate(inputs=[s_rnn1, w_a_avg], axis=-1)
    s_comparison = h3(h4(s_comparison))
    w_rnn2_rp = RepeatVector(MAX_SEQ_LENGTH)(w_rnn2)
    s_comparison_weight = concatenate(inputs=[s_comparison, w_rnn2_rp], axis=-1)
    s_comparison_weight = d5(h5(s_comparison_weight))
    s_comparison_weight = Lambda(lambda a: K.squeeze(a, axis=-1))(s_comparison_weight)
    s_comparison_weight = Lambda(get_softmax)(s_comparison_weight)
    s_aggregation = dot(inputs=[s_comparison_weight, s_comparison], axes=1)

    sim_w_s = subtract(inputs=[w_aggregation, s_aggregation])
    sim_wifi_shop = Lambda(lambda a: K.abs(a))(sim_w_s)

    # rec model
    bigru_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    s_r = bigru_r(shop_e)
    r_0 = bigru_r(r_e_0)
    r_1 = bigru_r(r_e_1)
    r_2 = bigru_r(r_e_2)
    # r_3 = bigru_r(r_e_3)
    # r_4 = bigru_r(r_e_4)

    bigruagg = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    bialignagg = BiAlignAggLayer(nn_dim=NN_DIM, agg_nn=bigruagg)
    r_0 = bialignagg([s_r, r_0])
    r_1 = bialignagg([s_r, r_1])
    r_2 = bialignagg([s_r, r_2])
    # r_3 = bialignagg([s_r, r_3])
    # r_4 = bialignagg([s_r, r_4])

    rs = Lambda(get_stack)([r_0, r_1, r_2])#, r_3, r_4])#, sr_5, sr_6, sr_7, sr_8, sr_9])
    print(K.int_shape(rs))

    s_agg = bigruagg(s_r)  # 使用shop过网络的结果做RECs的聚合
    smatt_sagg = SoftmaxAttLayer(main_tensor=s_agg)
    sim_s_rec = smatt_sagg(rs)
    # sim_s_rec = Lambda(get_mean)(rs)

    # sr model
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

    print(K.int_shape(sr_0))
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6])#, sr_7, sr_8, sr_9])
    print(K.int_shape(ts))
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)

    sim_con = concatenate([sim_wifi_shop, sim_s_rec, sr])
    print(K.int_shape(sim_con))

    sim_con = Dropout(rate=0.4)(sim_con)  # 试把dropout放到后面

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)  # NUM_DENSE * 2
    # score = Dense(NUM_DENSE, activation='relu')(score)
    score = Dense(NUM_SEC_DENSE, activation='relu')(score)
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
                          sr_input_4, sr_input_5, sr_input_6,# sr_input_7, sr_input_8, sr_input_9,
                          rec_input_0, rec_input_1, rec_input_2],#, rec_input_3, rec_input_4],
                  outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'

    return model


def combine_hiem():
    global MODEL_NAME
    MODEL_NAME = 'combine_rec_sr_hiem' + start_time
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

    embedding_layer = Embedding(output_dim=CHAR_LEN_c, input_dim=GRAM_LEN_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                  weights=[embedding_matrix_s],
                                  mask_zero=True, trainable=False)
    embedding_layer_hiem = Embedding(output_dim=CHAR_LEN_s, input_dim=GRAM_LEN_s + 1,
                                     embeddings_initializer='uniform', mask_zero=True, trainable=True)
    wifi_e, shop_e = embedding_layer_hiem(wifi_input), embedding_layer_hiem(shop_input)
    s_forrec = embedding_layer_s(shop_input)
    r_e_0 = embedding_layer_s(rec_input_0)
    r_e_1 = embedding_layer_s(rec_input_1)
    r_e_2 = embedding_layer_s(rec_input_2)
    # r_e_3 = embedding_layer_s(rec_input_3)
    # r_e_4 = embedding_layer_s(rec_input_4)

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

    # w-s hiem model
    ws_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    ws_r2 = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    w, s = ws_r2(ws_r(wifi_e)), ws_r2(ws_r(shop_e))

    inter = InterAttLayer()
    w_inter, s_inter = inter([w, s])
    sub_i = subtract([w, w_inter])
    mul_i = multiply([w, w_inter])
    w_inter = concatenate([sub_i, mul_i], axis=-1)
    sub_j = subtract([s, s_inter])
    mul_j = multiply([s, s_inter])
    s_inter = concatenate([sub_j, mul_j], axis=-1)

    intra_attention = TimeDistributed(Dense(1))
    squ, soft = Lambda(squeeze), Lambda(get_softmax)
    beta_w = intra_attention(w_inter)
    beta_s = intra_attention(s_inter)
    beta_w, beta_s = soft(beta_w), soft(beta_s)
    print(K.int_shape(beta_w))
    w_intra = dot([beta_w, w_inter], axes=1)
    s_intra = dot([beta_s, s_inter], axes=1)
    w_intra, s_intra = squ(w_intra), squ(s_intra)

    sim_wifi_shop = concatenate([w_intra, s_intra])

    # rec model
    bigru_r = Bidirectional(GRU(units=NN_DIM, return_sequences=True), merge_mode='concat')
    s_r = bigru_r(s_forrec)  # 原来用shop_e 觉得不对 改了
    r_0 = bigru_r(r_e_0)
    r_1 = bigru_r(r_e_1)
    r_2 = bigru_r(r_e_2)
    # r_3 = bigru_r(r_e_3)
    # r_4 = bigru_r(r_e_4)

    bigruagg = Bidirectional(GRU(units=NN_DIM), merge_mode='concat')
    bialignagg = BiAlignAggLayer(nn_dim=NN_DIM, agg_nn=bigruagg)
    r_0 = bialignagg([s_r, r_0])
    r_1 = bialignagg([s_r, r_1])
    r_2 = bialignagg([s_r, r_2])
    # r_3 = bialignagg([s_r, r_3])
    # r_4 = bialignagg([s_r, r_4])

    rs = Lambda(get_stack)([r_0, r_1, r_2])#, r_3, r_4])#, sr_5, sr_6, sr_7, sr_8, sr_9])
    print(K.int_shape(rs))

    s_agg = bigruagg(s_r)  # 使用shop过网络的结果做RECs的聚合
    smatt_sagg = SoftmaxAttLayer(main_tensor=s_agg)
    sim_s_rec = smatt_sagg(rs)
    # sim_s_rec = Lambda(get_mean)(rs)

    # sr model
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

    print(K.int_shape(sr_0))
    ts = Lambda(get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6])#, sr_7, sr_8, sr_9])
    print(K.int_shape(ts))
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)

    sim_con = concatenate([sim_wifi_shop, sim_s_rec, sr])
    print(K.int_shape(sim_con))

    sim_con = Dropout(rate=0.4)(sim_con)  # 试把dropout放到后面

    score = Dense(NUM_DENSE * 2, activation='relu')(sim_con)  # NUM_DENSE * 2
    # score = Dense(NUM_DENSE, activation='relu')(score)
    score = Dense(NUM_SEC_DENSE, activation='relu')(score)
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
                          sr_input_4, sr_input_5, sr_input_6,# sr_input_7, sr_input_8, sr_input_9,
                          rec_input_0, rec_input_1, rec_input_2],#, rec_input_3, rec_input_4],
                  outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'

    return model


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


def extract_data_simple_rec(process_column, ngram=3):  # 带rec进网络 选一个最大的
    raw_data = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin'] + ['label'] + ['ltable_name', 'rtable_name']]
    raw_data['rec'] = ''
    rec_dict = dp.get_search_recommendation()
    for i in range(len(raw_data)):  # 转小写
        temp = raw_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        if rec_dict.__contains__(temp['ltable_name']):  # 选字符串最匹配的
            temp_score_list = list()
            for rec in rec_dict[temp['ltable_name']]:
                temp_score = sum(pf.jaccard(pf.get_ngram(rec, k, True), pf.get_ngram(temp['rtable_name'], k, True))
                                 for k in range(1, 4)) / 3 + 1 / (pf.edit_dis(rec, temp['rtable_name']) + 1)
                temp_score_list.append(temp_score)
            max_index = temp_score_list.index(max(temp_score_list))
            temp['rec'] = pf.chinese2pinyin(rec_dict[temp['ltable_name']][max_index])
        else:
            temp['rec'] = temp['ltable_pinyin']
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
    for i in range(len(new_data)):
        temp = new_data.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index.get(temp[column])]
            else:
                temp[column] = [gram2index.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        new_data.iloc[i] = temp

    return new_data


def extract_data_simple_rec_v2(process_column, ngram=3, fuzzy_rec=False):  # 带rec进网络 允许读取模糊查询获得的rec
    raw_data = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin'] + ['label'] + ['ltable_name', 'rtable_name']]
    if fuzzy_rec:
        rec_dict = dp.get_search_rec_all_with_fuzzy(dp.pois_global, ['r1', 'r2', 'r3'], statistic=False)  # , 'l1', 'l2'
    else:
        rec_dict = dp.get_search_recommendation()
    for i in range(len(raw_data)):  # 转小写
        temp = raw_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        raw_data.iloc[i] = temp

    gramset, charset = set(), set()  # 获取ngram
    rec_result = [list() for _ in range(MAX_REC_NUM)]
    for index, row in raw_data.iterrows():
        if rec_dict.__contains__(row['ltable_name']):
            temp_score_dict = dict()
            done = 0
            for rec in rec_dict[row['ltable_name']]:
                temp_score = sum(pf.jaccard(pf.get_ngram(rec, k, True), pf.get_ngram(row['rtable_name'], k, True))
                                 for k in range(1, 4)) / 3 + 1 / (pf.edit_dis(rec, row['rtable_name']) + 1)
                temp_score_dict[row['rtable_name']] = temp_score
                for name in sorted(temp_score_dict, key=temp_score_dict.__getitem__, reverse=True):
                    if done < MAX_REC_NUM:
                        this_pinyin = pf.chinese2pinyin(name)
                        charset = charset | set(this_pinyin)
                        if len(this_pinyin) < ngram:
                            grams = [this_pinyin]
                        else:
                            grams = [this_pinyin[i:i + ngram] for i in range(len(this_pinyin) - ngram + 1)]
                        gramset = gramset | set(grams)
                        rec_result[done].append(this_pinyin)
                        done += 1
                    else:
                        break
                while done < MAX_REC_NUM:
                    rec_result[done].append(row['ltable_pinyin'])
                    done += 1
        else:
            for _ in range(MAX_REC_NUM):
                rec_result[_].append(row['ltable_pinyin'])

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
    for i in range(len(new_data)):
        temp = new_data.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index.get(temp[column])]
            else:
                temp[column] = [gram2index.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        new_data.iloc[i] = temp
    for index, i in enumerate(rec_result):
        temp = list()
        for rec in i:
            if len(rec) < ngram:
                temp.append([gram2index.get(rec)])
            else:
                temp.append([gram2index.get(rec[j:j+ngram]) for j in range(len(rec) - ngram + 1)])
        rec_result[index] = sequence.pad_sequences(temp, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    return new_data, rec_result


def read_search_result(sources, clean=False, title=False):
    search_result = dict()
    f_name = 'wifi_search_result.txt'
    assert not (title & clean)
    if clean:
        print('Using clean SE')
        f_name = 'clean_' + f_name
    if title:
        print('Using title SE')
        f_name = 'title_' + f_name
    for source in sources:
        search_docs = dict()
        with open('{}/data_{}/{}'.format(se_path, source, f_name), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wifi, docs = line.strip().split('\t')
                docs = eval(docs)
                docs = docs[:MAX_SR_NUM]
                search_docs[wifi] = docs
        search_result[source] = search_docs
    return search_result


def extract_data_complex(process_column, sources, ngram=3, need_rec_score=False):
    raw_data = pd.read_csv('{}/linking/matching/our/match.csv'.format(ex_path))
    raw_data = raw_data[['ltable_pinyin', 'rtable_pinyin'] + ['label'] + ['ltable_name', 'rtable_name']]
    search_result = read_search_result(sources, clean=False, title=True)  # lowercase done; 读取去特殊字符停用词的就用clean 只要title就title
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

    print('extract {}-gram {}'.format(ngram, len(gramset)))
    print('extract character {}'.format(len(charset)))
    global GRAM_LEN_c, CHAR_LEN_c, embedding_matrix_c
    GRAM_LEN_c, CHAR_LEN_c = len(gramset), len(charset)
    embedding_matrix_c = np.zeros((GRAM_LEN_c + 1, CHAR_LEN_c), dtype=int)
    # embedding_matrix = np.delete(embedding_matrix, [0], axis=1)

    gram2index = {gram: index+1 for index, gram in enumerate(gramset)}
    index2gram = {gram2index[gram]: gram for gram in gram2index}
    char2index = {char: index for index, char in enumerate(charset)}
    # index2char = {char2index[char]: char for char in char2index}

    for index in index2gram:
        for char in index2gram[index]:
            embedding_matrix_c[index, char2index[char]] += 1

    new_data = raw_data.copy()
    if need_rec_score:
        rec_dict = dp.get_search_recommendation()
        new_data['rec_score'] = 0.00001
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
                temp['rec_score'] += score
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


def extract_data_simple_pairwise(process_column, ngram=3):
    raw_data = pd.read_csv('{}/linking/matching/our/match_pairwise.csv'.format(ex_path))
    raw_data = raw_data[['false_pinyin', 'true_pinyin', 'wifi_pinyin', 'wifi_name'] + ['label'] + ['wifi']]
    for i in range(len(raw_data)):
        temp = raw_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        raw_data.iloc[i] = temp

    gramset_s, charset_s = set(), set()
    for index, row in raw_data.iterrows():
        for column in process_column:
            charset_s = charset_s | set(row[column])
            if len(row[column]) < ngram:
                grams = [row[column]]
            else:
                grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
            gramset_s = gramset_s | set(grams)
    print('simple: extract {}-gram {}'.format(ngram, len(gramset_s)))
    print('simple: extract character {}'.format(len(charset_s)))
    global GRAM_LEN_s, CHAR_LEN_s, embedding_matrix_s
    GRAM_LEN_s, CHAR_LEN_s = len(gramset_s), len(charset_s)
    embedding_matrix_s = np.zeros((GRAM_LEN_s + 1, CHAR_LEN_s), dtype=int)
    gram2index_s = {gram: index + 1 for index, gram in enumerate(gramset_s)}
    index2gram_s = {gram2index_s[gram]: gram for gram in gram2index_s}
    char2index_s = {char: index for index, char in enumerate(charset_s)}
    for index in index2gram_s:
        for char in index2gram_s[index]:
            embedding_matrix_s[index, char2index_s[char]] += 1

    new_data_s = raw_data.copy()
    for i in range(len(new_data_s)):
        temp = new_data_s.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index_s.get(temp[column])]
            else:
                temp[column] = [gram2index_s.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        new_data_s.iloc[i] = temp

    return new_data_s


def extract_data_complex_pairwise(process_column, sources, ngram=3):
    raw_data = pd.read_csv('{}/linking/matching/our/match_pairwise.csv'.format(ex_path))
    raw_data = raw_data[['false_pinyin', 'true_pinyin', 'wifi_pinyin', 'wifi_name'] + ['label'] + ['wifi']]
    search_result = read_search_result(sources, clean=False)  # lowercase done; 读取去特殊字符停用词的就用clean
    for i in range(len(raw_data)):
        temp = raw_data.iloc[i].copy()
        for column in process_column:
            temp[column] = temp[column].lower()
        raw_data.iloc[i] = temp

    ws_data = raw_data[['false_pinyin', 'true_pinyin']].copy()
    for i in range(len(ws_data)):
        ws_data.iloc[i]['false_pinyin'] = raw_data.iloc[i]['wifi_pinyin'] + raw_data.iloc[i]['false_pinyin']
        ws_data.iloc[i]['true_pinyin'] = raw_data.iloc[i]['wifi_pinyin'] + raw_data.iloc[i]['true_pinyin']

    gramset_s, charset_s = set(), set()
    for index, row in raw_data.iterrows():
        for column in process_column:
            charset_s = charset_s | set(row[column])
            if len(row[column]) < ngram:
                grams = [row[column]]
            else:
                grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
            gramset_s = gramset_s | set(grams)
    gramset_c, charset_c = gramset_s, charset_s
    for index, row in ws_data.iterrows():
        for column in ['false_pinyin', 'true_pinyin']:
            charset_c = charset_c | set(row[column])
            if len(row[column]) < ngram:
                grams = [row[column]]
            else:
                grams = [row[column][i:i+ngram] for i in range(len(row[column]) - ngram + 1)]
            gramset_c = gramset_c | set(grams)
    for source in sources:
        for key in search_result[source].keys():
            for c in search_result[source][key]:
                charset_c = charset_c | set(c)
                if len(c) < ngram:
                    grams = [c]
                else:
                    grams = [c[i:i + ngram] for i in range(len(c) - ngram + 1)]
                gramset_c = gramset_c | set(grams)

    print('simple: extract {}-gram {}'.format(ngram, len(gramset_s)))
    print('simple: extract character {}'.format(len(charset_s)))
    global GRAM_LEN_s, CHAR_LEN_s, embedding_matrix_s
    GRAM_LEN_s, CHAR_LEN_s = len(gramset_s), len(charset_s)
    embedding_matrix_s = np.zeros((GRAM_LEN_s + 1, CHAR_LEN_s), dtype=int)
    gram2index_s = {gram: index + 1 for index, gram in enumerate(gramset_s)}
    index2gram_s = {gram2index_s[gram]: gram for gram in gram2index_s}
    char2index_s = {char: index for index, char in enumerate(charset_s)}
    for index in index2gram_s:
        for char in index2gram_s[index]:
            embedding_matrix_s[index, char2index_s[char]] += 1

    print('complex: extract {}-gram {}'.format(ngram, len(gramset_c)))
    print('complex: extract character {}'.format(len(charset_c)))
    global GRAM_LEN_c, CHAR_LEN_c, embedding_matrix_c
    GRAM_LEN_c, CHAR_LEN_c = len(gramset_c), len(charset_c)
    embedding_matrix_c = np.zeros((GRAM_LEN_c + 1, CHAR_LEN_c), dtype=int)
    gram2index_c = {gram: index+1 for index, gram in enumerate(gramset_c)}
    index2gram_c = {gram2index_c[gram]: gram for gram in gram2index_c}
    char2index_c = {char: index for index, char in enumerate(charset_c)}
    for index in index2gram_c:
        for char in index2gram_c[index]:
            embedding_matrix_c[index, char2index_c[char]] += 1

    new_data_s = raw_data.copy()
    for i in range(len(new_data_s)):
        temp = new_data_s.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index_s.get(temp[column])]
            else:
                temp[column] = [gram2index_s.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        new_data_s.iloc[i] = temp
    new_data_s = new_data_s[process_column]
    new_data_c = raw_data.copy()
    for i in range(len(new_data_c)):
        temp = new_data_c.iloc[i].copy()
        for column in process_column:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index_c.get(temp[column])]
            else:
                temp[column] = [gram2index_c.get(temp[column][j:j + ngram]) for j in
                                range(len(temp[column]) - ngram + 1)]
        new_data_c.iloc[i] = temp
    for i in range(len(ws_data)):
        temp = ws_data.iloc[i].copy()
        for column in ['false_pinyin', 'true_pinyin']:
            if len(temp[column]) < ngram:
                temp[column] = [gram2index_c.get(temp[column])]
            else:
                temp[column] = [gram2index_c.get(temp[column][j:j+ngram]) for j in range(len(temp[column]) - ngram + 1)]
        ws_data.iloc[i] = temp
    max_source_length = 0
    for source in sources:
        for key in search_result[source].keys():
            this = search_result[source][key]
            for index, c in enumerate(this):
                if len(c) < ngram:
                    this[index] = [gram2index_c.get(c)]
                else:
                    this[index] = [gram2index_c.get(c[j:j + ngram]) for j in range(len(c) - ngram + 1)]
                    if len(this[index]) > max_source_length:
                        max_source_length = len(this[index])
            search_result[source][key] = this
    global MAX_S_SEQ_LENGTH
    if max_source_length < MAX_S_SEQ_LENGTH:
        MAX_S_SEQ_LENGTH = max_source_length
        print('CHANGE MAX_S_SEQ_LENGTH =', MAX_S_SEQ_LENGTH)
    return new_data_s, new_data_c, ws_data, search_result


def generate_simple_pos4pairwise_insensitive(test_data, test, columns, n_p_ratio=5):
    # 同 generate_pos4pairwise_insentitive
    new_test = {c: list() for c in columns}
    new_pos_num = 0
    test_label = [0 for _ in range(len(test_data))]
    for i in range(int(len(test_data) / n_p_ratio)):
        index = i * n_p_ratio
        new_test['wifi_pinyin'].append(test['wifi_pinyin'][index])
        new_test['true_pinyin'].append(test['true_pinyin'][index])
        new_test['false_pinyin'].append(test['true_pinyin'][index])
        test_label.append(1)
        new_pos_num += 1
    print('generate {} pos in {} neg'.format(new_pos_num, len(test_data)))
    if new_pos_num:
        for c in columns:
            test[c] = np.r_[test[c], np.array(new_test[c])]
    return test_label


def generate_simple_pos4pairwise(done_wifi_posshop, test_data, test, columns):
    new_test = {c: list() for c in columns}
    new_pos_num, flag = 0, False
    test_label = [0 for _ in range(len(test_data))]
    for index in range(len(test_data)):
        if done_wifi_posshop.__contains__(test_data.iloc[index]['wifi']):
            if test['true_pinyin'][index].tolist() not in done_wifi_posshop[test_data.iloc[index]['wifi']]:
                done_wifi_posshop[test_data.iloc[index]['wifi']].append(test['true_pinyin'][index].tolist())
                flag = True
        else:
            done_wifi_posshop[test_data.iloc[index]['wifi']] = [test['true_pinyin'][index].tolist()]
            flag = True
        if flag:
            new_test['wifi_pinyin'].append(test['wifi_pinyin'][index])
            new_test['true_pinyin'].append(test['true_pinyin'][index])
            new_test['false_pinyin'].append(test['true_pinyin'][index])
            test_label.append(1)
            new_pos_num += 1
            flag = False
    print('generate {} pos in {} neg'.format(new_pos_num, len(test_data)))
    if new_pos_num:
        for c in columns:
            test[c] = np.r_[test[c], np.array(new_test[c])]
    return test_label


def generate_pos4pairwise_insensitive(test_data, test, test_s, test_ws, test_sr, columns, s_columns,
                                      need_s=True, n_p_ratio=5):
    # 这个方法适用于正负例1：n且需要手动分train test val的情况 为每n个pair直接取正例 对pos重复不敏感
    new_test, new_test_s, new_test_ws = {c: list() for c in columns}, {c: list() for c in s_columns},\
                                        {c: list() for c in s_columns}
    new_test_sr = [list() for _ in test_sr]
    new_pos_num = 0
    test_label = [0 for _ in range(len(test_data))]
    for i in range(int(len(test_data) / n_p_ratio)):
        index = i * n_p_ratio
        new_test['wifi_pinyin'].append(test['wifi_pinyin'][index])
        new_test['true_pinyin'].append(test['true_pinyin'][index])
        new_test['false_pinyin'].append(test['true_pinyin'][index])
        if need_s:
            new_test_s['true_pinyin'].append(test_s['true_pinyin'][index])
            new_test_s['false_pinyin'].append(test_s['true_pinyin'][index])
        new_test_ws['true_pinyin'].append(test_ws['true_pinyin'][index])
        new_test_ws['false_pinyin'].append(test_ws['true_pinyin'][index])
        for l_index, l in enumerate(new_test_sr):
            l.append(test_sr[l_index][index])
        test_label.append(1)
        new_pos_num += 1
    print('generate {} pos in {} neg'.format(new_pos_num, len(test_data)))
    if new_pos_num:
        for c in columns:
            test[c] = np.r_[test[c], np.array(new_test[c])]
        for c in s_columns:
            if need_s:
                test_s[c] = np.r_[test_s[c], np.array(new_test_s[c])]
            test_ws[c] = np.r_[test_ws[c], np.array(new_test_ws[c])]
        for index in range(len(test_sr)):
            test_sr[index] = np.r_[test_sr[index], np.array(new_test_sr[index])]
    return test_label


def generate_pos4pairwise(done_wifi_posshop, test_data, test, test_s, test_ws, test_sr, columns, s_columns, need_s=True):
    new_test, new_test_s, new_test_ws = {c: list() for c in columns}, {c: list() for c in s_columns}, \
                                        {c: list() for c in s_columns}
    new_test_sr = [list() for _ in test_sr]
    new_pos_num, flag = 0, False
    test_label = [0 for _ in range(len(test_data))]
    for index in range(len(test_data)):
        if done_wifi_posshop.__contains__(test_data.iloc[index]['wifi']):
            if test['true_pinyin'][index].tolist() not in done_wifi_posshop[test_data.iloc[index]['wifi']]:
                done_wifi_posshop[test_data.iloc[index]['wifi']].append(test['true_pinyin'][index].tolist())
                flag = True
        else:
            done_wifi_posshop[test_data.iloc[index]['wifi']] = [test['true_pinyin'][index].tolist()]
            flag = True
        if flag:
            new_test['wifi_pinyin'].append(test['wifi_pinyin'][index])
            new_test['true_pinyin'].append(test['true_pinyin'][index])
            new_test['false_pinyin'].append(test['true_pinyin'][index])
            if need_s:
                new_test_s['true_pinyin'].append(test_s['true_pinyin'][index])
                new_test_s['false_pinyin'].append(test_s['true_pinyin'][index])
            new_test_ws['true_pinyin'].append(test_ws['true_pinyin'][index])
            new_test_ws['false_pinyin'].append(test_ws['true_pinyin'][index])
            for l_index, l in enumerate(new_test_sr):
                l.append(test_sr[l_index][index])
            test_label.append(1)
            new_pos_num += 1
            flag = False
    print('generate {} pos in {} neg'.format(new_pos_num, len(test_data)))
    if new_pos_num:
        for c in columns:
            test[c] = np.r_[test[c], np.array(new_test[c])]
        for c in s_columns:
            if need_s:
                test_s[c] = np.r_[test_s[c], np.array(new_test_s[c])]
            test_ws[c] = np.r_[test_ws[c], np.array(new_test_ws[c])]
        for index in range(len(test_sr)):
            test_sr[index] = np.r_[test_sr[index], np.array(new_test_sr[index])]
    return test_label


def name2sr_onlyone(df, search_result):
    for i in range(len(df)):
        temp = df.iloc[i].copy()
        # sr = list()
        # for source in SOURCES:
        #     wifi_search = search_result[source]
        #     if wifi_search.__contains__(temp['ltable_name']):
        #         sr.extend(wifi_search[temp['ltable_name']])
        if search_result['baidu'].__contains__(df.iloc[i]['ltable_name']):
            if search_result['baidu'][df.iloc[i]['ltable_name']]:
                temp['ltable_name'] = search_result['baidu'][df.iloc[i]['ltable_name']][0]
            else:
                temp['ltable_name'] = []
        else:
            temp['ltable_name'] = []
        df.iloc[i] = temp
    return df


def name2sr(df, search_result):
    result = list()
    for i in range(len(df)):
        temp = df.iloc[i].copy()
        # sr = list()
        # for source in SOURCES:
        #     wifi_search = search_result[source]
        #     if wifi_search.__contains__(temp['ltable_name']):
        #         sr.extend(wifi_search[temp['ltable_name']])
        if search_result['baidu'].__contains__(df.iloc[i]['ltable_name']):
            if search_result['baidu'][df.iloc[i]['ltable_name']]:
                # temp['ltable_name'] = search_result['baidu'][df.iloc[i]['ltable_name']][0]
                result.append(sequence.pad_sequences(
                    [j for j in search_result['baidu'][df.iloc[i]['ltable_name']]]
                    + [[0] for _ in range(MAX_SR_NUM - len(search_result['baidu'][df.iloc[i]['ltable_name']]))],
                    maxlen=MAX_S_SEQ_LENGTH, padding='post', truncating='post').tolist())
            else:
                result.append(sequence.pad_sequences([[0] for _ in range(MAX_SR_NUM)], maxlen=MAX_S_SEQ_LENGTH,
                                                     padding='post').tolist())
        else:
            result.append(sequence.pad_sequences([[0] for _ in range(MAX_SR_NUM)], maxlen=MAX_S_SEQ_LENGTH,
                                                 padding='post').tolist())
    result = np.array(result)
    return result


def get_sr_respectively(df, search_result, c_name='ltable_name'):
    result = [list() for _ in range(MAX_SR_NUM)]
    for i in range(len(df)):
        if search_result['baidu'].__contains__(df.iloc[i][c_name]):
            if search_result['baidu'][df.iloc[i][c_name]]:
                for j in range(MAX_SR_NUM):
                    if j < len(search_result['baidu'][df.iloc[i][c_name]]):
                        result[j].append(search_result['baidu'][df.iloc[i][c_name]][j])
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
    for index, i in enumerate(result):
        result[index] = np.array(sequence.pad_sequences(i, maxlen=MAX_S_SEQ_LENGTH, padding='post', truncating='post'))
    return result


def get_sr_respectively_v2(df, search_result):
    result = list()
    for i in range(len(df)):
        if search_result['baidu'].__contains__(df.iloc[i]['ltable_name']):
            if search_result['baidu'][df.iloc[i]['ltable_name']]:
                for j in range(MAX_SR_NUM):
                    if j < len(search_result['baidu'][df.iloc[i]['ltable_name']]):
                        result[j].append(search_result['baidu'][df.iloc[i]['ltable_name']][j])
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
    for index, i in enumerate(result):
        result[index] = np.array(sequence.pad_sequences(i, maxlen=MAX_S_SEQ_LENGTH, padding='post', truncating='post'))
    return [result]


def our_simple_model(folds, m='rnn', bid=False, save_log=False):
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data, raw_datam = extract_data_simple(process_column=columns, ngram=3)
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

            model = s2s_model() #simple_model(m, bid)  # 'lstm' 'rnn' 'gru'
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


def our_simple_model_pairwise(folds, n_p_ratio=5):
    columns = ['wifi_pinyin', 'true_pinyin', 'false_pinyin']
    new_data_s = extract_data_simple_pairwise(process_column=columns, ngram=3)
    pre_result = list()
    # done_wifi_posshop = dict()

    k_fold = KFold(n_splits=folds, shuffle=True)
    shop_no_list = [_ for _ in range(int(len(new_data_s) / n_p_ratio))]
    for fold_num, (train_list_index, test_list_index) in enumerate(k_fold.split(shop_no_list)):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))

        train_index, test_index, val_index = list(), list(), list()
        v_fold = KFold(n_splits=10, shuffle=True)
        # t_shop_no_list = [shop_no_list[i] for i in train_list_index]
        for train_v_list_index, val_list_index in v_fold.split(train_list_index):
            for _ in train_v_list_index:
                train_index.extend([n_p_ratio * train_list_index[_] + k for k in range(n_p_ratio)])
            for _ in val_list_index:
                val_index.extend(([n_p_ratio * train_list_index[_] + k for k in range(n_p_ratio)]))
            for _ in test_list_index:
                test_index.extend([n_p_ratio * _ + k for k in range(n_p_ratio)])
            train, test, val = dict(), dict(), dict()
            for c in columns:
                tra = np.array(new_data_s.iloc[train_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
                tes = np.array(new_data_s.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
                va = np.array(new_data_s.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va
            train_label = new_data_s.iloc[train_index]['label']
            val_label = new_data_s.iloc[val_index]['label']
            # test_label = generate_simple_pos4pairwise(dict(), new_data_s.iloc[test_index], test, columns)
            test_label = generate_simple_pos4pairwise_insensitive(new_data_s.iloc[test_index], test, columns, n_p_ratio=n_p_ratio)

            model = simple_model_pairwise()
            # if not fold_num:
            #     print(model.summary())
            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns], train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=1,  # 2 1
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns], val_label),
                      callbacks=[
                          EarlyStopping(
                              monitor='val_loss',
                              min_delta=0.0001,
                              patience=3,
                              verbose=1,
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
            test_predict = model.predict([test[c] for c in columns], batch_size=BATCH_SIZE, verbose=1)
            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if test_label[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if test_label[index] == 1:
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


def our_simple_model_v2(folds, save_log=False):
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data, raw_datam = extract_data_simple(process_column=columns, ngram=3, need_rec_score=True)
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
            train_rec = np.array(new_data_train.iloc[t_index]['rec_score'])
            test_rec = np.array(new_data.iloc[test_index]['rec_score'])
            val_rec = np.array(new_data_train.iloc[val_index]['rec_score'])

            model = simple_model_v2()
            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns] + [train_rec], train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=2,
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + [val_rec], val_label),
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
            test_predict = model.predict([test[c] for c in columns] + [test_rec], batch_size=BATCH_SIZE, verbose=1)
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


def our_simple_model_v3(folds, save_log=False):
    columns = ['ltable_pinyin', 'rtable_pinyin', 'rec']
    new_data = extract_data_simple_rec(process_column=columns, ngram=3)
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

            model = simple_model_v3()
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


def our_simple_model_v4(folds, save_log=False):
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data, rec_result = extract_data_simple_rec_v2(process_column=columns, ngram=3, fuzzy_rec=False)
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
            train_rec, test_rec, val_rec = [0 for _ in range(MAX_REC_NUM)], [0 for _ in range(MAX_REC_NUM)], \
                                           [0 for _ in range(MAX_REC_NUM)]
            for i in range(MAX_REC_NUM):
                train_rec[i] = rec_result[i][train_index][t_index]
                test_rec[i] = rec_result[i][test_index]
                val_rec[i] = rec_result[i][train_index][val_index]

            model = baseline_rec(name='dm_hy')  # simple_model_v5_single()  baseline_rec(name='hiem')
            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns] + train_rec, train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=2,
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + val_rec, val_label),
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
            test_predict = model.predict([test[c] for c in columns] + test_rec, batch_size=BATCH_SIZE, verbose=1)
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


def our_complex_model(folds, onlyone=True, softmax=False, m='rnn'):
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data_s = extract_data_simple(process_column=columns, ngram=3)[0]  # 190515 simple_model low dimen
    new_data, ws_data, search_result = extract_data_complex(process_column=columns, sources=SOURCES, ngram=3)
    pre_result = list()

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
                # tra = np.array(new_data.iloc[train_index][c])
                tra = np.array(new_data_s_train.iloc[t_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
                # tes = np.array(new_data.iloc[test_index][c])
                tes = np.array(new_data_s.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
                va = np.array(new_data_s_train.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va
            train_ws = np.array(ws_data.iloc[train_index].iloc[t_index]['ltable_pinyin'])
            train_ws = sequence.pad_sequences(train_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_ws = np.array(ws_data.iloc[test_index]['ltable_pinyin'])
            test_ws = sequence.pad_sequences(test_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_ws = np.array(ws_data.iloc[train_index].iloc[val_index]['ltable_pinyin'])
            val_ws = sequence.pad_sequences(val_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']

            if onlyone:
                train_sr = name2sr_onlyone(new_data.iloc[train_index].copy(), search_result)
                train_sr = sequence.pad_sequences(np.array(train_sr['ltable_name']), maxlen=MAX_S_SEQ_LENGTH,
                                                  padding='post')
                test_sr = name2sr_onlyone(new_data.iloc[test_index].copy(), search_result)
                test_sr = sequence.pad_sequences(np.array(test_sr['ltable_name']), maxlen=MAX_S_SEQ_LENGTH, padding='post')
            else:
                # train_sr = name2sr(new_data_train.iloc[t_index].copy(), search_result)
                # test_sr = name2sr(new_data.iloc[test_index].copy(), search_result)
                # val_sr = name2sr(new_data_train.iloc[val_index].copy(), search_result)
                train_sr = get_sr_respectively(new_data_train.iloc[t_index].copy(), search_result)
                test_sr = get_sr_respectively(new_data.iloc[test_index].copy(), search_result)
                val_sr = get_sr_respectively(new_data_train.iloc[val_index].copy(), search_result)

            # model = complex_model(m, onlyone=onlyone, softmax=softmax)  # 'lstm' 'rnn' 'gru'
            model = align_complex_model()
            # bigru_complex_model  align_complex_model www19_model bigru_complex_model_v2
            # if not fold_num:
            #     print(model.summary())

            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns] + [train_ws] + train_sr, train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=1,  # 2
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + [val_ws] + val_sr, val_label),
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
            # test_result = model.evaluate([test[c] for c in columns] + [test_ws] + test_sr, test_label, batch_size=BATCH_SIZE, verbose=1)
            # print(test_result)
            # pre_result.append(test_result)
            test_predict = model.predict([test[c] for c in columns] + [test_ws] + test_sr, batch_size=BATCH_SIZE, verbose=1)
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


def our_complex_model_pairwise(folds, n_p_ratio=5):
    columns = ['wifi_pinyin', 'true_pinyin', 'false_pinyin']
    s_columns = ['true_pinyin', 'false_pinyin']
    new_data_s, new_data_c, ws_data, search_result = extract_data_complex_pairwise(process_column=columns,
                                                                                   sources=SOURCES, ngram=3)
    pre_result = list()

    k_fold = KFold(n_splits=folds, shuffle=True)
    assert len(new_data_c) % n_p_ratio == 0
    shop_no_list = [_ for _ in range(int(len(new_data_c) / n_p_ratio))]
    for fold_num, (train_list_index, test_list_index) in enumerate(k_fold.split(shop_no_list)):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))

        train_index, test_index, val_index = list(), list(), list()
        v_fold = KFold(n_splits=10, shuffle=True)
        # t_shop_no_list = [shop_no_list[i] for i in train_list_index]
        for train_v_list_index, val_list_index in v_fold.split(train_list_index):
            for _ in train_v_list_index:
                train_index.extend([n_p_ratio * train_list_index[_] + k for k in range(n_p_ratio)])
            for _ in val_list_index:
                val_index.extend(([n_p_ratio * train_list_index[_] + k for k in range(n_p_ratio)]))
            for _ in test_list_index:
                test_index.extend([n_p_ratio * _ + k for k in range(n_p_ratio)])
            train, test, val = dict(), dict(), dict()
            for c in columns:
                tra = np.array(new_data_s.iloc[train_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
                tes = np.array(new_data_s.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
                va = np.array(new_data_s.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va
            train_ws, test_ws, val_ws = dict(), dict(), dict()
            train_s, test_s, val_s = dict(), dict(), dict()
            for c in s_columns:
                train_ws[c] = np.array(ws_data.iloc[train_index][c])
                train_ws[c] = sequence.pad_sequences(train_ws[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')
                test_ws[c] = np.array(ws_data.iloc[test_index][c])
                test_ws[c] = sequence.pad_sequences(test_ws[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')
                val_ws[c] = np.array(ws_data.iloc[val_index][c])
                val_ws[c] = sequence.pad_sequences(val_ws[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_label = new_data_c.iloc[train_index]['label']
            val_label = new_data_c.iloc[val_index]['label']
            train_sr = get_sr_respectively(new_data_c.iloc[train_index].copy(), search_result, c_name='wifi_name')
            test_sr = get_sr_respectively(new_data_c.iloc[test_index].copy(), search_result, c_name='wifi_name')
            val_sr = get_sr_respectively(new_data_c.iloc[val_index].copy(), search_result, c_name='wifi_name')
            # test_label = generate_pos4pairwise(dict(), new_data_c.iloc[test_index], test, test_s, test_ws, test_sr,
            #                                    columns, s_columns)
            test_label = generate_pos4pairwise_insensitive(new_data_c.iloc[test_index], test, test_s, test_ws, test_sr,
                                                           columns, s_columns, need_s=False, n_p_ratio=n_p_ratio)

            model = align_complex_model_pairwise()  # bigru_complex_model_pairwise   align_complex_model_pairwise
            # if not fold_num:
            #     print(model.summary())
            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns] + [train_ws[c] for c in s_columns] + train_sr, train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=1,  # 2 1
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + [val_ws[c] for c in s_columns] + val_sr, val_label),
                      callbacks=[
                          EarlyStopping(
                              monitor='val_loss',
                              min_delta=0.0001,
                              patience=3,
                              verbose=1,
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
            test_predict = model.predict([test[c] for c in columns] + [test_ws[c] for c in s_columns] + test_sr,
                                         batch_size=BATCH_SIZE, verbose=1)
            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if test_label[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if test_label[index] == 1:
                        fn += 1
            print(tp, fp, fn)
            try:
                print(tp / (tp + fp), tp / (tp + fn))
            except Exception as e:
                print(e)
            pre_result.append([tp, fp, fn])

            K.clear_session()
            del train, test, train_ws, test_ws, train_label, test_label
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


def our_complex_model_v2(folds, save_log=False):
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data_s = extract_data_simple(process_column=columns, ngram=3)[0]  # 190515 simple_model low dimen
    new_data, ws_data, search_result = extract_data_complex(process_column=columns, sources=SOURCES, ngram=3)
    pre_result = list()

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
            train_ws = np.array(ws_data.iloc[train_index].iloc[t_index]['ltable_pinyin'])
            train_ws = sequence.pad_sequences(train_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_ws = np.array(ws_data.iloc[test_index]['ltable_pinyin'])
            test_ws = sequence.pad_sequences(test_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_ws = np.array(ws_data.iloc[train_index].iloc[val_index]['ltable_pinyin'])
            val_ws = sequence.pad_sequences(val_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_s = np.array(new_data_train.iloc[t_index]['rtable_pinyin'])
            train_s = sequence.pad_sequences(train_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_s = np.array(new_data.iloc[test_index]['rtable_pinyin'])
            test_s = sequence.pad_sequences(test_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_s = np.array(new_data_train.iloc[val_index]['rtable_pinyin'])
            val_s = sequence.pad_sequences(val_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']
            train_sr = get_sr_respectively(new_data_train.iloc[t_index].copy(), search_result)
            test_sr = get_sr_respectively(new_data.iloc[test_index].copy(), search_result)
            val_sr = get_sr_respectively(new_data_train.iloc[val_index].copy(), search_result)

            # model = complex_model(m, onlyone=onlyone, softmax=softmax)  # 'lstm' 'rnn' 'gru'
            model = baseline_sr(name='dm_hy')  # bigru_complex_model_v2
            # bigru_complex_model  align_complex_model www19_model bigru_complex_model_v2
            # if not fold_num:
            #     print(model.summary())

            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns] + [train_s, train_ws] + train_sr, train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=2,  # 2 1
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + [val_s, val_ws] + val_sr, val_label),
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
            # test_result = model.evaluate([test[c] for c in columns] + [test_ws] + test_sr, test_label, batch_size=BATCH_SIZE, verbose=1)
            # print(test_result)
            # pre_result.append(test_result)
            test_predict = model.predict([test[c] for c in columns] + [test_s, test_ws] + test_sr, batch_size=BATCH_SIZE, verbose=1)
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


def our_complex_model_v2_pairwise(folds, n_p_ratio=5):
    columns = ['wifi_pinyin', 'true_pinyin', 'false_pinyin']
    s_columns = ['true_pinyin', 'false_pinyin']
    new_data_s, new_data_c, ws_data, search_result = extract_data_complex_pairwise(process_column=columns,
                                                                                   sources=SOURCES, ngram=3)
    pre_result = list()
    # done_wifi_posshop = dict()

    k_fold = KFold(n_splits=folds, shuffle=True)
    assert len(new_data_c) % n_p_ratio == 0
    shop_no_list = [_ for _ in range(int(len(new_data_c) / n_p_ratio))]
    for fold_num, (train_list_index, test_list_index) in enumerate(k_fold.split(shop_no_list)):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))

        train_index, test_index, val_index = list(), list(), list()
        v_fold = KFold(n_splits=10, shuffle=True)
        # t_shop_no_list = [shop_no_list[i] for i in train_list_index]
        for train_v_list_index, val_list_index in v_fold.split(train_list_index):
            for _ in train_v_list_index:
                train_index.extend([n_p_ratio * train_list_index[_] + k for k in range(n_p_ratio)])
            for _ in val_list_index:
                val_index.extend(([n_p_ratio * train_list_index[_] + k for k in range(n_p_ratio)]))
            for _ in test_list_index:
                test_index.extend([n_p_ratio * _ + k for k in range(n_p_ratio)])
            train, test, val = dict(), dict(), dict()
            for c in columns:
                tra = np.array(new_data_s.iloc[train_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
                tes = np.array(new_data_s.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
                va = np.array(new_data_s.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va
            train_ws, test_ws, val_ws = dict(), dict(), dict()
            train_s, test_s, val_s = dict(), dict(), dict()
            for c in s_columns:
                train_ws[c] = np.array(ws_data.iloc[train_index][c])
                train_ws[c] = sequence.pad_sequences(train_ws[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')
                test_ws[c] = np.array(ws_data.iloc[test_index][c])
                test_ws[c] = sequence.pad_sequences(test_ws[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')
                val_ws[c] = np.array(ws_data.iloc[val_index][c])
                val_ws[c] = sequence.pad_sequences(val_ws[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')

                train_s[c] = np.array(new_data_c.iloc[train_index][c])
                train_s[c] = sequence.pad_sequences(train_s[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')
                test_s[c] = np.array(new_data_c.iloc[test_index][c])
                test_s[c] = sequence.pad_sequences(test_s[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')
                val_s[c] = np.array(new_data_c.iloc[val_index][c])
                val_s[c] = sequence.pad_sequences(val_s[c], maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_label = new_data_c.iloc[train_index]['label']
            val_label = new_data_c.iloc[val_index]['label']
            train_sr = get_sr_respectively(new_data_c.iloc[train_index].copy(), search_result, c_name='wifi_name')
            test_sr = get_sr_respectively(new_data_c.iloc[test_index].copy(), search_result, c_name='wifi_name')
            val_sr = get_sr_respectively(new_data_c.iloc[val_index].copy(), search_result, c_name='wifi_name')
            # test_label = generate_pos4pairwise(dict(), new_data_c.iloc[test_index], test, test_s, test_ws, test_sr, columns, s_columns)
            test_label = generate_pos4pairwise_insensitive(new_data_c.iloc[test_index], test, test_s, test_ws, test_sr,
                                                           columns, s_columns, n_p_ratio=n_p_ratio)

            model = bigru_complex_model_v2_pairwise()
            # if not fold_num:
            #     print(model.summary())
            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns] + [train_s[c] for c in s_columns] + [train_ws[c] for c in s_columns]
                      + train_sr, train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=1,  # 2 1
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + [val_s[c] for c in s_columns]
                                       + [val_ws[c] for c in s_columns] + val_sr, val_label),
                      callbacks=[
                          EarlyStopping(
                              monitor='val_loss',
                              min_delta=0.0001,
                              patience=3,
                              verbose=1,
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
            test_predict = model.predict([test[c] for c in columns] + [test_s[c] for c in s_columns]
                                         + [test_ws[c] for c in s_columns] + test_sr, batch_size=BATCH_SIZE, verbose=1)
            tp, fp, fn = 0, 0, 0
            for index, i in enumerate(test_predict):
                if i > 0.5:
                    if test_label[index] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if test_label[index] == 1:
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


def our_complex_model_v3(folds, save_log=False):
    # 这版用了recommendation信息 特征的形式
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data_s = extract_data_simple(process_column=columns, ngram=3)[0]  # 190515 simple_model low dimen
    new_data, ws_data, search_result = extract_data_complex(process_column=columns, sources=SOURCES, ngram=3,
                                                            need_rec_score=True)
    pre_result = list()

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

            train_ws = np.array(ws_data.iloc[train_index].iloc[t_index]['ltable_pinyin'])
            train_ws = sequence.pad_sequences(train_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_ws = np.array(ws_data.iloc[test_index]['ltable_pinyin'])
            test_ws = sequence.pad_sequences(test_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_ws = np.array(ws_data.iloc[train_index].iloc[val_index]['ltable_pinyin'])
            val_ws = sequence.pad_sequences(val_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_s = np.array(new_data_train.iloc[t_index]['rtable_pinyin'])
            train_s = sequence.pad_sequences(train_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_s = np.array(new_data.iloc[test_index]['rtable_pinyin'])
            test_s = sequence.pad_sequences(test_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_s = np.array(new_data_train.iloc[val_index]['rtable_pinyin'])
            val_s = sequence.pad_sequences(val_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']
            train_sr = get_sr_respectively(new_data_train.iloc[t_index].copy(), search_result)
            test_sr = get_sr_respectively(new_data.iloc[test_index].copy(), search_result)
            val_sr = get_sr_respectively(new_data_train.iloc[val_index].copy(), search_result)

            train_rec = np.array(new_data_train.iloc[t_index]['rec_score'])
            test_rec = np.array(new_data.iloc[test_index]['rec_score'])
            val_rec = np.array(new_data_train.iloc[val_index]['rec_score'])

            # model = complex_model(m, onlyone=onlyone, softmax=softmax)  # 'lstm' 'rnn' 'gru'
            model = bigru_complex_model_v3()
            # bigru_complex_model  align_complex_model www19_model bigru_complex_model_v2
            # if not fold_num:
            #     print(model.summary())

            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns] + [train_s, train_ws] + train_sr + [train_rec], train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=2,  # 2 1
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns] + [val_s, val_ws] + val_sr + [val_rec], val_label),
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
            test_predict = model.predict([test[c] for c in columns] + [test_s, test_ws] + test_sr + [test_rec],
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
                            with open('{}/linking/log/FP-{}.log'.format(ex_path, MODEL_NAME), 'a+',
                                      encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ltable_name'],
                                                                  new_data.iloc[test_index].iloc[index]['rtable_name']))
                else:
                    if t_label[index] == 1:
                        fn += 1
                        if save_log:
                            with open('{}/linking/log/FN-{}.log'.format(ex_path, MODEL_NAME), 'a+',
                                      encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ltable_name'],
                                                                  new_data.iloc[test_index].iloc[index]['rtable_name']))
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


def our_complex_model_v4(folds, save_log=False):
    # 这版用了recommendation信息 向量进网络的形式
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data_s = extract_data_simple_rec(process_column=columns + ['rec'], ngram=3)
    new_data, ws_data, search_result = extract_data_complex(process_column=columns, sources=SOURCES, ngram=3,
                                                            need_rec_score=False)
    pre_result = list()

    k_fold = StratifiedKFold(n_splits=folds, shuffle=True)
    for fold_num, (train_index, test_index) in enumerate(k_fold.split(new_data, new_data['label'])):
        print('Fold {} of {}\n'.format(fold_num + 1, folds))
        new_data_train = new_data.iloc[train_index]
        new_data_s_train = new_data_s.iloc[train_index]

        val_folder = StratifiedKFold(n_splits=10, shuffle=True)
        for t_index, val_index in val_folder.split(new_data_train, new_data_train['label']):
            # print(t_index, val_index)
            train, test, val = dict(), dict(), dict()
            for c in columns + ['rec']:
                tra = np.array(new_data_s_train.iloc[t_index][c])
                tra = sequence.pad_sequences(tra, maxlen=MAX_SEQ_LENGTH, padding='post')
                tes = np.array(new_data_s.iloc[test_index][c])
                tes = sequence.pad_sequences(tes, maxlen=MAX_SEQ_LENGTH, padding='post')
                va = np.array(new_data_s_train.iloc[val_index][c])
                va = sequence.pad_sequences(va, maxlen=MAX_SEQ_LENGTH, padding='post')
                train[c] = tra
                test[c] = tes
                val[c] = va

            train_ws = np.array(ws_data.iloc[train_index].iloc[t_index]['ltable_pinyin'])
            train_ws = sequence.pad_sequences(train_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_ws = np.array(ws_data.iloc[test_index]['ltable_pinyin'])
            test_ws = sequence.pad_sequences(test_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_ws = np.array(ws_data.iloc[train_index].iloc[val_index]['ltable_pinyin'])
            val_ws = sequence.pad_sequences(val_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_s = np.array(new_data_train.iloc[t_index]['rtable_pinyin'])
            train_s = sequence.pad_sequences(train_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_s = np.array(new_data.iloc[test_index]['rtable_pinyin'])
            test_s = sequence.pad_sequences(test_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_s = np.array(new_data_train.iloc[val_index]['rtable_pinyin'])
            val_s = sequence.pad_sequences(val_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_label = new_data_train.iloc[t_index]['label']
            test_label = new_data.iloc[test_index]['label']
            val_label = new_data_train.iloc[val_index]['label']
            train_sr = get_sr_respectively(new_data_train.iloc[t_index].copy(), search_result)
            test_sr = get_sr_respectively(new_data.iloc[test_index].copy(), search_result)
            val_sr = get_sr_respectively(new_data_train.iloc[val_index].copy(), search_result)

            model = bigru_complex_model_v4()
            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
            model.fit([train[c] for c in columns + ['rec']] + [train_s, train_ws] + train_sr, train_label,
                      batch_size=BATCH_SIZE,
                      epochs=MAX_EPOCHS,
                      verbose=2,  # 2 1
                      # validation_split=0.1,
                      validation_data=([val[c] for c in columns + ['rec']] + [val_s, val_ws] + val_sr, val_label),
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
            test_predict = model.predict([test[c] for c in columns + ['rec']] + [test_s, test_ws] + test_sr,
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
                            with open('{}/linking/log/FP-{}.log'.format(ex_path, MODEL_NAME), 'a+',
                                      encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ltable_name'],
                                                                  new_data.iloc[test_index].iloc[index]['rtable_name']))
                else:
                    if t_label[index] == 1:
                        fn += 1
                        if save_log:
                            with open('{}/linking/log/FN-{}.log'.format(ex_path, MODEL_NAME), 'a+',
                                      encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ltable_name'],
                                                                  new_data.iloc[test_index].iloc[index]['rtable_name']))
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


def our_combine_model_v1(folds, save_log=False):
    # w-s ws-s-sr s-rec 三个模块结合 方法使用complexv2 和simplev5
    columns = ['ltable_pinyin', 'rtable_pinyin']
    new_data_s, rec_result = extract_data_simple_rec_v2(process_column=columns, ngram=5, fuzzy_rec=False)
    new_data, ws_data, search_result = extract_data_complex(process_column=columns, sources=SOURCES, ngram=5,
                                                            need_rec_score=False)
    pre_result = list()

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

            train_ws = np.array(ws_data.iloc[train_index].iloc[t_index]['ltable_pinyin'])
            train_ws = sequence.pad_sequences(train_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_ws = np.array(ws_data.iloc[test_index]['ltable_pinyin'])
            test_ws = sequence.pad_sequences(test_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_ws = np.array(ws_data.iloc[train_index].iloc[val_index]['ltable_pinyin'])
            val_ws = sequence.pad_sequences(val_ws, maxlen=MAX_S_SEQ_LENGTH, padding='post')

            train_s = np.array(new_data_train.iloc[t_index]['rtable_pinyin'])
            train_s = sequence.pad_sequences(train_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            test_s = np.array(new_data.iloc[test_index]['rtable_pinyin'])
            test_s = sequence.pad_sequences(test_s, maxlen=MAX_S_SEQ_LENGTH, padding='post')
            val_s = np.array(new_data_train.iloc[val_index]['rtable_pinyin'])
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
            model_checkpoint_path = '{}/linking/matching/our/model_checkpoint_{}.h5'.format(ex_path, MODEL_NAME)
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
                            with open('{}/linking/log/FP-{}.log'.format(ex_path, MODEL_NAME), 'a+',
                                      encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ltable_name'],
                                                                  new_data.iloc[test_index].iloc[index]['rtable_name']))
                else:
                    if t_label[index] == 1:
                        fn += 1
                        if save_log:
                            with open('{}/linking/log/FN-{}.log'.format(ex_path, MODEL_NAME), 'a+',
                                      encoding='utf-8') as f:
                                f.write('Fold{}\t{}\t{}\n'.format(fold_num + 1,
                                                                  new_data.iloc[test_index].iloc[index]['ltable_name'],
                                                                  new_data.iloc[test_index].iloc[index]['rtable_name']))
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
    if sys.platform == 'win32':
        start_time = start_time.replace(':', ' ')
    _p, _r, times = 0, 0, 10
    for i in range(times):
        temp_p, temp_r = our_simple_model(folds=5, m='gru', bid=True, save_log=True)
        # temp_p, temp_r = our_simple_model_pairwise(folds=5)
        # temp_p, temp_r = our_simple_model_v2(folds=5, save_log=True)
        # temp_p, temp_r = our_simple_model_v3(folds=5, save_log=True)
        # temp_p, temp_r = our_simple_model_v4(folds=5, save_log=True)
        # temp_p, temp_r = our_complex_model(folds=5, onlyone=False, softmax=True, m='gru')
        # temp_p, temp_r = our_complex_model_pairwise(folds=5)
        # temp_p, temp_r = our_complex_model_v2(folds=5, save_log=True)
        # temp_p, temp_r = our_complex_model_v2_pairwise(folds=5)
        # temp_p, temp_r = our_complex_model_v3(folds=5, save_log=True)
        # temp_p, temp_r = our_complex_model_v4(folds=5, save_log=True)
        # temp_p, temp_r = our_combine_model_v1(folds=5, save_log=False)

        _p += temp_p
        _r += temp_r
    _p /= times
    _r /= times
    _f = 2 * _p * _r / (_p + _r)
    print('P: {}\tR: {}\tF: {}\n'.format(_p, _r, _f))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))