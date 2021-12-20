
from layers import *
from keras.models import Model
from keras.layers import *
from keras import backend as K, optimizers
import models.common.utils as utils
import tensorflow as tf


def wi_matcher_base(max_seq_len, char_len_s, gram_len_s, embedding_matrix_s, nn_dim, num_dense, lr=0.002):
    wifi_input = Input(shape=(max_seq_len,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(max_seq_len,), dtype='int32', name='shop_input')

    embedding_layer = Embedding(output_dim=char_len_s, input_dim=gram_len_s + 1, input_length=max_seq_len,
                                weights=[embedding_matrix_s], mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    r = Bidirectional(GRU(units=nn_dim), merge_mode='concat')
    w, s = r(w_e), r(s_e)
    sim_w_s = Subtract()([w, s])

    score = Dense(num_dense, activation='relu')(sim_w_s)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)

    if lr!= None:
        nadam = optimizers.nadam(lr=lr)
    else:
        nadam = optimizers.nadam()
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['acc'])
    model.summary()
    return model
    # parallel_model  = multi_gpu_model(model, gpus=2)
    # parallel_model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['acc'])
    # parallel_model.summary()
    # return parallel_model


def wi_matcher_whole(max_seq_len, max_s_seq_len,
                     char_len_c, gram_len_c, char_len_s, gram_len_s,
                     embedding_matrix_c, embedding_matrix_s,
                     nn_dim, num_dense, num_sec_dense):
    wifi_input = Input(shape=(max_seq_len,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(max_seq_len,), dtype='int32', name='shop_input')
    s_input = Input(shape=(max_s_seq_len,), dtype='int32', name='s_input')
    ws_input = Input(shape=(max_s_seq_len,), dtype='int32', name='ws_input')

    sr_input_0 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_0')
    sr_input_1 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_1')
    sr_input_2 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_2')
    sr_input_3 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_3')
    sr_input_4 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_4')
    sr_input_5 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_5')
    sr_input_6 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_6')
    # sr_input_7 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_7')
    # sr_input_8 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_8')
    # sr_input_9 = Input(shape=(max_s_seq_len,), dtype='int32', name='sr_input_9')

    rec_input_0 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_0')
    rec_input_1 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_1')
    rec_input_2 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_2')
    # rec_input_3 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_3')
    # rec_input_4 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_4')
    # rec_input_5 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_5')
    # rec_input_6 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_6')
    # rec_input_7 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_7')
    # rec_input_8 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_8')
    # rec_input_9 = Input(shape=(max_seq_len,), dtype='int32', name='rec_input_9')

    embedding_layer = Embedding(output_dim=char_len_c, input_dim=gram_len_c + 1,
                                weights=[embedding_matrix_c],
                                mask_zero=True, trainable=False)
    embedding_layer_s = Embedding(output_dim=char_len_s, input_dim=gram_len_s + 1,
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

    rs = Bidirectional(GRU(units=nn_dim), merge_mode='concat')  # w-s模块
    wifi, shop = rs(wifi_e), rs(shop_e)
    sim_wifi_shop = subtract([wifi, shop])
    print(K.int_shape(sim_wifi_shop))

    bigru_r = Bidirectional(GRU(units=nn_dim, return_sequences=True), merge_mode='concat')
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

    bigruagg = Bidirectional(GRU(units=nn_dim), merge_mode='concat')
    bialignagg = BiAlignAggLayer(nn_dim=nn_dim, agg_nn=bigruagg)
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

    r = Bidirectional(GRU(units=nn_dim, return_sequences=True), merge_mode='concat')  # ws-s-sr模块
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
    ts = Lambda(utils.get_stack)([sr_0, sr_1, sr_2, sr_3, sr_4, sr_5, sr_6])  # , sr_7, sr_8, sr_9])
    # print(K.int_shape(ts))
    smatt = SoftmaxAttLayer(main_tensor=ws)
    sr = smatt(ts)

    sim_con = concatenate([sim_wifi_shop, sim_s_rec, sr])  # 连接三模型
    print(K.int_shape(sim_con))

    sim_con = Dropout(rate=0.4)(sim_con)  # 试把dropout放到后面

    # sim_con = Dropout(0.5)(sim_con)
    score = Dense(num_dense * 2, activation='relu')(sim_con)  # num_dense * 2
    # score = Dense(num_dense, activation='relu')(score)
    score = Dense(num_sec_dense, activation='relu')(score)
    # score = Dropout(0.5)(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input, s_input, ws_input, sr_input_0, sr_input_1, sr_input_2, sr_input_3,
                          sr_input_4, sr_input_5, sr_input_6,  # sr_input_7, sr_input_8, sr_input_9,
                          rec_input_0, rec_input_1, rec_input_2],
                  # , rec_input_3, rec_input_4, rec_input_5, rec_input_6,
                  # rec_input_7, rec_input_8, rec_input_9],
                  outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])  # 'binary_crossentropy'
    model.summary()
    return model


def wi_matcher_whole_new(max_seq_len, max_sr_seq_len, max_qr_num, max_sr_num,
                         sr_char_len, sr_gram_len, qr_char_len, qr_gram_len,
                         sr_embed_mat, qr_embed_mat,
                         nn_dim, num_dense, num_sec_dense):
    ssid_input = Input(shape=(max_seq_len,), dtype='int32', name='ssid_input')
    venue_input = Input(shape=(max_seq_len,), dtype='int32', name='venue_input')
    sr_venue_input = Input(shape=(max_sr_seq_len,), dtype='int32', name='sr_venue_input')
    sr_sv_input = Input(shape=(max_sr_seq_len,), dtype='int32', name='sr_sv_input')

    sr_data_input = Input(shape=(max_sr_num, max_sr_seq_len,), dtype='int32', name='sr_data_input')
    qr_data_input = Input(shape=(max_qr_num, max_seq_len,), dtype='int32', name='qr_data_input')

    base_qr_embed_layer = Embedding(output_dim=qr_char_len, input_dim=qr_gram_len + 1,
                                    weights=[qr_embed_mat],
                                    mask_zero=True, trainable=False)

    # Base module
    base_ssid_embedding = base_qr_embed_layer(ssid_input)
    base_qr_venue_embedding = base_qr_embed_layer(venue_input)

    base_bi_gru = Bidirectional(GRU(units=nn_dim), merge_mode='concat')
    base_h_ssid = base_bi_gru(base_ssid_embedding)
    base_h_venue = base_bi_gru(base_qr_venue_embedding)
    r_base = subtract([base_h_ssid, base_h_venue])

    # Query Recommendation Module
    qr_data_embedding = base_qr_embed_layer(qr_data_input)

    qr_bi_gru = Bidirectional(GRU(units=nn_dim, return_sequences=True), merge_mode='concat')
    qr_h_data = TimeDistributed(qr_bi_gru)(qr_data_embedding)

    qr_h_venue = qr_bi_gru(base_qr_venue_embedding)

    qr_att_bi_gru = Bidirectional(GRU(units=nn_dim), merge_mode='concat')
    qr_bi_sa_att = BiAlignAggLayer(nn_dim=nn_dim, agg_nn=qr_att_bi_gru)

    qr_h_data_list = Lambda(lambda h_data: [qr_bi_sa_att([qr_h_venue, h_data[:, i, :, :]]) for i in range(max_qr_num)])(qr_h_data)

    qr_h_data = Lambda(utils.get_stack)(qr_h_data_list)

    qr_p_venue = qr_att_bi_gru(qr_h_venue)
    qr_softmax_att = SoftmaxAttLayer(main_tensor=qr_p_venue)

    r_qr = qr_softmax_att(qr_h_data)

    # Search Result Module
    sr_embed_layer = Embedding(output_dim=sr_char_len, input_dim=sr_gram_len + 1,
                               weights=[sr_embed_mat],
                               mask_zero=True, trainable=False)
    sr_venue_embedding = sr_embed_layer(sr_venue_input)
    sr_sv_embedding = sr_embed_layer(sr_sv_input)
    sr_data_embedding = sr_embed_layer(sr_data_input)

    sr_bi_gru = Bidirectional(GRU(units=nn_dim, return_sequences=True), merge_mode='concat')
    sr_h_venue = sr_bi_gru(sr_venue_embedding)
    sr_h_sv = sr_bi_gru(sr_sv_embedding)
    sr_h_data = TimeDistributed(sr_bi_gru)(sr_data_embedding)

    sr_h_sv = Lambda(lambda a: a[:, -1, :])(sr_h_sv)

    sr_sq_att = AlignSubLayer()

    sr_h_data_list = Lambda(lambda h_data: [sr_sq_att([sr_h_venue, h_data[:, i, :, :]]) for i in range(max_sr_num)])(sr_h_data)

    sr_h_data = Lambda(utils.get_stack)(sr_h_data_list)

    sr_softmax_att = SoftmaxAttLayer(main_tensor=sr_h_sv)
    r_sr = sr_softmax_att(sr_h_data)

    r_concat = concatenate([r_base, r_qr, r_sr])

    r_concat = Dropout(rate=0.4)(r_concat)
    score = Dense(num_dense * 2, activation='relu')(r_concat)
    score = Dense(num_sec_dense, activation='relu')(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(
        inputs=[
            ssid_input, venue_input, sr_venue_input, sr_sv_input,
            sr_data_input, qr_data_input
        ], outputs=score)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
    model.summary()

    return model
