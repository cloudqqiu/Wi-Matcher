import keras.optimizers
from keras.layers import *
from keras.models import Model
from layers import *

import models.common.utils as utils


def wi_matcher_base_qr(max_seq_len, max_qr_num,
                       qr_char_len, qr_gram_len, qr_embed_mat,
                       nn_dim, num_dense, num_sec_dense):
    ssid_input = Input(shape=(max_seq_len,), dtype='int32', name='ssid_input')
    venue_input = Input(shape=(max_seq_len,),
                        dtype='int32', name='venue_input')

    qr_data_input = Input(shape=(max_qr_num, max_seq_len,),
                          dtype='int32', name='qr_data_input')

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

    qr_bi_gru = Bidirectional(
        GRU(units=nn_dim, return_sequences=True), merge_mode='concat')
    qr_h_data = TimeDistributed(qr_bi_gru)(qr_data_embedding)

    qr_h_venue = qr_bi_gru(base_qr_venue_embedding)

    qr_att_bi_gru = Bidirectional(GRU(units=nn_dim), merge_mode='concat')
    qr_bi_sa_att = BiAlignAggLayer(nn_dim=nn_dim, agg_nn=qr_att_bi_gru)

    qr_h_data_list = Lambda(lambda h_data: [qr_bi_sa_att([qr_h_venue, h_data[:, i, :, :]]) for i in range(max_qr_num)])(
        qr_h_data)

    qr_h_data = Lambda(utils.get_stack)(qr_h_data_list)

    qr_p_venue = qr_att_bi_gru(qr_h_venue)
    qr_softmax_att = SoftmaxAttLayer(main_tensor=qr_p_venue)

    r_qr = qr_softmax_att(qr_h_data)

    r_concat = concatenate([r_base, r_qr])

    r_concat = Dropout(rate=0.4)(r_concat)
    score = Dense(num_dense * 2, activation='relu')(r_concat)
    score = Dense(num_sec_dense, activation='relu')(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(
        inputs=[
            ssid_input, venue_input, qr_data_input
        ], outputs=score)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam', metrics=['acc'])
    model.summary()

    return model


def wi_matcher_base_sr(max_seq_len, max_sr_seq_len, max_sr_num,
                       sr_char_len, sr_gram_len, base_char_len, base_gram_len,
                       sr_embed_mat, base_embed_mat,
                       nn_dim, num_dense, num_sec_dense):
    ssid_input = Input(shape=(max_seq_len,), dtype='int32', name='ssid_input')
    venue_input = Input(shape=(max_seq_len,),
                        dtype='int32', name='venue_input')
    sr_venue_input = Input(shape=(max_sr_seq_len,),
                           dtype='int32', name='sr_venue_input')
    sr_sv_input = Input(shape=(max_sr_seq_len,),
                        dtype='int32', name='sr_sv_input')
    sr_data_input = Input(shape=(max_sr_num, max_sr_seq_len,),
                          dtype='int32', name='sr_data_input')

    base_qr_embed_layer = Embedding(output_dim=base_char_len, input_dim=base_gram_len + 1,
                                    weights=[base_embed_mat],
                                    mask_zero=True, trainable=False)

    # Base module
    base_ssid_embedding = base_qr_embed_layer(ssid_input)
    base_qr_venue_embedding = base_qr_embed_layer(venue_input)

    base_bi_gru = Bidirectional(GRU(units=nn_dim), merge_mode='concat')
    base_h_ssid = base_bi_gru(base_ssid_embedding)
    base_h_venue = base_bi_gru(base_qr_venue_embedding)
    r_base = subtract([base_h_ssid, base_h_venue])

    # Search Result Module
    sr_embed_layer = Embedding(output_dim=sr_char_len, input_dim=sr_gram_len + 1,
                               weights=[sr_embed_mat],
                               mask_zero=True, trainable=False)
    sr_venue_embedding = sr_embed_layer(sr_venue_input)
    sr_sv_embedding = sr_embed_layer(sr_sv_input)
    sr_data_embedding = sr_embed_layer(sr_data_input)

    sr_bi_gru = Bidirectional(
        GRU(units=nn_dim, return_sequences=True), merge_mode='concat')
    sr_h_venue = sr_bi_gru(sr_venue_embedding)
    sr_h_sv = sr_bi_gru(sr_sv_embedding)
    sr_h_data = TimeDistributed(sr_bi_gru)(sr_data_embedding)

    sr_h_sv = Lambda(lambda a: a[:, -1, :])(sr_h_sv)

    sr_sq_att = AlignSubLayer()

    sr_h_data_list = Lambda(lambda h_data: [sr_sq_att([sr_h_venue, h_data[:, i, :, :]]) for i in range(max_sr_num)])(
        sr_h_data)

    sr_h_data = Lambda(utils.get_stack)(sr_h_data_list)

    sr_softmax_att = SoftmaxAttLayer(main_tensor=sr_h_sv)
    r_sr = sr_softmax_att(sr_h_data)

    r_concat = concatenate([r_base, r_sr])

    r_concat = Dropout(rate=0.4)(r_concat)
    score = Dense(num_dense * 2, activation='relu')(r_concat)
    score = Dense(num_sec_dense, activation='relu')(score)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(
        inputs=[
            ssid_input, venue_input, sr_venue_input, sr_sv_input, sr_data_input
        ], outputs=score)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam', metrics=['acc'])
    model.summary()

    return model


def hi_em_var(max_seq_len, char_len_s, gram_len_s, embedding_matrix_s, nn_dim, num_dense):
    # Structure is similar to WiMatcher base module
    wifi_input = Input(shape=(max_seq_len,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(max_seq_len,), dtype='int32', name='shop_input')

    embedding_layer = Embedding(output_dim=char_len_s, input_dim=gram_len_s + 1, input_length=max_seq_len,
                                weights=[embedding_matrix_s], mask_zero=True, trainable=False)

    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    r = Bidirectional(
        GRU(units=nn_dim, return_sequences=True), merge_mode='concat')
    r2 = Bidirectional(
        GRU(units=nn_dim, return_sequences=True), merge_mode='concat')

    w, s = r2(r(w_e)), r2(r(s_e))

    inter = InterAttLayer()
    w_inter, s_inter = inter([w, s])
    sub_i = subtract([w, w_inter])
    mul_i = multiply([w, w_inter])
    w_inter = concatenate([sub_i, mul_i], axis=-1)
    sub_j = subtract([s, s_inter])
    mul_j = multiply([s, s_inter])
    s_inter = concatenate([sub_j, mul_j], axis=-1)

    intra_attention = TimeDistributed(Dense(1))
    squ, soft = Lambda(utils.squeeze), Lambda(utils.get_softmax)
    beta_w = intra_attention(w_inter)
    beta_s = intra_attention(s_inter)
    beta_w, beta_s = soft(beta_w), soft(beta_s)
    print(K.int_shape(beta_w))
    w_intra = dot([beta_w, w_inter], axes=1)
    s_intra = dot([beta_s, s_inter], axes=1)
    w_intra, s_intra = squ(w_intra), squ(s_intra)

    sim_w_s = concatenate([w_intra, s_intra])
    print(K.int_shape(sim_w_s))

    score = Dense(num_dense * 2, activation='relu')(sim_w_s)  # NUM_DENSE * 2
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def deepmatcher_var(max_seq_len, char_len_s, gram_len_s, embedding_matrix_s, nn_dim, num_dense, lr=0.001):
    wifi_input = Input(shape=(max_seq_len,), dtype='int32', name='wifi_input')
    shop_input = Input(shape=(max_seq_len,), dtype='int32', name='shop_input')

    embedding_layer = Embedding(output_dim=char_len_s, input_dim=gram_len_s + 1, input_length=max_seq_len,
                                weights=[embedding_matrix_s],
                                mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)
    rnn1, rnn2 = Bidirectional(GRU(units=nn_dim, return_sequences=True), merge_mode='concat'), \
        Bidirectional(GRU(units=nn_dim), merge_mode='concat')

    h1, h2 = HighwayLayer(), HighwayLayer()
    w_a, s_a = h1(h2(w_e)), h1(h2(s_e))

    w_rnn1, s_rnn1 = rnn1(w_e), rnn1(s_e)
    w_rnn2, s_rnn2 = rnn2(w_e), rnn2(s_e)
    d3, d4 = Dense(
        units=nn_dim * 2, activation='relu'), Dense(units=nn_dim * 2, activation='tanh')
    d5, d6 = Dense(units=1, activation='relu'), Dense(
        units=nn_dim * 2, activation='tanh')
    h3, h4, h5, h6 = HighwayLayer(), HighwayLayer(), HighwayLayer(), HighwayLayer()

    softalign_w_s = dot(
        inputs=[w_a, Lambda(utils.get_transpose)(s_a)], axes=(2, 1))
    softalign_w_s = Lambda(utils.get_softmax)(softalign_w_s)

    s_a_avg = dot(inputs=[softalign_w_s, s_rnn1], axes=1)

    w_comparison = concatenate(inputs=[w_rnn1, s_a_avg], axis=-1)

    w_comparison = h3(h4(w_comparison))
    s_rnn2_rp = RepeatVector(max_seq_len)(s_rnn2)

    w_comparison_weight = concatenate(
        inputs=[w_comparison, s_rnn2_rp], axis=-1)
    w_comparison_weight = d5(h5(w_comparison_weight))

    w_comparison_weight = Lambda(
        lambda a: K.squeeze(a, axis=-1))(w_comparison_weight)
    w_comparison_weight = Lambda(utils.get_softmax)(w_comparison_weight)

    w_aggregation = dot(inputs=[w_comparison_weight, w_comparison], axes=1)

    softalign_s_w = dot(
        inputs=[s_a, Lambda(utils.get_transpose)(w_a)], axes=(2, 1))
    softalign_s_w = Lambda(utils.get_softmax)(softalign_s_w)
    w_a_avg = dot(inputs=[softalign_s_w, w_rnn1], axes=1)
    s_comparison = concatenate(inputs=[s_rnn1, w_a_avg], axis=-1)

    s_comparison = h3(h4(s_comparison))
    w_rnn2_rp = RepeatVector(max_seq_len)(w_rnn2)
    s_comparison_weight = concatenate(
        inputs=[s_comparison, w_rnn2_rp], axis=-1)
    s_comparison_weight = d5(h5(s_comparison_weight))
    s_comparison_weight = Lambda(
        lambda a: K.squeeze(a, axis=-1))(s_comparison_weight)
    s_comparison_weight = Lambda(utils.get_softmax)(s_comparison_weight)
    s_aggregation = dot(inputs=[s_comparison_weight, s_comparison], axes=1)

    sim_w_s = subtract(inputs=[w_aggregation, s_aggregation])

    sim_w_s = Lambda(lambda a: K.abs(a))(sim_w_s)

    score = Dense(num_dense, activation='relu')(sim_w_s)
    score = Dense(1, activation='sigmoid')(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)
    # model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(lr=0.001), metrics=['acc'])

    model.summary()
    return model
