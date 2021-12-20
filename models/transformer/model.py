from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def transformer_model(max_seq_len, char_len_s, gram_len_s, embedding_matrix_s, nn_dim, num_dense, transformer_embed=256, d_ff=512, lr=0.002, num_layers=6, num_heads=8):
    wifi_input = layers.Input(shape=(max_seq_len,), dtype='int32', name='wifi_input')
    shop_input = layers.Input(shape=(max_seq_len,), dtype='int32', name='shop_input')

    embedding_layer = layers.Embedding(output_dim=char_len_s, input_dim=gram_len_s + 1, input_length=max_seq_len,
                                weights=[embedding_matrix_s], mask_zero=True, trainable=False)
    w_e, s_e = embedding_layer(wifi_input), embedding_layer(shop_input)

    # r = Bidirectional(GRU(units=nn_dim), merge_mode='concat')
    dim_trans_layer = layers.Dense(transformer_embed)
    w_e, s_e = dim_trans_layer(w_e), dim_trans_layer(s_e)

    r = [TransformerBlock(transformer_embed, num_heads, d_ff) for _ in range(num_layers)]

    for i in range(num_layers):
      w_e, s_e = r[i](w_e), r[i](s_e)

    pooling_layer = layers.GlobalAveragePooling1D()
    w = pooling_layer(w_e)
    s = pooling_layer(s_e)
    sim_w_s = layers.Subtract()([w, s])

    score = layers.Dense(num_dense, activation="relu")(sim_w_s)
    score = layers.Dense(1, activation="sigmoid")(score)

    model = Model(inputs=[wifi_input, shop_input], outputs=score)

    if lr!= None:
        nadam = optimizers.Nadam(learning_rate=lr)
    else:
        nadam = optimizers.Nadam()
    model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['acc'])
    model.summary()
    return model