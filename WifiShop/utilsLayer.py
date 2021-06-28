from keras.layers import *
from keras.activations import relu
import tensorflow as tf


def get_softmax(tensor):
    return K.softmax(tensor)


def get_softmax_row(tensor):
    return K.softmax(tensor, axis=1)


def get_mean(tensor):
    return K.mean(tensor)


def get_abs(tensor):
    return K.abs(tensor)


def get_l2(tensor):
    return K.l2_normalize(tensor, axis=-1)


def get_std(tensor, axis):
    return K.std(tensor, axis=axis)


def get_repeat(tensor, nums):
    return K.repeat(tensor, n=nums)


def get_max(tensor, axis):
    return K.max(tensor, axis=axis)


def get_transpose(tensor):
    return tf.transpose(tensor, [0, 2, 1])


class NegLogPairLossLayer(Layer):
    def __init__(self, **kwargs):
        super(NegLogPairLossLayer, self).__init__(** kwargs)

    def neg_pair_pair_loss(self, labels, y_true, y_false):
        sub = y_true - y_false
        sub = K.sigmoid(sub)
        print(K.int_shape(sub))
        return K.mean(K.binary_crossentropy(labels, sub), axis=-1)

    def build(self, input_shape):
        self.supports_masking = True
        super(NegLogPairLossLayer, self).build(input_shape)

    def call(self, inputs):
        labels, y_true, y_false = inputs
        loss = self.neg_pair_pair_loss(labels, y_true, y_false)
        self.add_loss(loss, inputs=inputs)
        return y_false


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


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, main_tensor):
        x = K.stack(inputs, axis=1)
        x_transpose = K.permute_dimensions(x, (0, 2, 1))
        weights = K.dot(main_tensor, x_transpose)
        weights = K.softmax(weights)
        outputs = K.dot(weights, x)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class SoftmaxAttLayer(Layer):
    def __init__(self, main_tensor, **kwargs):
        self.main_tensor = main_tensor
        self.softlambda = Lambda(get_softmax)
        super(SoftmaxAttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        super(SoftmaxAttLayer, self).build(input_shape)

    def call(self, inputs):
        weights = dot([inputs, self.main_tensor], axes=[2, 1])
        weights = self.softlambda(weights)
        outputs = dot([weights, inputs], axes=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class AlignLayer(Layer):
    def __init__(self, context_tensor, **kwargs):
        self.context_tensor = context_tensor
        self.softlambda = Lambda(get_softmax)
        self.meanlambda = Lambda(get_mean)
        self.highway = HighwayLayer()
        # self.timed = TimeDistributed(Dense(1))
        # self.relu = ThresholdedReLU(0.5)
        self.l2lambda = Lambda(get_l2)
        super(AlignLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[2] * 2, 1),
                                 initializer='uniform',
                                 trainable=True)
        # self.timed = TimeDistributed(Dense(input_shape[2] * 2))
        super(AlignLayer, self).build(input_shape)

    def call(self, inputs):
        weights = dot([self.context_tensor, inputs], axes=[2, 2])
        # print(K.int_shape(weights))
        weights = self.softlambda(weights)
        # print(K.int_shape(weights))
        outputs = dot([weights, self.context_tensor], axes=1)
        # print(K.int_shape(outputs))
        # print('input', K.int_shape(inputs))
        outputs = concatenate([inputs, outputs], axis=-1)
        # print(K.int_shape(outputs))
        # outputs = self.timed(outputs)
        # outputs = self.highway(outputs)

        # outputs = self.meanlambda(outputs)
        # print(K.int_shape(self.W))
        w = K.dot(outputs, self.W)
        # print(K.int_shape(w))
        w = K.squeeze(w, axis=-1)
        # print(K.int_shape(w))

        # w = self.relu(w)
        w = relu(w, alpha=0.01, threshold=0.3)
        # w = self.softlambda(w)
        w = self.l2lambda(w)

        outputs = dot([w, outputs], axes=1)
        # m_context = self.meanlambda(self.context_tensor)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0],  input_shape[2] * 2


class AlignSubLayer(Layer):
    def __init__(self, **kwargs):
        self.softlambda = Lambda(get_softmax)
        # self.meanlambda = Lambda(get_mean)
        # self.highway = HighwayLayer()
        # self.timed = TimeDistributed(Dense(1))
        # self.relu = ThresholdedReLU(0.5)
        # self.l2lambda = Lambda(get_l2)
        self.abslambda = Lambda(get_abs)
        super(AlignSubLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1][2], 1),
                                 initializer='uniform',
                                 trainable=True)
        # self.timed = TimeDistributed(Dense(input_shape[2] * 2))
        super(AlignSubLayer, self).build(input_shape)

    def call(self, inputs):
        context, main = inputs
        weights = dot([context, main], axes=[2, 2])
        # print(K.int_shape(weights))
        weights = self.softlambda(weights)
        # print(K.int_shape(weights))
        outputs = dot([weights, context], axes=1)
        # print(K.int_shape(outputs))
        # print('input', K.int_shape(inputs))
        outputs = subtract([main, outputs])
        # print(K.int_shape(outputs))
        # outputs = self.timed(outputs)
        # outputs = self.highway(outputs)

        # outputs = self.abslambda(outputs)  # 可选 sub加abs

        # outputs = self.meanlambda(outputs)
        # print(K.int_shape(self.W))
        w = K.dot(outputs, self.W)
        # print(K.int_shape(w))
        w = K.squeeze(w, axis=-1)
        # print(K.int_shape(w))

        # w = self.relu(w)
        # w = relu(w, alpha=0.01, threshold=0.3)
        w = self.softlambda(w)
        # w = self.l2lambda(w)

        outputs = dot([w, outputs], axes=1)


        # m_context = self.meanlambda(self.context_tensor)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1][0],  input_shape[1][2]


class AlignOnlySubLayer(Layer):
    def __init__(self, **kwargs):
        self.softlambda = Lambda(get_softmax)
        super(AlignOnlySubLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        super(AlignOnlySubLayer, self).build(input_shape)

    def call(self, inputs):
        context, main = inputs
        weights = dot([context, main], axes=[2, 2])
        weights = self.softlambda(weights)
        outputs = dot([weights, context], axes=1)
        outputs = subtract([main, outputs])
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1]


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
        self.rsoftlambda = Lambda(get_softmax_row)
        self.softlambda = Lambda(get_softmax)
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
        self.rsoftlambda = Lambda(get_softmax_row)
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
        self.softlambda = Lambda(get_softmax)
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


class BiAlignLayer(Layer):
    def __init__(self, agg_nn, **kwargs):
        self.rsoftlambda = Lambda(get_softmax_row)
        self.softlambda = Lambda(get_softmax)
        self.agg_nn = agg_nn
        super(BiAlignLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        super(BiAlignLayer, self).build(input_shape)

    def call(self, inputs):
        i, j = inputs
        weight = dot([i, j], axes=2)
        weight_j = self.rsoftlambda(weight)
        weight_i = self.softlambda(weight)
        weighted_i = dot([weight_i, i], axes=1)
        weighted_j = dot([weight_j, j], axes=[2, 1])
        output_i = subtract([i, weighted_j])
        output_j = subtract([j, weighted_i])
        output_i, output_j = self.agg_nn(output_i), self.agg_nn(output_j)
        con = K.stack([output_i, output_j], axis=1)
        output = K.mean(con, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.nn_dim * 2


class BiAlignAggLayer(Layer):
    def __init__(self, nn_dim, agg_nn, **kwargs):
        self.nn_dim = nn_dim
        self.agg_nn = agg_nn
        self.rsoftlambda = Lambda(get_softmax_row)
        self.softlambda = Lambda(get_softmax)
        self.abslambda = Lambda(get_abs)
        super(BiAlignAggLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        super(BiAlignAggLayer, self).build(input_shape)

    def call(self, inputs):
        i, j = inputs
        weight = dot([i, j], axes=2)
        weight_j = self.rsoftlambda(weight)
        weight_i = self.softlambda(weight)
        weighted_i = dot([weight_i, i], axes=1)
        weighted_j = dot([weight_j, j], axes=[2, 1])
        output_i = subtract([i, weighted_j])
        output_j = subtract([j, weighted_i])
        output_i, output_j = self.abslambda(output_i), self.abslambda(output_j)  # 可选 sub加abs
        output_i, output_j = self.agg_nn(output_i), self.agg_nn(output_j)
        con = K.stack([output_i, output_j], axis=1)
        output = K.mean(con, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.nn_dim * 2


class S2SMLayer(Layer):  # CIKM'19 seq2seqmatcher main layer
    def __init__(self, filter_size=3, filter_num=100, **kwargs):
        self.std_1_lambda = Lambda(get_std, arguments={'axis': 1})
        self.std_2_lambda = Lambda(get_std, arguments={'axis': 2})
        self.repeat_c_lambda = None
        self.repeat_m_lambda = None
        self.max_1_lambda = Lambda(get_max, arguments={'axis': 1})
        self.cnn = Conv1D(filter_num, kernel_size=filter_size, activation='relu')
        self.trans = Lambda(get_transpose)
        self.abs = Lambda(get_abs)
        self.kmax = KmaxLayer(k=3)
        self.outputd2 = filter_num * 2
        super(S2SMLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        super(S2SMLayer, self).build(input_shape)

    def call(self, inputs):
        context, main = inputs
        weights = dot([context, main], axes=[2, 2])
        print(K.int_shape(weights))

        # a_top, a_top_idx = tf.nn.top_k(weights, 3, sorted=False)
        # kth = K.min(a_top, axis=2, keepdims=True)
        # top2 = K.greater_equal(weights, kth)
        # mask = K.cast(top2, dtype=tf.float32)
        # v = multiply([weights, mask])
        # sum = tf.reciprocal(K.sum(v, axis=2))  # 取倒数
        # sum = K.repeat(sum, n=K.int_shape(weights)[2])
        # print(K.int_shape(sum))
        # sum = self.trans(sum)
        # weights_kmax_c = multiply([v, sum])
        #
        # t_weights = self.trans(weights)
        # a_top, a_top_idx = tf.nn.top_k(t_weights, 3, sorted=False)
        # kth = K.min(a_top, axis=2, keepdims=True)
        # top2 = K.greater_equal(t_weights, kth)
        # mask = K.cast(top2, dtype=tf.float32)
        # v = multiply([t_weights, mask])
        # sum = tf.reciprocal(K.sum(v, axis=2))  # 取倒数
        # sum = K.repeat(sum, n=K.int_shape(t_weights)[2])
        # print(K.int_shape(sum))
        # sum = self.trans(sum)
        # weights_kmax_m = multiply([v, sum])


        weights_kmax_c = self.kmax(weights)
        # print(K.int_shape(weights_kmax_c))
        weights_kmax_m = self.kmax(self.trans(weights))

        std_c = self.std_2_lambda(weights)
        # print(K.int_shape(std_c))
        std_m = self.std_1_lambda(weights)
        self.repeat_c_lambda = Lambda(get_repeat, arguments={'nums': K.int_shape(context)[2]})
        self.repeat_m_lambda = Lambda(get_repeat, arguments={'nums': K.int_shape(main)[2]})
        std_c = self.repeat_c_lambda(std_c)
        std_c = self.trans(std_c)
        # print(K.int_shape(std_c))
        std_m = self.repeat_m_lambda(std_m)
        std_m = self.trans(std_m)

        # std_c = K.std(weights, axis=2)  # ,keepdims=True to see the std function
        # std_m = K.std(weights, axis=1)
        # std_c = K.repeat(std_c, n=K.int_shape(context)[1])
        # std_m = K.repeat(std_m, n=K.int_shape(main)[1])

        att_merge_c = dot([weights_kmax_c, context], axes=1) # weights  weights_kmax_c
        # print(K.int_shape(att_merge_c))
        outputs_m = self.abs(subtract([main, att_merge_c]))
        # print(K.int_shape(outputs_m))
        att_merge_m = dot([weights_kmax_m, main], axes=[2, 1]) # weights [2, 1]   weights_kmax_m 1
        outputs_c = self.abs(subtract([context, att_merge_m]))

        outputs_m = multiply([std_m, outputs_m])
        # print(K.int_shape(outputs_m))
        outputs_c = multiply([std_c, outputs_c])

        cnn_m = self.cnn(outputs_m)
        # print(K.int_shape(cnn_m))
        # maxpooling_m = K.max(cnn_m, axis=1)
        maxpooling_m = self.max_1_lambda(cnn_m)
        # print(K.int_shape(maxpooling_m))

        cnn_c = self.cnn(outputs_c)
        # maxpooling_c = K.max(cnn_c, axis=1)
        maxpooling_c = self.max_1_lambda(cnn_c)

        output = concatenate([maxpooling_c, maxpooling_m])

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.outputd2


class KmaxLayer(Layer):  # Kmax layer for CIKM'19 seq2seqmatcher
    def __init__(self, k=3, **kwargs):
        self.k = k
        super(KmaxLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        self.trans = Lambda(get_transpose)
        super(KmaxLayer, self).build(input_shape)

    def call(self, inputs):
        a_top, a_top_idx = tf.nn.top_k(inputs, self.k, sorted=False)
        # print(K.int_shape(a_top))
        # 获取第k大值
        kth = K.min(a_top, axis=2, keepdims=True)
        # print(K.int_shape(kth))
        # 大于第k大值的为true，小于为false
        top2 = K.greater_equal(inputs, kth)
        # print(K.int_shape(top2))
        # 映射为掩码，大于第k大值的为1，小于为0
        mask = K.cast(top2, dtype=tf.float32)
        # print(K.int_shape(mask))
        # 只保留张量top k的值不变，其余值变为0
        v = multiply([inputs, mask])
        # print(K.int_shape(v))
        # 不为零的数进行归一化操作，即每个数除以该向量的和
        sum = tf.reciprocal(K.sum(v, axis=2))  # 取倒数
        # print(K.int_shape(sum))
        sum = K.repeat(sum, n=K.int_shape(inputs)[2])
        # print(K.int_shape(sum))
        sum = self.trans(sum)
        norms = multiply([v, sum])

        return norms
