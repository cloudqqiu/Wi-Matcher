from keras import backend as K
from keras.layers import *

import models.common.utils as utils


class SoftmaxAttLayer(Layer):
    def __init__(self, main_tensor, **kwargs):
        self.main_tensor = main_tensor
        self.softlambda = Lambda(utils.get_softmax)
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


class AlignSubLayer(Layer):
    def __init__(self, **kwargs):
        self.softlambda = Lambda(utils.get_softmax)
        # self.meanlambda = Lambda(get_mean)
        # self.highway = HighwayLayer()
        # self.timed = TimeDistributed(Dense(1))
        # self.relu = ThresholdedReLU(0.5)
        # self.l2lambda = Lambda(get_l2)
        self.abslambda = Lambda(utils.get_abs)
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


class BiAlignAggLayer(Layer):
    def __init__(self, nn_dim, agg_nn, **kwargs):
        self.nn_dim = nn_dim
        self.agg_nn = agg_nn
        self.rsoftlambda = Lambda(utils.get_softmax_row)
        self.softlambda = Lambda(utils.get_softmax)
        self.abslambda = Lambda(utils.get_abs)
        super(BiAlignAggLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.supports_masking = True
        super(BiAlignAggLayer, self).build(input_shape)

    def call(self, inputs):
        i, j = inputs[0], inputs[1]
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

