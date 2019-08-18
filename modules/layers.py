# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn

from modules.inits import *
from utils.tools import camera_trans, camera_trans_inv, reduce_var, reduce_std

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=True, gcn_block_id=1,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if gcn_block_id == 1:
            self.support = placeholders['support1']
        elif gcn_block_id == 2:
            self.support = placeholders['support2']
        elif gcn_block_id == 3:
            self.support = placeholders['support3']

        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']
        return self.act(output)


class GraphPooling(Layer):
    def __init__(self, placeholders, pool_id=1, **kwargs):
        super(GraphPooling, self).__init__(**kwargs)
        self.pool_idx = placeholders['pool_idx'][pool_id - 1]

    def _call(self, inputs):
        X = inputs
        add_feat = (1 / 2.0) * tf.reduce_sum(tf.gather(X, self.pool_idx), 1)
        outputs = tf.concat([X, add_feat], 0)
        return outputs


class GraphProjection(Layer):
    def __init__(self, placeholders, **kwargs):
        super(GraphProjection, self).__init__(**kwargs)

        self.img_feat = placeholders['img_feat']
        self.camera = placeholders['cameras']
        self.view_number = 3

    def _call(self, inputs):
        coord = inputs
        out1_list = []
        out2_list = []
        out3_list = []
        out4_list = []

        for i in range(self.view_number):
            point_origin = camera_trans_inv(self.camera[0], inputs)
            point_crrent = camera_trans(self.camera[i], point_origin)
            X = point_crrent[:, 0]
            Y = point_crrent[:, 1]
            Z = point_crrent[:, 2]
            h = 248.0 * tf.divide(-Y, -Z) + 112.0
            w = 248.0 * tf.divide(X, -Z) + 112.0

            h = tf.minimum(tf.maximum(h, 0), 223)
            w = tf.minimum(tf.maximum(w, 0), 223)
            n = tf.cast(tf.fill(tf.shape(h), i), tf.float32)

            indeces = tf.stack([n, h, w], 1)

            idx = tf.cast(indeces / (224.0 / 56.0), tf.int32)
            out1 = tf.gather_nd(self.img_feat[0], idx)
            out1_list.append(out1)
            idx = tf.cast(indeces / (224.0 / 28.0), tf.int32)
            out2 = tf.gather_nd(self.img_feat[1], idx)
            out2_list.append(out2)
            idx = tf.cast(indeces / (224.0 / 14.0), tf.int32)
            out3 = tf.gather_nd(self.img_feat[2], idx)
            out3_list.append(out3)
            idx = tf.cast(indeces / (224.0 / 7.00), tf.int32)
            out4 = tf.gather_nd(self.img_feat[3], idx)
            out4_list.append(out4)
        # ----
        all_out1 = tf.stack(out1_list, 0)
        all_out2 = tf.stack(out2_list, 0)
        all_out3 = tf.stack(out3_list, 0)
        all_out4 = tf.stack(out4_list, 0)

        # 3*N*[64+128+256+512] -> 3*N*F
        image_feature = tf.concat([all_out1, all_out2, all_out3, all_out4], 2)
        # 3*N*F -> N*F
        # image_feature = tf.reshape(tf.transpose(image_feature, [1, 0, 2]), [-1, FLAGS.feat_dim * 3])

        #image_feature = tf.reduce_max(image_feature, axis=0)
        image_feature_max = tf.reduce_max(image_feature, axis=0)
        image_feature_mean = tf.reduce_mean(image_feature, axis=0)
        image_feature_std = reduce_std(image_feature, axis=0)

        outputs = tf.concat([coord, image_feature_max, image_feature_mean, image_feature_std], 1)
        return outputs


class SampleHypothesis(Layer):
    def __init__(self, placeholders, **kwargs):
        super(SampleHypothesis, self).__init__(**kwargs)
        self.sample_delta = placeholders['sample_coord']

    def __call__(self, mesh_coords):
        """
        Local Grid Sample for fast matching init mesh
        :param mesh_coords:
        [N,S,3] ->[NS,3] for projection
        :return: sample_points_per_vertices: [NS, 3]
        """
        with tf.name_scope(self.name):
            center_points = tf.expand_dims(mesh_coords, axis=1)
            center_points = tf.tile(center_points, [1, 43, 1])

            delta = tf.expand_dims(self.sample_delta, 0)

            sample_points_per_vertices = tf.add(center_points, delta)

            outputs = tf.reshape(sample_points_per_vertices, [-1, 3])
        return outputs


class LocalGConv(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=False, act=tf.nn.relu, bias=True, **kwargs):
        super(LocalGConv, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['sample_adj']

        self.bias = bias
        self.local_graph_vert = 43

        self.output_dim = output_dim
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs  # N, S, VF
        # dropout
        x = tf.nn.dropout(x, 1 - self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            pre_sup = tf.einsum('ijk,kl->ijl', x, self.vars['weights_' + str(i)])
            support = tf.einsum('ij,kjl->kil', self.support[i], pre_sup)
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class DeformationReasoning(Layer):
    def __init__(self, input_dim, output_dim, placeholders, gcn_block=-1, args=None, **kwargs):
        super(DeformationReasoning, self).__init__(**kwargs)
        self.delta_coord = placeholders['sample_coord']
        self.s = 43
        self.f = args.stage2_feat_dim
        self.hidden_dim = 192
        with tf.variable_scope(self.name):
            self.local_conv1 = LocalGConv(input_dim=input_dim, output_dim=self.hidden_dim, placeholders=placeholders)
            self.local_conv2 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, placeholders=placeholders)
            self.local_conv3 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, placeholders=placeholders)
            self.local_conv4 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, placeholders=placeholders)
            self.local_conv5 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, placeholders=placeholders)
            self.local_conv6 = LocalGConv(input_dim=self.hidden_dim, output_dim=1, placeholders=placeholders)

    def _call(self, inputs):
        proj_feat, prev_coord = inputs[0], inputs[1]
        with tf.name_scope(self.name):
            x = proj_feat  # NS, F
            x = tf.reshape(x, [-1, self.s, self.f])  # N,S,F
            x1 = self.local_conv1(x)
            x2 = self.local_conv2(x1)
            x3 = tf.add(self.local_conv3(x2), x1)
            x4 = self.local_conv4(x3)
            x5 = tf.add(self.local_conv5(x4), x3)
            x6 = self.local_conv6(x5)  # N, S, 1
            score = tf.nn.softmax(x6, axis=1)  # N, S, 1
            tf.summary.histogram('score', score)
            delta_coord = score * self.delta_coord
            next_coord = tf.reduce_sum(delta_coord, axis=1)
            next_coord += prev_coord
            return next_coord


class LocalGraphProjection(Layer):
    def __init__(self, placeholders, **kwargs):
        super(LocalGraphProjection, self).__init__(**kwargs)

        self.img_feat = placeholders['img_feat']
        self.camera = placeholders['cameras']
        self.view_number = 3

    def _call(self, inputs):
        coord = inputs
        out1_list = []
        out2_list = []
        out3_list = []
        # out4_list = []

        for i in range(self.view_number):
            point_origin = camera_trans_inv(self.camera[0], inputs)
            point_crrent = camera_trans(self.camera[i], point_origin)
            X = point_crrent[:, 0]
            Y = point_crrent[:, 1]
            Z = point_crrent[:, 2]
            h = 248.0 * tf.divide(-Y, -Z) + 112.0
            w = 248.0 * tf.divide(X, -Z) + 112.0

            h = tf.minimum(tf.maximum(h, 0), 223)
            w = tf.minimum(tf.maximum(w, 0), 223)
            n = tf.cast(tf.fill(tf.shape(h), i), tf.int32)

            x = h / (224.0 / 224)
            y = w / (224.0 / 224)
            out1 = self.bi_linear_sample(self.img_feat[0], n, x, y)
            out1_list.append(out1)
            x = h / (224.0 / 112)
            y = w / (224.0 / 112)
            out2 = self.bi_linear_sample(self.img_feat[1], n, x, y)
            out2_list.append(out2)
            x = h / (224.0 / 56)
            y = w / (224.0 / 56)
            out3 = self.bi_linear_sample(self.img_feat[2], n, x, y)
            out3_list.append(out3)
        # ----
        all_out1 = tf.stack(out1_list, 0)
        all_out2 = tf.stack(out2_list, 0)
        all_out3 = tf.stack(out3_list, 0)

        # 3*N*[16+32+64] -> 3*N*F
        image_feature = tf.concat([all_out1, all_out2, all_out3], 2)

        image_feature_max = tf.reduce_max(image_feature, axis=0)
        image_feature_mean = tf.reduce_mean(image_feature, axis=0)
        image_feature_std = reduce_std(image_feature, axis=0)

        outputs = tf.concat([coord, image_feature_max, image_feature_mean, image_feature_std], 1)
        return outputs

    def bi_linear_sample(self, img_feat, n, x, y):
        x1 = tf.floor(x)
        x2 = tf.ceil(x)
        y1 = tf.floor(y)
        y2 = tf.ceil(y)
        Q11 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x1, tf.int32), tf.cast(y1, tf.int32)], 1))
        Q12 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x1, tf.int32), tf.cast(y2, tf.int32)], 1))
        Q21 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x2, tf.int32), tf.cast(y1, tf.int32)], 1))
        Q22 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x2, tf.int32), tf.cast(y2, tf.int32)], 1))

        weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y2, y))
        Q11 = tf.multiply(tf.expand_dims(weights, 1), Q11)
        weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y2, y))
        Q21 = tf.multiply(tf.expand_dims(weights, 1), Q21)
        weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y, y1))
        Q12 = tf.multiply(tf.expand_dims(weights, 1), Q12)
        weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y, y1))
        Q22 = tf.multiply(tf.expand_dims(weights, 1), Q22)
        outputs = tf.add_n([Q11, Q21, Q12, Q22])
        return outputs
