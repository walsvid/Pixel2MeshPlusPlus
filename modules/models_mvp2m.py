# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn
import os
import tensorflow.contrib.layers as tfcontriblayers

from modules.losses import mesh_loss, laplace_loss
from modules.layers import GraphConvolution, GraphPooling, GraphProjection


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'suffix'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        save_dir_suffix = kwargs.get('suffix', '')
        self.save_dir_suffix = save_dir_suffix

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output1_2 = None
        self.output2_2 = None

        self.loss = 0
        self.inc_loss = 0
        self.pose_loss = 0
        self.optimizer = None
        self.optimizer_inc = None
        self.opt_op_vp = None
        self.opt_op_vi = None
        self.opt_op = None
        self.summary = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None, ckpt_path=None, step=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')
        saver = tf.train.Saver(self.vars, max_to_keep=0)
        save_path = saver.save(sess, os.path.join(ckpt_path, '{}.ckpt'.format(self.name)), global_step=step)
        print('Model saved in file: {}, epoch {}'.format(save_path, step))

    def load(self, sess=None, ckpt_path=None, step=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')
        # print(self.vars)
        saver = tf.train.Saver(self.vars)
        save_path = os.path.join(ckpt_path, '{}.ckpt-{}'.format(self.name, step))
        saver.restore(sess, save_path)
        print('Model restored from file: {}, epoch {}'.format(save_path, step))



class MeshNetMVP2M(Model):
    def __init__(self, placeholders, args, **kwargs):
        super(MeshNetMVP2M, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.placeholders = placeholders
        self.args = args
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        self.summary_loss = None
        self.merged_summary_op = None
        self.build()

    def _loss(self):
        # Pixel2mesh loss
        self.loss += mesh_loss(self.output1, self.placeholders, 1)
        self.loss += mesh_loss(self.output2, self.placeholders, 2)
        self.loss += mesh_loss(self.output3, self.placeholders, 3)
        self.loss += laplace_loss(self.inputs, self.output1, self.placeholders, 1)
        self.loss += laplace_loss(self.output1_2, self.output2, self.placeholders, 2)
        self.loss += laplace_loss(self.output2_2, self.output3, self.placeholders, 3)

        # Weight decay loss
        conv_layers = list(range(1, 15)) + list(range(17, 31)) + list(range(33, 48))
        for layer_id in conv_layers:
            for var in self.layers[layer_id].vars.values():
                self.loss += 5e-6 * tf.nn.l2_loss(var)

    def _build(self):
        with tf.name_scope('pixel2mesh'):
            self.build_cnn18()  # update image feature
            # first project block
            self.layers.append(GraphProjection(placeholders=self.placeholders, name='graph_proj_1_layer_0'))
            self.layers.append(GraphConvolution(input_dim=self.args.feat_dim,
                                                output_dim=self.args.hidden_dim,
                                                gcn_block_id=1,
                                                placeholders=self.placeholders,
                                                name='graph_conv_blk1_1_layer_1', logging=self.logging))
            for _ in range(12):
                self.layers.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                    output_dim=self.args.hidden_dim,
                                                    gcn_block_id=1,
                                                    placeholders=self.placeholders,
                                                    name='graph_conv_blk1_{}_layer_{}'.format(2 + _, 2 + _),
                                                    logging=self.logging))
            # activation #15; layer #14; output 1
            self.layers.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                output_dim=self.args.coord_dim,
                                                act=lambda x: x,
                                                gcn_block_id=1,
                                                placeholders=self.placeholders,
                                                name='graph_conv_blk1_14_layer_14', logging=self.logging))
            # second project block
            self.layers.append(GraphProjection(placeholders=self.placeholders, name='graph_proj_2_layer_15'))
            self.layers.append(GraphPooling(placeholders=self.placeholders, pool_id=1, name='graph_pool_1to2_layer_16'))
            self.layers.append(GraphConvolution(input_dim=self.args.feat_dim + self.args.hidden_dim,
                                                output_dim=self.args.hidden_dim,
                                                gcn_block_id=2,
                                                placeholders=self.placeholders,
                                                name='graph_conv_blk2_1_layer_17', logging=self.logging))
            for _ in range(12):
                self.layers.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                    output_dim=self.args.hidden_dim,
                                                    gcn_block_id=2,
                                                    placeholders=self.placeholders,
                                                    name='graph_conv_blk2_{}_layer_{}'.format(2 + _, 18 + _),
                                                    logging=self.logging))
            self.layers.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                output_dim=self.args.coord_dim,
                                                act=lambda x: x,
                                                gcn_block_id=2,
                                                placeholders=self.placeholders,
                                                name='graph_conv_blk2_14_layer_30', logging=self.logging))
            # third project block
            self.layers.append(GraphProjection(placeholders=self.placeholders, name='graph_proj_3_layer_31'))
            self.layers.append(GraphPooling(placeholders=self.placeholders, pool_id=2, name='graph_pool_2to3_layer_32'))
            self.layers.append(GraphConvolution(input_dim=self.args.feat_dim + self.args.hidden_dim,
                                                output_dim=self.args.hidden_dim,
                                                gcn_block_id=3,
                                                placeholders=self.placeholders,
                                                name='graph_conv_blk3_1_layer_33', logging=self.logging))
            for _ in range(13):
                self.layers.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                    output_dim=self.args.hidden_dim,
                                                    gcn_block_id=3,
                                                    placeholders=self.placeholders,
                                                    name='graph_conv_blk3_{}_layer_{}'.format(2 + _, 34 + _),
                                                    logging=self.logging))
            self.layers.append(GraphConvolution(input_dim=self.args.hidden_dim,
                                                output_dim=self.args.coord_dim,
                                                act=lambda x: x,
                                                gcn_block_id=3,
                                                placeholders=self.placeholders,
                                                name='graph_conv_blk3_15_layer_47', logging=self.logging))

    def build_cnn18(self):
        x = self.placeholders['img_inp']
        # x = tf.expand_dims(x, 0)
# 224 224
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_1')
        x = tflearn.layers.conv.conv_2d(x, 16, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_2')
        x0 = x
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_3')
# 112 112
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_4')
        x = tflearn.layers.conv.conv_2d(x, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_5')
        x1 = x
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_6')
# 56 56
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_7')
        x = tflearn.layers.conv.conv_2d(x, 64, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_8')
        x2 = x
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_9')
# 28 28
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_10')
        x = tflearn.layers.conv.conv_2d(x, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_11')
        x3 = x
        x = tflearn.layers.conv.conv_2d(x, 256, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_12')
# 14 14
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_13')
        x = tflearn.layers.conv.conv_2d(x, 256, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_14')
        x4 = x
        x = tflearn.layers.conv.conv_2d(x, 512, (5, 5), strides=2, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_15')
# 7 7
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_16')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_17')
        x = tflearn.layers.conv.conv_2d(x, 512, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2', scope='cnn/conv2d_18')
        x5 = x
        # updata image feature
        self.placeholders.update({'img_feat': [tf.squeeze(x2), tf.squeeze(x3), tf.squeeze(x4), tf.squeeze(x5)]})
        self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.3

    def build(self):
        ''' Wrapper for _build() '''
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential resnet model
        eltwise = [3, 5, 7, 9, 11, 13,
                   19, 21, 23, 25, 27, 29,
                   35, 37, 39, 41, 43, 45]
        concat = [15, 31]
        # proj = [0, 15, 31]
        self.activations.append(self.inputs)
        for idx, layer in enumerate(self.layers[:48]):
            hidden = layer(self.activations[-1])
            if idx in eltwise:
                hidden = tf.add(hidden, self.activations[-2]) * 0.5
            if idx in concat:
                hidden = tf.concat([hidden, self.activations[-2]], 1)
            self.activations.append(hidden)

        self.output1 = self.activations[15]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=1)
        self.output1_2 = unpool_layer(self.output1)

        self.output2 = self.activations[31]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=2)
        self.output2_2 = unpool_layer(self.output2)

        self.output3 = self.activations[48]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/')
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self.opt_op = self.optimizer.minimize(self.loss)

        self.summary_loss = tf.summary.scalar('loss', self.loss)
        self.merged_summary_op = tf.summary.merge([self.summary_loss])
