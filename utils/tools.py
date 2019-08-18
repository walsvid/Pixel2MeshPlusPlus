# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import numpy as np
import tensorflow as tf
import pickle
import cv2


def construct_feed_dict(pkl, placeholders):
    coord = pkl['coord']
    pool_idx = pkl['pool_idx']
    faces = pkl['faces']
    lape_idx = pkl['lape_idx']

    edges = []
    for i in range(1, 4):
        adj = pkl['stage{}'.format(i)][1]
        edges.append(adj[0])

    feed_dict = dict()
    feed_dict.update({placeholders['features']: coord})
    feed_dict.update({placeholders['edges'][i]: edges[i] for i in list(range(len(edges)))})
    feed_dict.update({placeholders['faces'][i]: faces[i] for i in list(range(len(faces)))})
    feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in list(range(len(pool_idx)))})
    feed_dict.update({placeholders['lape_idx'][i]: lape_idx[i] for i in list(range(len(lape_idx)))})
    feed_dict.update({placeholders['support1'][i]: pkl['stage1'][i] for i in list(range(len(pkl['stage1'])))})
    feed_dict.update({placeholders['support2'][i]: pkl['stage2'][i] for i in list(range(len(pkl['stage2'])))})
    feed_dict.update({placeholders['support3'][i]: pkl['stage3'][i] for i in list(range(len(pkl['stage3'])))})

    for k in range(3):
        feed_dict.update({placeholders['faces_triangle'][k]: pkl['faces_triangle'][k]})

    feed_dict.update({placeholders['sample_coord']: pkl['sample_coord']})

    feed_dict.update({placeholders['sample_adj'][i]: pkl['sample_cheb_dense'][i] for i in range(len(pkl['sample_cheb_dense']))})

    return feed_dict


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims) + 1e-6)

# -------------------------------------------------------------------
# cameras
# -------------------------------------------------------------------


def normal(v):
    norm = tf.norm(v)
    if norm == 0:
        return v
    return tf.divide(v, norm)


def cameraMat(param):
    theta = param[0] * np.pi / 180.0
    camy = param[3] * tf.sin(param[1] * np.pi / 180.0)
    lens = param[3] * tf.cos(param[1] * np.pi / 180.0)
    camx = lens * tf.cos(theta)
    camz = lens * tf.sin(theta)
    Z = tf.stack([camx, camy, camz])

    x = camy * tf.cos(theta + np.pi)
    z = camy * tf.sin(theta + np.pi)
    Y = tf.stack([x, lens, z])
    X = tf.cross(Y, Z)

    cm_mat = tf.stack([normal(X), normal(Y), normal(Z)])
    return cm_mat, Z


def camera_trans(camera_metadata, xyz):
    c, o = cameraMat(camera_metadata)
    points = xyz[:, :3]
    pt_trans = points - o
    pt_trans = tf.matmul(pt_trans, tf.transpose(c))
    return pt_trans


def camera_trans_inv(camera_metadata, xyz):
    c, o = cameraMat(camera_metadata)
    inv_xyz = (tf.matmul(xyz, tf.matrix_inverse(tf.transpose(c)))) + o
    return inv_xyz


def load_demo_image(demo_image_list):
    imgs = np.zeros((3, 224, 224, 3))
    for idx, demo_img_path in enumerate(demo_image_list):
        img = cv2.imread(demo_img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img[np.where(img[:, :, 3] == 0)] = 255
        img = cv2.resize(img, (224, 224))
        img_inp = img.astype('float32') / 255.0
        imgs[idx] = img_inp[:, :, :3]
    return imgs
