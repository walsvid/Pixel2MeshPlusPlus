# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import numpy as np
import tensorflow as tf
from modules.chamfer import nn_distance
from scipy import stats


def laplace_coord(pred, placeholders, block_id):
    vertex = tf.concat([pred, tf.zeros([1, 3])], 0)
    indices = placeholders['lape_idx'][block_id - 1][:, :8]
    weights = tf.cast(placeholders['lape_idx'][block_id - 1][:, -1], tf.float32)

    weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1, 1]), [1, 3])
    laplace = tf.reduce_sum(tf.gather(vertex, indices), 1)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    return laplace


def laplace_loss(pred1, pred2, placeholders, block_id):
    # laplace term
    lap1 = laplace_coord(pred1, placeholders, block_id)
    lap2 = laplace_coord(pred2, placeholders, block_id)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), 1)) * 1500
    move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 50
    return laplace_loss + move_loss


def unit(tensor):
    return tf.nn.l2_normalize(tensor, axis=1)


def mesh_loss(pred, placeholders, block_id):
    gt_pt = placeholders['labels'][:, :3]  # gt points
    gt_nm = placeholders['labels'][:, 3:]  # gt normals

    # edge in graph
    nod1 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 0])
    nod2 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 1])
    edge = tf.subtract(nod1, nod2)

    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * 350

    # chamfer distance
    sample_pt = sample(pred, placeholders, block_id)
    sample_pred = tf.concat([pred, sample_pt], axis=0)
    dist1, idx1, dist2, idx2 = nn_distance(gt_pt, sample_pred)
    point_loss = (tf.reduce_mean(dist1) + 0.55 * tf.reduce_mean(dist2)) * 3000

    # normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal, placeholders['edges'][block_id - 1][:, 0])
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1))
    normal_loss = tf.reduce_mean(cosine) * 0.5

    total_loss = point_loss + edge_loss + normal_loss
    return total_loss


def sample(pred, placeholders, block_id):
    uni = tf.distributions.Uniform(low=0.0, high=1.0)
    faces = placeholders['faces_triangle'][block_id - 1]
    tilefaces = tf.py_func(choice_faces, [pred, faces], tf.int32)

    num_of_tile_faces = tf.shape(tilefaces)[0]

    xs = tf.gather(pred, tilefaces[:, 0])
    ys = tf.gather(pred, tilefaces[:, 1])
    zs = tf.gather(pred, tilefaces[:, 2])

    u = tf.sqrt(uni.sample([num_of_tile_faces, 1]))
    v = uni.sample([num_of_tile_faces, 1])
    points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs
    return points


def choice_faces(verts, faces):
    num = 4000
    u1, u2, u3 = np.split(verts[faces[:, 0]] - verts[faces[:, 1]], 3, axis=1)
    v1, v2, v3 = np.split(verts[faces[:, 1]] - verts[faces[:, 2]], 3, axis=1)
    a = (u2 * v3 - u3 * v2) ** 2
    b = (u3 * v1 - u1 * v3) ** 2
    c = (u1 * v2 - u2 * v1) ** 2
    Areas = np.sqrt(a + b + c) / 2
    Areas = Areas / np.sum(Areas)
    choices = np.expand_dims(np.arange(Areas.shape[0]), 1)
    dist = stats.rv_discrete(name='custm', values=(choices, Areas))
    choices = dist.rvs(size=num)
    select_faces = faces[choices]
    return select_faces


def laplace_loss_2(pred1, pred2, placeholders, block_id):
    # laplace term
    lap1 = laplace_coord(pred1, placeholders, block_id)
    lap2 = laplace_coord(pred2, placeholders, block_id)
    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), 1)) * 1500
    move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
    return laplace_loss + move_loss


def mesh_loss_2(pred, placeholders, block_id):
    gt_pt = placeholders['labels'][:, :3]  # gt points
    gt_nm = placeholders['labels'][:, 3:]  # gt normals

    # edge in graph
    nod1 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 0])
    nod2 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 1])
    edge = tf.subtract(nod1, nod2)

    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * 500

    # chamfer distance
    sample_pt = sample(pred, placeholders, block_id)
    sample_pred = tf.concat([pred, sample_pt], axis=0)
    dist1, idx1, dist2, idx2 = nn_distance(gt_pt, sample_pred)
    point_loss = (tf.reduce_mean(dist1) + 0.55 * tf.reduce_mean(dist2)) * 3000

    # normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal, placeholders['edges'][block_id - 1][:, 0])
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1))
    normal_loss = tf.reduce_mean(cosine) * 0.5

    total_loss = point_loss + edge_loss + normal_loss
    return total_loss
