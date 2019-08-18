# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import os
import sys
import numpy as np
import pickle as pickle
import tensorflow as tf
import pprint
import glob
import os
from modules.chamfer import nn_distance
from modules.config import execute

if __name__ == '__main__':
    print('=> set config')
    args = execute()
    pprint.pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    xyz1 = tf.placeholder(tf.float32, shape=(None, 3))
    xyz2 = tf.placeholder(tf.float32, shape=(None, 3))
    dist1, idx1, dist2, idx2 = nn_distance(xyz1, xyz2)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    pred_file_list = os.path.join(args.save_path, args.name, 'predict', str(args.test_epoch), '*_predict.xyz')
    xyz_list_path = glob.glob(pred_file_list)

    log_dir = os.path.join(args.save_path, args.name, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, '{}_cd.log'.format(args.test_epoch))

    name = {'02828884': 'bench', '03001627': 'chair', '03636649': 'lamp', '03691459': 'speaker', '04090263': 'firearm',
            '04379243': 'table', '04530566': 'watercraft', '02691156': 'plane', '02933112': 'cabinet',
            '02958343': 'car', '03211117': 'monitor', '04256520': 'couch', '04401088': 'cellphone'}
    length = {'02828884': 0, '03001627': 0, '03636649': 0, '03691459': 0, '04090263': 0, '04379243': 0, '04530566': 0,
              '02691156': 0, '02933112': 0, '02958343': 0, '03211117': 0, '04256520': 0, '04401088': 0}
    sum_pred = {'02828884': 0, '03001627': 0, '03636649': 0, '03691459': 0, '04090263': 0, '04379243': 0, '04530566': 0,
                '02691156': 0, '02933112': 0, '02958343': 0, '03211117': 0, '04256520': 0, '04401088': 0}

    index = 0
    total_num = len(xyz_list_path)
    for pred_path in xyz_list_path:
        lab_path = pred_path.replace('_predict', '_ground')
        ground = np.loadtxt(lab_path)[:, :3]
        predict = np.loadtxt(pred_path)

        class_id = pred_path.split('/')[-1].split('_')[0]
        length[class_id] += 1.0

        d1, i1, d2, i2 = sess.run([dist1, idx1, dist2, idx2], feed_dict={xyz1: predict, xyz2: ground})
        cd_distance = np.mean(d1) + np.mean(d2)
        sum_pred[class_id] += cd_distance

        index += 1
        print('processed number', index, total_num)

    print(log_path)
    log = open(log_path, 'a')
    for item in length:
        number = length[item] + 1e-6
        score = (sum_pred[item] / number) * 10000
        print(item, name[item], int(length[item]), score)
        print(item, name[item], int(length[item]), score, file=log)
    sess.close()
