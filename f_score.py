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


def f_score(points, labels, dist1, idx1, dist2, idx2, threshold):
    len_points = points.shape[0]
    len_labels = labels.shape[0]

    f_score = []
    for i in range(len(threshold)):
        num = len(np.where(dist1 <= threshold[i])[0]) + 0.0
        P = 100.0 * (num / len_points)
        num = len(np.where(dist2 <= threshold[i])[0]) + 0.0
        R = 100.0 * (num / len_labels)
        f_score.append((2 * P * R) / (P + R + 1e-6))
    return np.array(f_score)


if __name__ == '__main__':
    print('=> set config')
    args = execute()
    pprint.pprint(vars(args))
    pred_file_list = os.path.join(args.save_path, args.name, 'predict', str(args.test_epoch), '*_predict.xyz')
    xyz_list_path = glob.glob(pred_file_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # exit(0)
    xyz1 = tf.placeholder(tf.float32, shape=(None, 3))
    xyz2 = tf.placeholder(tf.float32, shape=(None, 3))
    dist1, idx1, dist2, idx2 = nn_distance(xyz1, xyz2)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # xyz_list_path = sys.argv[1]
    threshold = [0.00005, 0.00010, 0.00015, 0.00020]
    name = {'02828884': 'bench', '03001627': 'chair', '03636649': 'lamp', '03691459': 'speaker', '04090263': 'firearm',
            '04379243': 'table', '04530566': 'watercraft', '02691156': 'plane', '02933112': 'cabinet',
            '02958343': 'car', '03211117': 'monitor', '04256520': 'couch', '04401088': 'cellphone'}
    length = {'02828884': 0, '03001627': 0, '03636649': 0, '03691459': 0, '04090263': 0, '04379243': 0, '04530566': 0,
              '02691156': 0, '02933112': 0, '02958343': 0, '03211117': 0, '04256520': 0, '04401088': 0}
    sum_pred = {'02828884': np.zeros(4), '03001627': np.zeros(4), '03636649': np.zeros(4), '03691459': np.zeros(4),
                '04090263': np.zeros(4), '04379243': np.zeros(4), '04530566': np.zeros(4), '02691156': np.zeros(4),
                '02933112': np.zeros(4), '02958343': np.zeros(4), '03211117': np.zeros(4), '04256520': np.zeros(4),
                '04401088': np.zeros(4)}

    index = 0
    total_num = len(xyz_list_path)
    for pred_path in xyz_list_path:
        lab_path = pred_path.replace('_predict', '_ground')
        ground = np.loadtxt(lab_path)[:, :3]
        predict = np.loadtxt(pred_path)

        class_id = pred_path.split('/')[-1].split('_')[0]
        length[class_id] += 1.0
        d1, i1, d2, i2 = sess.run([dist1, idx1, dist2, idx2], feed_dict={xyz1: predict, xyz2: ground})
        sum_pred[class_id] += f_score(predict, ground, d1, i1, d2, i2, threshold)

        index += 1
        print('processed number', index, total_num)

    print(sum_pred)
    log_dir = os.path.join(args.save_path, args.name, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, '{}_f_score.log'.format(args.test_epoch))
    print(log_path)
    logfile = open(log_path, 'a')
    means = []
    for item in length:
        number = length[item] + 1e-6
        score = sum_pred[item] / number
        means.append(score)
        print(item, name[item], length[item], ' '.join(map(str, score)))
        print(item, name[item], length[item], ' '.join(map(str, score)), file=logfile)
    print('-' * 80)
    print('mean', 'all_data', 'total_number', np.mean(means, axis=0))
    sess.close()
