# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.

import sys
import os
import numpy as np


if __name__ == '__main__':
    xyz_list_path = sys.argv[1]
    xyzs = [xyz for xyz in os.listdir(xyz_list_path) if xyz.endswith('_predict_3.xyz')]
    v = np.full([2466, 1], 'v')
    for xyz in xyzs:
        print(xyz)
        obj_path = xyz.replace('.xyz', '.obj')
        xyzf = np.loadtxt(os.path.join(xyz_list_path, xyz))
        face = np.loadtxt('/home/wc/workspace/P2MPP/data/face3.obj', dtype='|S32')
        out = np.vstack((np.hstack((v, xyzf)), face))
        np.savetxt(os.path.join(xyz_list_path, obj_path), out, fmt='%s', delimiter=' ')
