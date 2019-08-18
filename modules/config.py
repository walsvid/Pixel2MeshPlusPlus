# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import argparse
import yaml
import re


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def execute():
    return parse_args(create_parser())


def create_parser():
    parser = argparse.ArgumentParser()

    # g = parser.add_argument_group('Device Targets')
    parser.add_argument('-f', '--config_file', dest='config_file', type=argparse.FileType(mode='r'))

    parser.add_argument('--train_file_path', help='train file path')
    parser.add_argument('--train_data_path', help='train data path')
    parser.add_argument('--train_image_path', help='train image path')

    parser.add_argument('--coarse_result_file_path', help='coarse result file path')
    parser.add_argument('--coarse_result_data_path', help='coarse result data path')
    parser.add_argument('--coarse_result_image_path', help='coarse result image path')

    parser.add_argument('--test_file_path', help='test file path')
    parser.add_argument('--test_data_path', help='test data path')
    parser.add_argument('--test_image_path', help='test image path')

    parser.add_argument('--train_mesh_root', help='init mesh root path')
    parser.add_argument('--test_mesh_root', help='init mesh root path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--test_epoch', type=int, default=0, help='test epoch')
    parser.add_argument('--hidden_dim', type=int, default=192, help='hidden dim')
    parser.add_argument('--feat_dim', type=int, default=2883, help='feat dim')
    parser.add_argument('--stage2_feat_dim', type=int, default=339, help='stage2 feat dim')

    parser.add_argument('--coord_dim', type=int, default=3, help='coord dim')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('-g', '--gpu_id', help='devices ids')
    parser.add_argument('-s', '--save_path', help='save root')
    parser.add_argument('-n', '--name', help='exprtiments name')
    parser.add_argument('--restore', type=str2bool, default=False)
    parser.add_argument('--is_debug', type=str2bool, default='no', help='is debug')
    parser.add_argument('--is_voxel_input', type=str2bool, default='no', help='is debug')
    # pretrained cnn
    parser.add_argument('--load_cnn', type=str2bool, default=False)
    parser.add_argument('--pre_trained_cnn_path', help='pre-trained cnn path')
    parser.add_argument('--cnn_step', type=int, help='cnn pre-trained step')
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        data = yaml.load(args.config_file, Loader=loader)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        # print(len(list(arg_dict.keys())))
        # print(len(list(data.keys())))
        for key, value in arg_dict.items():
            default_arg = parser.get_default(key)
            if arg_dict[key] == default_arg and key in data:
                arg_dict[key] = data[key]
    return args
