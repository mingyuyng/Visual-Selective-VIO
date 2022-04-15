import argparse
import os
import torch


class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
        parser.add_argument('--seq_len', type=int, default=11, help='sequence length for test')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--model_name', type=str, default='vf_512_if_256_3e-05', help='model name to load')
        parser.add_argument('--save_dir', type=str, default='results', help='path to save the result')
        parser.add_argument('--img_w', type=int, default=512, help='image width')
        parser.add_argument('--img_h', type=int, default=256, help='image height')
        parser.add_argument('--fuse_method', type=str, default='cat', help='fuse method')
        parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')
        parser.add_argument('--test_list', type=list, default=['05', '07', '10'], help='sequences to test')
        parser.add_argument('--window_size', type=int, default=30, help='window size to smooth the decisions')

        # model parameters
        parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
        parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
        parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')
        parser.add_argument('--visual_f_len', type=int, default=512, help='visual feature length')
        parser.add_argument('--imu_f_len', type=int, default=256, help='imu feature length')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.save_dir, opt.model_name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        opt.load_path = 'models/{}.model'.format(opt.model_name)

        self.opt = opt
        return self.opt
