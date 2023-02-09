# federated
import argparse
import os
import json
from util import basic

class BaseOptions(object):
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, rank):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.parser = self.gather_options(rank)
        self.opt = None

    def initialize(self, rank, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--ppid', type=str, default='0', help='parent pid')
        parser.add_argument('--pid', type=str, default='0', help='pid')
        parser.add_argument('--ad_stage', type=str, default='0', help='')
        parser.add_argument('--name', type=str, default='debug', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        parser.add_argument('--num_threads', default=5, type=int, help='# threads for loading data')
        parser.add_argument('--load_dir', type=str, default='', help='model paths to be loaded')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--valid_metric_value', type=float, default=-1, help='')
        parser.add_argument('--o_save_dir', type=str, default='experiment', help='')
        parser.add_argument('--save_dir', type=str, default='experiment', help='')
        parser.add_argument('--load_dir_ind', type=int, default = -1)
        parser.add_argument('--load_strict', type=int, default=1, help='model paths to be loaded')
        parser.add_argument('--vis_dir', default = './vis', help='save images output')
        parser.add_argument('--pre_check_dir', default = '', help='')
        parser.add_argument('--index', default=rank)
        self.initialized = True
        return parser

    def gather_options(self, rank):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
            parser = self.initialize(rank, parser)
        opt, _ = parser.parse_known_args()
        return parser

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
        #expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        hospital_list = ['sx', 'cc', 'sz', 'zs', 'nm', 'sc', 'gy', 'gzzl', \
                     'zd', 's3', 'fj', 'xa', 'ln', 'cq', 'qd'] 
        expr_dir = os.path.join(opt.checkpoints_dir, opt.path, hospital_list[opt.index])
        opt.o_save_dir = expr_dir
        opt.save_dir = expr_dir
        basic.mkdirs(expr_dir)
        basic.save_code('.', os.path.join(expr_dir, 'code.zip'))

        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        file_name = os.path.join(expr_dir, '{}_opt.json'.format(opt.phase))
        with open(file_name, 'w') as opt_file:
            json.dump(vars(opt), opt_file)

    def opt_revise(self,opt):
        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.parser.parse_args()
        opt = self.opt_revise(opt)
        self.print_options(opt)
        self.opt = opt
        return self.opt