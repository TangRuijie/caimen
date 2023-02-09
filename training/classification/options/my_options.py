from .base_options import *
import sys
import models
import datasets
import schedulers


class MyOptions(BaseOptions):
    def initialize(self, rank, parser):
        parser = super(MyOptions, self).initialize(rank, parser)
        parser.add_argument('--model', type=str, default='debug', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--load_net', type=int, default=1)
        parser.add_argument('--save_base_model', type=int, default=0)
        parser.add_argument('--balanced_loss_type', type=str, default='focal')
        parser.add_argument('--input_nc', type=int, default=1, help='')
        parser.add_argument('--output_nc', type=int, default=1, help='')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--class_num', type=int,default=2, help='class number of data')
        parser.add_argument('--grad_iter_size', type=int, default=1, help='# grad iter size')
        parser.add_argument('--l_state', type=str,default='train', help='learning state')
        parser.add_argument('--recall_thred', type=float, default=0.5, help='recall_thred')
        parser.add_argument('--vis_layer_names', type=str, default='["backbone.layer4"]', help='the names of visible layers')
        parser.add_argument('--vis_method', type=str, default='', help='the names of visible layers')
        parser.add_argument('--vis_all_modules', type=int, default=0)

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='debug', help='chooses train dataset')
        parser.add_argument('--dataset', type=object, default=None, help='created dataset')
        parser.add_argument('--v_dataset_mode', type=str, default='debug', help='chooses valid dataset')
        parser.add_argument('--v_dataset', type=object, default=None, help='created v_dataset')
        parser.add_argument('--train_id_list', type=object, default=None, help='train id list')
        parser.add_argument('--valid_id_list', type=object, default=None, help='valid id list')

        parser.add_argument('--batch_size', type=int, default=28, help='input batch size')
        parser.add_argument('--v_batch_size', type=int, default=1, help='valid input batch size')
        parser.add_argument('--serial_batches',type=int, default=0, help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--buffer_loader', type=int, default=0, help='with buffer loader')
        parser.add_argument('--loader_buffer_size', type=int, default=50)

        '''data augumentation'''
        parser.add_argument('--multi_vsets', type=int, default=1, help='multi validation datasets')
        parser.add_argument('--preprocess', type=str, default='resize', help='data augumintation [resize | crop | scale | \
                                                                                 translate | rotate | shear | elastc | flip | contrast | clane]')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')

        # rand_aug
        parser.add_argument('--rand_aug', type=int, default=0, help='')
        parser.add_argument('--rand_n', type=int, default=-1, help='')
        parser.add_argument('--rand_m', type=int, default=-1, help='')

        parser.add_argument('--aug_ratio', type=float, default=0.4, help='then crop to this size')
        parser.add_argument('--scale_per_x', type=tuple, default=(0.7,1.3), help='the range of image to scale, horizontal')
        parser.add_argument('--scale_per_y', type=tuple, default=(0.7,1.3), help='the range of image to scale, vertical')
        parser.add_argument('--max_dataset_size', type=float,default=float('inf'),help='')
        parser.add_argument('--translate_pix_x', type=tuple, default=(-30, 30),
                            help='the pixcel range of image to translate, horizontal')
        parser.add_argument('--translate_pix_y', type=tuple, default=(-30, 30),
                            help='the pixcel range of image to translate, vertical')
        parser.add_argument('--rotate_der', type=tuple, default=(-20, 20), help='rotate range')
        parser.add_argument('--shear_der', type=tuple, default=(-20, 20), help='shear range')
        parser.add_argument('--elastic_alpha', type=tuple, default=(0,3), help='elastic_alpha range')
        parser.add_argument('--contrast_gain', type=tuple, default=(3, 10), help='contrast_gain')
        parser.add_argument('--contrast_cutoff', type=tuple, default=(0.4, 0.7), help='contrast_cutoff')
        parser.add_argument('--clane_limit', type=tuple, default=(1, 10), help='clane_limit')

        parser.add_argument('--flip_rate', type=float, default=0.5, help='flip_rate')
        parser.add_argument('--scale_rate', type=float, default=0.5, help='scale_rate')
        parser.add_argument('--translate_rate', type=float, default=0.7, help='translate_rate')
        parser.add_argument('--rotate_rate', type=float, default=0.7, help='rotate_rate')
        parser.add_argument('--shear_rate', type=float, default=0.7, help='shear_rate')
        parser.add_argument('--elastic_rate', type=float, default=0.7, help='elastic_rate')
        parser.add_argument('--contrast_rate', type=float, default=0.7, help='contrast_rate')
        parser.add_argument('--clane_rate', type=float, default=0.7, help='clane_rate')
        parser.add_argument('--noise_scale', type=float, default=0.2, help='clane_rate')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--repeat_iter', type=int, default=0, help='train under the same setting for several times')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--plot_info', default='right,wrong', type=str, help='plot correct or wrong images')

        #visualization parameters
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        parser.add_argument('--v_print_freq', type=int, default=10,
                            help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=400,
                            help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4,
                            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_server', type=str, default="http://localhost",
                            help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main',
                            help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8099, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000,
                            help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=10,
                            help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--buffer_limited_num', type=int, default=float('inf'))

        parser.add_argument('--path', type=str, default='0')

        return parser

    def gather_options(self, rank):
        parser = BaseOptions.gather_options(self, rank)
        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)

        if opt.dataset_mode != opt.v_dataset_mode:
            dataset_name = opt.v_dataset_mode
            dataset_option_setter = datasets.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser)

        return parser

    def supply_opt_from_load_dir(self, opt):
        if opt.load_dir == '':
            return opt

        ignore_list = ['load_dir', 'load_dir_ind']
        parameters = [arg for arg in sys.argv[1:] if arg.startswith('--')]
        tmp_list = []
        for p in parameters:
            while p.startswith('--'):
                p = p[2:]
            tmp_list.append(p)
        parameters = set(tmp_list)
        #train_opt_path = os.path.join(opt.load_dir, 'train_opt.json')
        #if os.path.exists(train_opt_path):
        #    opt_json = json.load(open(train_opt_path))
        #    print('load parameter setting')
        #    for k, v in opt_json.items():
        #        if k not in parameters and hasattr(opt, k) and k not in ignore_list:
        #            setattr(opt, k, v)
        return opt

    def opt_revise(self,opt):
        model_name = opt.model
        model_option_supplier = models.get_info_supplier(model_name)
        opt = model_option_supplier(opt)
        opt = self.supply_opt_from_load_dir(opt)

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')

        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
                
        str_info = opt.plot_info.split(',')
        opt.plot_info = []
        for info in str_info:
            opt.plot_info.append(info)

        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.parser.parse_args()
        self.opt = self.opt_revise(opt)
        return self.opt

class TrainOptions(MyOptions):
    def initialize(self, rank, parser):
        parser = super().initialize(rank, parser)
        parser.add_argument('--valid_model', type=bool, default=True, help='valid the model')
        parser.add_argument('--valid_freq', type=int, default=-1, help='frequency of validating the latest model')
        parser.add_argument('--valid_freq_ratio', type=float, default=1, help='calculating the valid_freq according to the ratio if valid_freq <= 0')
        parser.add_argument('--save_epoch_freq', type=int, default=100, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--continue_epoch', type = str, default='optimal', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--op_name', type=str, default='SGD', help='# the name of optimizer')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--nesterov', type=bool, default=True, help='# nesterov')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for adam')
        parser.add_argument('--final_lr', type=float, default=0.01, help='final learning rate for adabound')
        parser.add_argument('--grad_clip_value', type=float, default=1, help='grad clip value')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='initial weight_decay for adam')
        parser.add_argument('--momentum', type=float, default=0.9, help='initial momentum for adam')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')

        #scheduler
        parser.add_argument('--lr_policy', type=str, default='mstep', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--with_warm_up', type=int, default=1)
        parser.add_argument('--warm_epoch', type=int, default=3)
        parser.add_argument('--lr_wait_epoch', type=int, default=2)
        parser.add_argument('--warm_start_factor', type=float, default=0.3)
        parser.add_argument('--lr_decay_factor', type=float, default=0.1)
        parser.add_argument('--lr_decay_iters', type=int, default=50)
        parser.add_argument('--patient_epoch', type=int, default=10)

        return parser

    def gather_options(self, rank):
        parser = MyOptions.gather_options(self, rank)
        opt, _ = parser.parse_known_args()

        policy_name = opt.lr_policy
        if policy_name not in schedulers.basic_schedulers:
            policy_option_setter = schedulers.get_option_setter(policy_name)
            parser = policy_option_setter(parser)
        return parser

class TestOptions(MyOptions):

    def initialize(self, rank, parser):
        parser = super().initialize(rank, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--test_train_data', type=int, default=0, help='')
        return parser

