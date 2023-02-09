from .base_options import *

class TemplateOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        '''Add your parameters here'''

        parser.add_argument('--output_nc', type=int, default=1, help='')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--class_num', type=int,default=2, help='class number of data')
        parser.add_argument('--grad_iter_size', type=int, default=1, help='# grad iter size')
        parser.add_argument('--l_state', type=str,default='train', help='learning state')
        parser.add_argument('--recall_thred', type=float, default=0.5, help='recall_thred')
        parser.add_argument('--vis_layer_names', type=str, default='["backbone.layer4"]', help='the names of visible layers')
        parser.add_argument('--vis_method', type=str, default=None, help='the names of visible layers')
        parser.add_argument('--vis_all_modules', type=int, default=0)
        return parser