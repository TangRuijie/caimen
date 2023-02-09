import sys
import time
from . import basic
from tensorboardX import SummaryWriter
from util.basic import *
import cv2

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id

        if self.display_id > 0:
            self.writer = SummaryWriter(log_dir='runs/' + opt.path, comment=opt.path)

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.path, 'loss_log.txt')
        self.o_log_p2 = self.log_name.split('/')[-2]
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def plot_current_info(self, model, citer, ptype):
        if self.display_id <= 0:
            return

        for k in model.loss_names:
            v = model.meters[k].val
            self.writer.add_scalar(ptype + '/loss_' + k, v, citer)

        for k in model.s_metric_names:
            v = model.meters[k].val
            self.writer.add_scalar(ptype + '/' + k, v, citer)

    def plot_global_info(self, model, citer, ptype):
        if self.display_id <= 0:
            return

        if self.opt.l_state == 'train' and len(model.schedulers) > 0:
            v = model.schedulers[0].lr
            self.writer.add_scalar(ptype + '/lr', v, citer)

        for k in model.loss_names:
            v = model.meters[k].avg
            self.writer.add_scalar(ptype + '/global_loss_' + k, v, citer)

        for k in model.s_metric_names:
            v = model.meters[k].avg
            self.writer.add_scalar(ptype + '/global_' + k, v, citer)

        for k in model.g_metric_names:
            v = model.meters[k].val
            self.writer.add_scalar(ptype + '/' + k, v, citer)

        for k in model.t_metric_names:
            v = model.meters[k].val
            v = numpy2table(v)
            self.writer.add_text(ptype + '/' + k, v, citer)

        model.plot_special_info()

    # losses: same format as |losses| of plot_current_losses
    def print_current_info(self, epoch, iters, model, t_comp, t_data):
        """print current losses on console; also save the losses to the disk


        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, model: %.3fs, data: %.3fs) ' % (epoch, iters, t_comp, t_data)

        if self.opt.l_state == 'train' and len(model.schedulers) > 0:
            v = model.schedulers[0].lr
            message += '%s: %f ' % ('lr', v)

        for k in model.loss_names:
            v = model.meters[k].val
            message += 'loss_%s: %.3f ' % (k, v)

        for k in model.s_metric_names:
            v = model.meters[k].val
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message

        # if self.opt.l_state != 'test':
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


    def print_global_info(self, epoch, iters, model, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """

        message = '(epoch: %d, iters: %d, time: %.3fm, data: %.3fm) \n' % (epoch, iters, t_comp, t_data)

        #learning rate

        if self.opt.l_state == 'train' and len(model.schedulers) > 0:
            v = model.schedulers[0].lr
            message += '%s: %f ' % ('lr', v)

        for k in model.loss_names:
            v = model.meters[k].avg
            message += 'loss_%s: %.3f ' % (k, v)

        for k in model.s_metric_names:
            v = model.meters[k].avg
            message += '%s: %.3f ' % (k, v)

        for k in model.g_metric_names:
            v = model.meters[k].val
            message += '%s: %.3f ' % (k, v)

        for k in model.t_metric_names:
            v = model.meters[k].sum
            message += '%s: \n %s\n' % (k, str(v))

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

        model.print_special_info(self.log_name)


    def visual_images(self,model):
        save_path = self.opt.image_save_dir + '/' + self.opt.path

        mkdir(save_path)

        if 'ac' in self.opt.plot_info:
            model.visualization()

        v_type_list = []
        if 'right' in self.opt.plot_info or 'correct' in self.opt.plot_info:
            v_type_list.append('correct')

        if 'wrong' in self.opt.plot_info:
            v_type_list.append('wrong')

        for v_type in v_type_list:
            im_list = model.get_v_images(v_type)
            for label,im in enumerate(im_list):
                if not isinstance(im,type(None)):
                    # self.writer.add_image(ptype + '/correct ' + str(label), im, citer)
                    im = np.repeat(np.transpose(im,((1,2,0))),3,-1)
                    cv2.imwrite(save_path + '/' + v_type + '_' + str(label) + '.jpg',im)
                    print('im shape ' + str(im.shape))
