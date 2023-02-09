import torch
import json
from types import FunctionType
# from imgaug import augmenters as iaa
from PIL import Image, ImageDraw, ImageFont
from util.image import *
from torch.autograd import Variable
from torch.autograd import Function
import cv2
import os.path as osp
import numpy as np


class GradCam(object):
    def __init__(self, model):
        self.model = model
        self.opt = model.opt
        self.opt.vis_layer_names = json.loads(self.opt.vis_layer_names)
        self.vis_layer_names = [name.replace('.','') for name in self.opt.vis_layer_names]
        # self.vis_layer_names = self.opt.vis_layer_names
        self.init_vis_layers()

    def init_vis_layers(self):

        def add_hook(net, root_name):
            if hasattr(net, '_modules'):
                for m_name, module in net._modules.items():
                    # if len(root_name):
                    #     m_name = root_name + '.' + m_name
                    m_name = root_name + m_name
                    if m_name in self.vis_layer_names or (self.opt.vis_all_modules and self.is_sub_modules(m_name)):
                        self.vis_info[m_name] = {}
                        self.vis_info[m_name]['output'] = []
                        self.vis_info[m_name]['grad'] = []
                        self.vis_info[m_name]['cam'] = []
                        self.vis_info[m_name]['vis_img'] = []

                        print('visualize ', m_name)
                        save_output_code = compile('def save_output' + m_name + '(module, input, output): '
                                                                                'vis_info = getattr(self, "vis_info");'
                                                                                'vis_info[\"' + m_name + '\"]["output"].append(output.detach());'
                                                   , "<string>", "exec")
                        func_space = {'self': self}
                        func_space.update(globals())
                        save_output = FunctionType(save_output_code.co_consts[0], func_space, "save_output")

                        module.register_forward_hook(save_output)

                        save_gradient_code = compile(
                            'def save_gradient' + m_name + '(module, input_grad, output_grad): '
                                                           'vis_info = getattr(self, "vis_info");'
                                                           'vis_info[\"' + m_name + '\"]["grad"].append(output_grad[0]);'
                            # 'print(\"' + m_name + '\" + " grad " +  torch.sum(output_grad))'
                            , "<string>", "exec")
                        save_gradient = FunctionType(save_gradient_code.co_consts[0], func_space, "save_gradient")
                        module.register_backward_hook(save_gradient)
                    add_hook(module, m_name)

        self.vis_info = {}
        # for i, m_name in enumerate(self.vis_layer_names):
        #     # m_name = m_name.replace('.','')
        #     self.vis_layer_names[i] = m_name
        #     self.vis_info[m_name] = {}
        #     self.vis_info[m_name]['output'] = []
        #     self.vis_info[m_name]['grad'] = []
        #     self.vis_info[m_name]['cam'] = []
        #     self.vis_info[m_name]['vis_img'] = []
        #
        for net_name in self.model.net_names:
            net = getattr(self.model, 'net_' + net_name)
            add_hook(net, '')

    def cal_grad(self, y, t_label):
        """

        Args:
            model:
            imgs: NHWC
            t_label: target label to be visualized

        Returns:

        """
        model = self.model
        output = y

        one_hots = torch.zeros(output.shape[0], output.shape[1]).cuda(model.opt.gpu_ids[0])
        one_hots[:, t_label] = 1

        ys = torch.sum(one_hots * output)
        model.zero_grad()
        ys.backward()

    def cal_cam(self):
        self.cat_info()
        cams = []
        for key in self.vis_info.keys():
            grads_val = self.vis_info[key]['grad']
            try:
                grads_val -= torch.min(grads_val)
            except:
                raise ValueError('The value of key ', key, ' is not right')

            grads_val = grads_val / torch.max(grads_val)
            target = self.vis_info[key]['output']
            weights = torch.mean(grads_val, dim=(-2, -1), keepdim=True)
            cam = weights * target
            cam = torch.sum(cam, dim=1)
            cam[cam < 0] = 0
            cam = cam / torch.max(cam)
            self.vis_info[key]['cam'] = cam
            cams.append(cam)

        self.cat_info()
        return cams

    def show_cam_on_image(self, imgs, intype):
        '''
        intype: NCHW or NHWC
        '''
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().detach().numpy()

        if intype == 'NCHW':
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            or_imgs = imgs

        imgs = imgs / 255.
        iaa_resize = iaa.Resize({"height": imgs.shape[1], "width": imgs.shape[2]}, interpolation="linear")
        for key in self.vis_info.keys():
            cams = self.vis_info[key]['cam']
            if isinstance(cams, torch.Tensor):
                cams = cams.cpu().detach().numpy()
            cams = np.transpose(cams, (1,2,0))
            vis_imgs = []
            masks = iaa_resize.augment_image(cams) * 255
            masks = np.transpose(masks, (2,0,1))
            masks = masks.astype('uint8')
            for i in range(imgs.shape[0]):
                img = imgs[i]
                
                heatmap = cv2.applyColorMap(masks[i], cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                cam = heatmap + np.float32(img)
                cam = cam / np.max(cam)
                vis_img = np.uint8(255 * cam)
                vis_imgs.append(vis_img)
            self.vis_info[key]['vis_img'] = np.stack(vis_imgs, axis=0)
            self.vis_info[key]['imgs'] = or_imgs

    def reset_info(self):
        for m_name in self.vis_info.keys():
            self.vis_info[m_name] = {}
            self.vis_info[m_name]['output'] = []
            self.vis_info[m_name]['grad'] = []
            self.vis_info[m_name]['cam'] = []
            self.vis_info[m_name]['imgs'] = []

    def cat_info(self):
        for m_name in self.vis_info.keys():
            if isinstance(self.vis_info[m_name]['cam'], list) and len(self.vis_info[m_name]['cam']) > 0:
                self.vis_info[m_name]['cam'] = torch.cat(self.vis_info[m_name]['cam'], dim=0)
            if isinstance(self.vis_info[m_name]['output'], list) and len(self.vis_info[m_name]['output']) > 0:
                self.vis_info[m_name]['output'] = torch.cat(self.vis_info[m_name]['output'], dim=0)
            if isinstance(self.vis_info[m_name]['grad'], list) and len(self.vis_info[m_name]['grad']) > 0:
                self.vis_info[m_name]['grad'] = torch.cat(self.vis_info[m_name]['grad'], dim=0)

    def is_sub_modules(self, name):
        for tmp_name in self.vis_layer_names:
            if name.startswith(tmp_name):
                return True
        return False


class ShowOrImage(object):

    def __init__(self, model = None):
        self.opt = model.opt
        self.model = model

    @staticmethod
    def show_or_image(image_tensor, intype, name_list = [], text_list = [], to_one_img = True, bbox_data = None, mark_list = None, vis_dir = 'buffer/plot_slice_with_path'):
        '''
        intype: NHWC or NCHW
        '''
        images = tensor2im(image_tensor, intype = intype, norm_to_255=True)

        if mark_list is not None:
            for index in mark_list:
                tmp_img = images[index]
                cv2.rectangle(tmp_img, (0,0), (tmp_img.shape[0], tmp_img.shape[1]), color=(0,255,0), thickness=10)
                # tmp_img = tmp_img.get()
                images[index] = tmp_img

        if bbox_data is not None:
            for i, bbox_list in enumerate(bbox_data):
                tmp_img = images[i]
                for bbox in bbox_list:
                    for j, e in enumerate(bbox):
                        if isinstance(e, torch.Tensor) or isinstance(e, np.ndarray):
                            e = e.item()
                        bbox[j] = int(e)
                    cv2.rectangle(tmp_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(0,255,0), 2)
                images[i] = tmp_img

        if to_one_img:
            images = cat_image(images, in_type = 'HWC', out_type='HWC')
            images = images[np.newaxis, :]
            name_list = name_list[:1]
            text_list = text_list[:1]


        for i in range(images.shape[0]):
            save_dir = osp.join(vis_dir, name_list[i] + '.jpg')
            tmp_image = images[i]

            if len(text_list) > 0:
                text_image = np.ones(tmp_image.shape) * 255
                text_image = text_image.astype('uint8')
                text_image = Image.fromarray(text_image)

                tmp_text = text_list[i]

                new_text = ''
                w_num = len(tmp_text)
                line_num = w_num // 30
                if w_num % 30 > 0:
                    line_num += 1
                for i in range(line_num):
                    new_text += tmp_text[30 * i : 30 * (i + 1)] + '\n'

                tmp_text = new_text

                draw = ImageDraw.Draw(text_image)
                fontStyle = ImageFont.truetype(
                    "NotoSansCJK-Bold.ttc", 80, encoding="utf-8")
                draw.text((10, 10), tmp_text, (0,0,0), font=fontStyle)

                text_image = np.array(text_image)
                tmp_image = np.concatenate([tmp_image, text_image], axis=0)

            cv2.imwrite(save_dir, tmp_image)

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.model.eval()
        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)

        one_hot = torch.sum(one_hot.cuda(self.opt.gpu_ids[0]) * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output
