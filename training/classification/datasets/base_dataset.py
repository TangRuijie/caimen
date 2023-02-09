"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""

import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from util.datasets.rand_augumentation import RandAugment
from torch.utils.data._utils.collate import default_collate
from abc import ABC, abstractmethod
from imgaug import augmenters as iaa
from torchvision.transforms.functional import normalize
import torch

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, data_type):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.data_type = data_type

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    def next_epoch(self):
        pass

    def get_stable_ids(self):
        tmp_list = []
        for sid in self.opt.stable_dict.keys():
            if self.opt.stable_dict[sid] > self.opt.stable_count:
                tmp_list.append(sid)
        return tmp_list

    @staticmethod
    def norm_data(x, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), ctype = 'CHW', norm_to_one = True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.copy())
        x = x.float()

        if ctype == 'HWC':
            if len(x.shape) == 3:
                x = x.permute(2,0,1)
            elif len(x.shape) == 4:
                x = x.permute(0, 3, 1, 2)

        if norm_to_one: 
            x = x / 255.
        # if x.shape[0] != 3 and x.shape[0] != 4:
        #     assert False, 'The dim is not right'

        x = BaseDataset.normalize(x, mean, std)
        return x

    @staticmethod
    def collect_fn(batch):
        return default_collate(batch)

    @staticmethod
    def normalize(tensor, mean, std, inplace=False):
        """Normalize a tensor image with mean and standard deviation.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        See :class:`~torchvision.transforms.Normalize` for more details.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # if not _is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if len(tensor.shape) == 3:
            tmp_mean = mean[:, None, None]
            tmp_std = std[:, None, None]
        elif len(tensor.shape) == 4:
            tmp_mean = mean[None, :, None, None]
            tmp_std = std[None, :, None, None]
        else:
            raise ValueError()

        tensor.sub_(tmp_mean).div_(tmp_std)
        return tensor


    @staticmethod
    def get_aug_transform(opt, preprocess='', in_type = 'CHW', out_type = 'CHW', rand_aug = False, rand_n = -1, rand_m = -1, keep_same = True):
        '''
        data_type: HW, CHW, HWC, NCHW, NHWC
        '''
        transform_list = []
        if preprocess == '':
            preprocess = opt.preprocess

        if 'resize' in preprocess:
            t = iaa.Resize({"height": opt.load_size, "width": opt.load_size}, interpolation='linear')
            transform_list.append(t)

        if 'c_crop' in preprocess:
            t = iaa.CropToFixedSize(width=opt.crop_size, height=opt.crop_size, position='center')
            transform_list.append(t)

        elif 'u_crop' in preprocess:
            t = iaa.CropToFixedSize(width=opt.crop_size, height=opt.crop_size, position='uniform')
            transform_list.append(t)

        if 'scale' in preprocess:
            t = iaa.Affine(scale={"x": opt.scale_per_x, "y": opt.scale_per_y})
            t = iaa.Sometimes(opt.aug_ratio, t)
            transform_list.append(t)

        if 'translate' in preprocess:
            t = iaa.Affine(translate_px={"x": opt.translate_pix_x, "y": opt.translate_pix_y})
            t = iaa.Sometimes(opt.aug_ratio, t)
            transform_list.append(t)

        if 'rotate' in preprocess:
            t = iaa.Affine(rotate=opt.rotate_der)
            t = iaa.Sometimes(opt.aug_ratio, t)
            transform_list.append(t)

        if 'shear' in preprocess:
            t = iaa.Affine(rotate=opt.shear_der)
            t = iaa.Sometimes(opt.aug_ratio, t)
            transform_list.append(t)

        if 'elastic' in preprocess:
            t = iaa.ElasticTransformation(alpha=opt.elastic_alpha, sigma=0.25)
            t = iaa.Sometimes(opt.aug_ratio, t)
            transform_list.append(t)

        if 'flip' in preprocess:
            t = iaa.Fliplr(opt.flip_rate)
            transform_list.append(t)

        if 'contrast' in preprocess:
            t = iaa.SigmoidContrast(gain=opt.contrast_gain, cutoff=opt.contrast_cutoff)
            t = iaa.Sometimes(opt.aug_ratio, t)
            transform_list.append(t)

        if 'clane' in preprocess:
            t = iaa.CLAHE(clip_limit=opt.clane_limit)
            t = iaa.Sometimes(opt.aug_ratio, t)
            transform_list.append(t)

        if 'noise' in preprocess:
            t = iaa.AdditiveGaussianNoise(scale=opt.noise_scale * 255)
            t = iaa.Sometimes(opt.aug_ratio, t)
            transform_list.append(t)

        seq = iaa.Sequential(transform_list)

        def img_trans(img):

            def input_init(x):
                if in_type == 'CHW':
                    x = np.transpose(x,(1,2,0))
                elif in_type == 'NCHW':
                    x = np.transpose(x,(0,2,3,1))
                if rand_aug:
                    x = Image.fromarray(x)
                    t = RandAugment(n = rand_n, m = rand_m)
                    x = t(x)
                    x = np.array(x)
                return x

            img = input_init(img)

            if in_type.startswith('N'):
                if keep_same:
                    tmp = seq._to_deterministic()
                    tmp_list = []
                    for i in range(img.shape[0]):
                        tmp_list.append(tmp.augment_image(img[i]))
                    img = np.stack(tmp_list, axis=0)
                else:
                    img = seq.augment_images(images = img)
            else:
                img = seq.augment_image(image = img)

            if out_type == 'CHW':
                img = np.transpose(img, (2,0,1))
            elif out_type == 'NCHW':
                img = np.transpose(img, (0,3,1,2))

            return img

        return lambda x: img_trans(x)

    @staticmethod
    def get_torch_transform(opt, preprocess = '', in_type = 'CHW', out_type = 'CHW', rand_aug = False, rand_n = -1, rand_m = -1):
        if preprocess == '':
            preprocess = opt.preprocess
        if rand_n < 0:
            rand_n = opt.rand_n
        if rand_m < 0:
            rand_m = opt.rand_m

        transform_list = []

        if 'resize' in preprocess:
            t = transforms.Resize(size=(opt.load_size, opt.load_size))
            transform_list.append(t)

        if 'c_crop' in preprocess:
            t = transforms.CenterCrop(size=(opt.crop_size, opt.crop_size))
            transform_list.append(t)

        elif 'u_crop' in preprocess:
            t = transforms.RandomCrop(size=(opt.crop_size, opt.crop_size))
            transform_list.append(t)

        affine_params = {'degrees': 0}
        af_num = 0

        if 'rotate' in preprocess:
            af_num += 1
            affine_params['degrees'] = opt.rotate_der

        if 'scale' in preprocess:
            af_num += 1
            affine_params['scale'] = (opt.scale_per_x[0], opt.scale_per_y[1])

        if 'translate' in preprocess:
            af_num += 1
            affine_params['translate'] = (opt.translate_pix_x[1]/opt.crop_size, opt.translate_pix_y[1]/opt.crop_size)

        if 'shear' in preprocess:
            af_num += 1
            affine_params['shear'] = opt.shear_rate

        if af_num > 0:
            t = transforms.RandomAffine(**affine_params)
            # t = transforms.RandomAffine(degrees=0)
            transform_list.append(t)

        if 'flip' in preprocess:
            t = transforms.RandomHorizontalFlip(p=0.5)
            transform_list.append(t)

        img_aug = transforms.Compose(transform_list)

        if rand_aug:
            img_aug.transforms.insert(0, RandAugment(n = rand_n, m = rand_m))

        def im_t(x):
            if in_type == 'CHW':
                x = np.transpose(x, (1,2,0))
            x = x.astype('uint8')
            x = Image.fromarray(x)
            return x

        if out_type == 'CHW':
            trans = lambda x: np.transpose(img_aug(im_t(x)), (2,0,1))
        else:
            trans = lambda x: img_aug(im_t(x))
        return trans

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
