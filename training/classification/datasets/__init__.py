"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
import torch.multiprocessing as multiprocessing
from multiprocessing import Manager
from .base_dataset import BaseDataset
# from .buffer_loader import BufferDataLoader
from torch.utils.data import DataLoader
# from .torch131_loader import DataLoader
import numpy as np

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt,data_type='train'):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from datasets import create_dataset
        >>> dataset = create_dataset(opt)
    """
    try:
        data_loader = CustomDatasetDataLoader(opt,data_type)
        dataset = data_loader.load_data()
        return dataset
    except:
        return []


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt,data_type):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        if data_type == 'train':
            dataset_class = find_dataset_using_name(opt.dataset_mode)
            self.dataset = dataset_class(opt,data_type)
            opt.dataset = self
            batch_size = opt.batch_size
            num_threads = opt.num_threads
        else:
            dataset_class = find_dataset_using_name(opt.v_dataset_mode)
            self.dataset = dataset_class(opt,data_type)
            opt.v_dataset = self
            batch_size = opt.v_batch_size
            # if self.opt.l_state == 'train':
            #     # num_threads = opt.num_threads//2
            #     num_threads = 3
            # else:
            num_threads = opt.num_threads

        print("dataset [%s] was created" % type(self.dataset).__name__)

        if opt.buffer_loader:
            self.dataloader = BufferDataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(num_threads),
                buffer_size=opt.loader_buffer_size)
        else:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=not opt.serial_batches,
                collate_fn=self.dataset.collect_fn,
                num_workers=int(num_threads))

        if opt.l_state == 'train' and data_type == 'train' and self.opt.valid_freq <= 0:
            self.opt.valid_freq = max(int(self.opt.valid_freq_ratio * len(self.dataloader)), 1)


        if self.opt.l_state == 'train' and data_type == 'valid':
            if self.opt.train_id_list is not None and self.opt.valid_id_list is not None:
                train_id_list = self.opt.train_id_list
                valid_id_list = self.opt.valid_id_list
                assert len(set(train_id_list) & set(valid_id_list)) == 0, 'the datasets overlap ' + str(len((set(train_id_list)&set(valid_id_list))))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        if self.opt.max_dataset_size == float('inf'):
            return len(self.dataset)
        else:
            return min(len(self.dataset), int(self.opt.max_dataset_size))

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
