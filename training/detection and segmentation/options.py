import os
import argparse

class TrainingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="training options")

        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)
        self.parser.add_argument("--lr_scale",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--adam_lr",
                                 type=float,
                                 help="learning rate of adam",
                                 default=1e-4)
        self.parser.add_argument("--sgd_lr",
                                 type=float,
                                 help="learning rate of SGD",
                                 default=1e-1)
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--max_iter",
                                 type=int,
                                 help="maximum iterations of the training",
                                 default=101)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=45)
        self.parser.add_argument("--log_path",
                                 type=str,
                                 help="log path of the model",
                                 default="weights")
        self.parser.add_argument("--local_rank",
                                 type=int,
                                 help="node rank for distributed training",
                                 default=0)
        self.parser.add_argument("--model_id",
                                 type=int,
                                 default=0)
        self.parser.add_argument("--use_adabound",
                                 type=bool,
                                 default=True)
        self.parser.add_argument("--federate",
                                 action="store_true")
        self.parser.add_argument("--visualize",
                                 action="store_true")
        self.parser.add_argument("--post_process",
                                 action="store_true")
        self.parser.add_argument("--use_crop",
                                 action="store_true")
        self.parser.add_argument("--hospital",
                                 type=str)
        self.parser.add_argument("--direction", type=str, default="axial")
        self.parser.add_argument("--test_split",
                                 type=str)
        self.parser.add_argument("--modality", type=str)
        self.parser.add_argument("--pretrain", action="store_true")
        self.parser.add_argument("--gpu", type=str, default="0,1,2,3")
        self.parser.add_argument("--test_hospital", type=str)
        self.parser.add_argument("--loss", type=str, default="bce")
        self.parser.add_argument("--high_sen", action="store_true")
        self.parser.add_argument("--high_spe", action="store_true")
        self.parser.add_argument("--high_value", type=float, default=0.95)
    
    def parse(self):
        return self.parser.parse_args()

class SegmentStasticOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="training options")

        self.parser.add_argument("--model1", type=str)
        self.parser.add_argument("--model2", type=str)
        self.parser.add_argument("--testset", type=str)
        self.parser.add_argument("--modality1", type=str)
        self.parser.add_argument("--modality2", type=str)
        self.parser.add_argument("--direction1", type=str)
        self.parser.add_argument("--direction2", type=str)
        self.parser.add_argument("--location", type=str, default='all')
        self.parser.add_argument("--disease", type=int, default=-1)
    
    def parse(self):
        return self.parser.parse_args()