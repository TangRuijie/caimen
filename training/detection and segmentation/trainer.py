import torch
import torch.distributed as distrib
import torch.utils.data.distributed
import torch.optim as optim
import numpy as np 
from options import TrainingOptions
from dataset import Dataset as slice_dataset
from fusion_dataset import Dataset as scan_dataset
from utils import readlines
import torch.nn.functional as F
from losses import dice_loss, bce_loss
from adabound import AdaBound
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch.nn as nn

import json
import os

# FedAvg: average network weights
def average_gradients(model, epoch_degree):
    e = epoch_degree
    e_ = torch.tensor([epoch_degree]).cuda()
    distrib.all_reduce(e_, op=distrib.ReduceOp.SUM)
    for param in model.parameters():
        if param.requires_grad:
            param.data *= e
            distrib.all_reduce(param.data, op=distrib.ReduceOp.SUM)
            param.data /= e_.float()

opt = TrainingOptions()
args = opt.parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

##########################################
# For reproducibility
#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed_all(0)

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
##########################################

distrib.init_process_group(backend="nccl")
local_rank = distrib.get_rank()
torch.cuda.set_device(args.local_rank)

base_path = "DATA/segmentation"

ce_trainset = ['sx', 'cc', 'zs', 'sc', 'gy', 'zd', 'xa', 'qd']
pl_trainset = ['sx', 'cc', 'sz', 'zs', 'nm', 'sc', 'gy', 'gzzl', 'zd', 'fj', 'xa', 'ln', 'cq', 'qd']

# use different datasets for different hosts
if args.direction == "axial":
    if args.modality in ['trans', 'PL']:
        if args.hospital == 'all':
            train_filenames = []
            neg_filenames = []
            for hospital in pl_trainset:
                train_filenames += readlines(os.path.join(base_path, "data", "PL", f"{hospital.upper()}", "label", "train", f"{args.direction}", "pos.txt"))
                neg_filenames += readlines(os.path.join(base_path, "data", "PL", f"{hospital.upper()}", "label", "train", f"{args.direction}", "neg.txt"))
        else:
            train_filenames = readlines(os.path.join(base_path, "data", "PL", f"{args.hospital.upper()}", "label", "train", f"{args.direction}", "pos.txt"))
            neg_filenames = readlines(os.path.join(base_path, "data", "PL", f"{args.hospital.upper()}", "label", "train", f"{args.direction}", "neg.txt"))
    else:
        if args.hospital == 'all':
            train_filenames = []
            neg_filenames = []
            for hospital in ce_trainset:
                train_filenames += readlines(os.path.join(base_path, "data", "CE", f"{hospital.upper()}", "label", "train", f"{args.direction}", "pos.txt"))
                neg_filenames += readlines(os.path.join(base_path, "data", "CE", f"{hospital.upper()}", "label", "train", f"{args.direction}", "neg.txt"))
        else:
            train_filenames = readlines(os.path.join(base_path, "data", "CE", f"{args.hospital.upper()}", "label", "train", f"{args.direction}", "pos.txt"))
            neg_filenames = readlines(os.path.join(base_path, "data", "CE", f"{args.hospital.upper()}", "label", "train", f"{args.direction}", "neg.txt"))
else:
    if args.modality in ["trans", "PL"]:
        if args.hospital == 'all':
            train_filenames = []
            neg_filenames = []
            for hospital in pl_trainset:
                train_filenames += readlines(os.path.join(base_path, "data", "PL", f"{hospital.upper()}", "label", "train", f"{args.direction}", "pos.txt"))[::2]
                neg_filenames += readlines(os.path.join(base_path, "data", "PL", f"{hospital.upper()}", "label", "train", f"{args.direction}", "neg.txt"))[::10]
        else:
            train_filenames = readlines(os.path.join(base_path, "data", "PL", f"{args.hospital.upper()}", "label", "train", f"{args.direction}", "pos.txt"))[::2]
            neg_filenames = readlines(os.path.join(base_path, "data", "PL", f"{args.hospital.upper()}", "label", "train", f"{args.direction}", "neg.txt"))[::10]
    else:
        if args.hospital == 'all':
            train_filenames = []
            neg_filenames = []
            for hospital in ce_trainset:
                train_filenames += readlines(os.path.join(base_path, "data", "CE", f"{hospital.upper()}", "label", "train", f"{args.direction}", "pos.txt"))[::2]
                neg_filenames += readlines(os.path.join(base_path, "data", "CE", f"{hospital.upper()}", "label", "train", f"{args.direction}", "neg.txt"))[::10]
        else:
            train_filenames = readlines(os.path.join(base_path, "data", "CE", f"{args.hospital.upper()}", "label", "train", f"{args.direction}", "pos.txt"))[::2]
            neg_filenames = readlines(os.path.join(base_path, "data", "CE", f"{args.hospital.upper()}", "label", "train", f"{args.direction}", "neg.txt"))[::10]
train_filenames = train_filenames + neg_filenames

if args.modality in ["trans", "PL"]:
    if args.hospital == 'all':
        test_filenames = []
        normal_filenames = []
        for hospital in pl_trainset:
            test_filenames += readlines(os.path.join(base_path, "data", "PL", hospital.upper(), "label", "val", "cls.txt"))
            normal_filenames += readlines(os.path.join(base_path, "data", "PL", hospital.upper(), "label", "val", "neg.txt"))
    else:
        test_filenames = readlines(os.path.join(base_path, "data", "PL", args.hospital.upper(), "label", "val", "cls.txt"))
        normal_filenames = readlines(os.path.join(base_path, "data", "PL", args.hospital.upper(), "label", "val", "neg.txt"))
    normalset = scan_dataset(normal_filenames, is_train=False, use_crop=True)
    normalloader = torch.utils.data.DataLoader(normalset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
elif args.modality == "CE":
    if args.hospital == 'all':
        test_filenames = []
        for hospital in ce_trainset:
            test_filenames += readlines(os.path.join(base_path, "data", "CE", hospital.upper(), "label", "val", "cls.txt"))
    else:
        test_filenames = readlines(os.path.join(base_path, "data", "CE", f"{args.hospital.upper()}", "label", "val", "cls.txt"))

trainset = slice_dataset(train_filenames, base_path, is_train=True, direction=args.direction, use_crop=args.use_crop)
testset = scan_dataset(test_filenames, is_train=False, use_crop=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 1
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))
model = ViT_seg(config_vit, img_size=256, num_classes=1).cuda()

if args.federate:
    fml_str = "fml_"
    sub_dir = "fml/"
else:
    fml_str = ""
    sub_dir = "standalone/"
if os.path.exists(os.path.join(args.log_path, f"{sub_dir}models_{fml_str}{args.hospital}_{args.modality}_{args.direction}_{args.loss}", "weights_latest", "model.pth")):
    model = torch.nn.DataParallel(model)
    model_path = os.path.join(args.log_path, f"{sub_dir}models_{fml_str}{args.hospital}_{args.modality}_{args.direction}_{args.loss}", "weights_latest", "model.pth")
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
else:
    if args.modality == "trans":
        model = torch.nn.DataParallel(model)
        model_path = os.path.join(args.log_path, f"{sub_dir}models_{fml_str}zd_CE_{args.direction}_{args.loss}", "weights_latest", "model.pth")
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)
    else:
        model.load_from(weights=np.load("model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"))
        model = torch.nn.DataParallel(model)
parameters_to_train = [p for p in model.parameters() if p.requires_grad]

if args.use_adabound:
    optimizer = AdaBound(parameters_to_train, lr=args.adam_lr*args.lr_scale, final_lr=args.sgd_lr*args.lr_scale)
else:
    optimizer = optim.Adam(parameters_to_train, args.adam_lr*args.lr_scale)
model.train()

if os.path.exists(f"log/{args.direction}/{fml_str}{args.hospital}_{args.modality}_{args.loss}.json"):
    with open(f"log/{args.direction}/{fml_str}{args.hospital}_{args.modality}_{args.loss}.json", "r") as f:
        exp_record = json.load(f)
else:
    if args.modality == "CE":
        exp_record = {"epoch": 0, "max_dice": 0}
    else:
        exp_record = {"epoch": 0, "max_dice": 0, "max_auc": 0}
max_dice = exp_record["max_dice"]
start_epoch = exp_record["epoch"]

if args.federate:
    if local_rank == 0:
        average_gradients(model, 1)
    else:
        average_gradients(model, 0)

for epoch in range(start_epoch+1, args.max_iter):
    print(f"start {epoch}th training...")
    for ii, (id, CT, mask) in enumerate(trainloader):
        CT, mask = CT.cuda().float(), mask.cuda().float()

        pred = model(CT)
        pred = nn.Sigmoid()(pred)
        loss = bce_loss(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.federate:
        average_gradients(model, len(train_filenames)) # this will block processes finishing one epoch until all processes finish one epoch
    adam_lr_ = args.adam_lr * (1.0 - epoch / args.max_iter) ** 0.9
    sgd_lr_ = args.sgd_lr * (1.0 - epoch / args.max_iter) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = adam_lr_
        param_group['final_lr'] = sgd_lr_

    check_iter = 5
    if args.direction != "axial":
        check_iter = 10
    if epoch % check_iter == 0 and epoch >= 70:
        model.eval()
        avg_dice = 0
        count = 0
        
        with torch.no_grad():
            for ii, (cls_label, CT, mask) in enumerate(testloader):
                CT, mask = CT.cuda().float(), mask.cuda().float()
                _, d, _, _ = CT.shape
                res = []
                if args.direction == 'coronal':
                    CT = CT.permute(0, 2, 1, 3)
                    CT = F.interpolate(CT, size=(256, 256), mode='bilinear')
                elif args.direction == 'sagittal':
                    CT = CT.permute(0, 3, 1, 2)
                    CT = F.interpolate(CT, size=(256, 256), mode='bilinear')

                bs = 20
                idx = 0
                while idx < CT.shape[1]:
                    scan = CT[:, idx:idx+bs].permute(1, 0, 2, 3)
                    x = model(scan)
                    x = nn.Sigmoid()(x).permute(1, 0, 2, 3)
                    res.append(x.detach())
                    idx += bs
                res = torch.cat(res, dim=1)

                if args.direction == 'coronal':
                    res = F.interpolate(res, size=(d, 256), mode="bilinear")
                    res = res.permute(0, 2, 1, 3)
                elif args.direction == 'sagittal':
                    res = F.interpolate(res, size=(d, 256), mode="bilinear")
                    res = res.permute(0, 2, 3, 1)
                res[res < 0.5] = 0
                res[res >= 0.5] = 1
                dice = 1 - dice_loss(res, mask)
                avg_dice += dice
                count += 1
        
        current_dice = avg_dice/count
        print(f"current_dice:{current_dice}")
        
        save_folder = os.path.join(args.log_path, f"{sub_dir}models_{fml_str}{args.hospital}_{args.modality}_{args.direction}_{args.loss}", "weights_latest")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "model.pth")
        to_save = model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("optim"))
        torch.save(optimizer.state_dict(), save_path)

        if current_dice >= max_dice:
            max_dice = current_dice
            save_folder = os.path.join(args.log_path, f"{sub_dir}models_{fml_str}{args.hospital}_{args.modality}_{args.direction}_{args.loss}", "weights_best")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_path = os.path.join(save_folder, "model.pth")
            to_save = model.state_dict()
            torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("optim"))
            torch.save(optimizer.state_dict(), save_path)
        model.train()
    exp_record["epoch"] = epoch
    exp_record["max_dice"] = float(max_dice)
    with open(f"log/{args.direction}/{fml_str}{args.hospital}_{args.modality}_{args.loss}.json", "w") as f:
        json.dump(exp_record, f)
    save_folder = os.path.join(args.log_path, f"{sub_dir}models_{fml_str}{args.hospital}_{args.modality}_{args.direction}_{args.loss}", "weights_latest")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, "model.pth")
    to_save = model.state_dict()
    torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("optim"))
    torch.save(optimizer.state_dict(), save_path)