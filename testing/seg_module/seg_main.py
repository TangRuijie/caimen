import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .networks.vit_seg_modeling import VisionTransformer as ViT_seg
from .networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .utils import wc_transform

# labeling segmentation mask using Multi-View Fusion
def labeling(CT, model_axial, model_coronal, model_sagittal):
    CT = torch.from_numpy(CT)
    CT = CT.unsqueeze(0)
    CT = CT.cuda().float()

    b, d, h, w = CT.shape
    CT_axial = CT # 1, d, h, w
    CT_coronal = F.interpolate(CT.permute(0, 2, 1, 3), size=(256, 256), mode='bilinear')
    CT_sagittal = F.interpolate(CT.permute(0, 3, 1, 2), size=(256, 256), mode='bilinear')

    axial_res, coronal_res, sagittal_res = [], [], []
    batch_size = 24
    # axial_segmentation
    for idx in range(0, CT_axial.shape[1], batch_size):
        scan = CT_axial[:, idx:idx+batch_size].permute(1, 0, 2, 3)
        output = nn.Sigmoid()(model_axial(scan)).permute(1, 0, 2, 3)
        axial_res.append(output.detach())
    axial_res = torch.cat(axial_res, dim=1)
    # coronal segmentation
    for idx in range(0, CT_coronal.shape[1], batch_size):
        scan = CT_coronal[:, idx:idx+batch_size].permute(1, 0, 2, 3)
        output = nn.Sigmoid()(model_coronal(scan)).permute(1, 0, 2, 3)
        coronal_res.append(output.detach())
    coronal_res = torch.cat(coronal_res, dim=1)
    coronal_res = F.interpolate(coronal_res, size=(d, w), mode='bilinear')
    coronal_res = coronal_res.permute(0, 2, 1, 3)
    # sagittal segmentation
    for idx in range(0, CT_sagittal.shape[1], batch_size):
        scan = CT_sagittal[:, idx:idx+batch_size].permute(1, 0, 2, 3)
        output = nn.Sigmoid()(model_sagittal(scan)).permute(1, 0, 2, 3)
        sagittal_res.append(output.detach())
    sagittal_res = torch.cat(sagittal_res, dim=1)
    sagittal_res = F.interpolate(sagittal_res, size=(d, h), mode='bilinear')
    sagittal_res = sagittal_res.permute(0, 2, 3, 1)
    
    # fusion
    fusion_res = torch.cat([axial_res, coronal_res, sagittal_res], dim=0)
    assert fusion_res.shape == (3, d, h, w)
    fusion_res = fusion_res.mean(dim=0, keepdim=True)
    detect_conf = fusion_res.max().float()
    fusion_res[fusion_res < 0.5] = 0
    fusion_res[fusion_res > 0.5] = 1
    seg_mask = fusion_res[0].cpu().numpy().astype(np.uint8)

    return detect_conf, seg_mask

def load_seg_model(model_name):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    model = ViT_seg(config_vit, img_size=256, num_classes=1).cuda()
    model = torch.nn.DataParallel(model)

    model_dict = torch.load(f"seg_module/model_weights/{model_name}.pth")
    model.load_state_dict(model_dict)

    model.cuda()
    model.eval()

    return model

def ai_segmentation(ct):
    model_a, model_c, model_s = load_seg_model("seg_model_axial"), load_seg_model("seg_model_coronal"), load_seg_model("seg_model_sagittal")
    ct = wc_transform(ct, 0, 400)
    detect_conf, seg_mask = labeling(ct, model_a, model_c, model_s)

    return detect_conf, seg_mask