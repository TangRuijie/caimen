import torch
import torch.nn as nn

def bce_loss(predict, target):
    loss = nn.BCELoss()

    return loss(predict, target)

def dice_loss(predict, target):
    epsilon = 1
    assert predict.size() == target.size(), "the size of predict and target must be equal."
    num = predict.size(0)
    
    pre = predict.contiguous().view(num, -1)
    tar = target.contiguous().view(num, -1)
    
    intersection = (pre * tar).sum(1)
    union = (pre + tar).sum(1)

    score = (2*intersection + epsilon) / (union + epsilon)

    score = 1 - score.sum() / num
    
    return score