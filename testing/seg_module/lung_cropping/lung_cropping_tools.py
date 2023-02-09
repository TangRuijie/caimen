import numpy as np
import cv2
import copy
from skimage import measure
import torch
from .deeplab import DeepLab
from skimage import transform as sktsf

def refine_lung_region(imgsSegResult):
    imgs = np.where(imgsSegResult > 0.5,1,0)
    region = measure.label(imgs)
    props = measure.regionprops(region)
    numPix = []
    for ia in range(len(props)):
        numPix += [[props[ia].area, ia]]
    numPix = sorted(numPix, key=lambda x: x[0], reverse=True)
    firstBig, firstindex = numPix[0]
    if len(numPix)>1:
        secondBig, secondindex = numPix[1]
    else:
        secondBig, secondindex = 0,firstindex

    if secondBig < firstBig / 4:
        minx, miny, minz, maxx, maxy, maxz = props[firstindex].bbox  # [minr, maxr),[minc, maxc)
        x, y, z = props[firstindex].coords[0]
        imgs[region != region[x, y, z]] = 0
        newrg = measure.label(imgs)
        newprops = measure.regionprops(newrg)
        bbox = [[minx, maxx], [miny, maxy], [minz, maxz]]
    else:
        minx1, miny1, minz1, maxx1, maxy1, maxz1 = props[firstindex].bbox  # [minr, maxr),[minc, maxc)
        minx2, miny2, minz2, maxx2, maxy2, maxz2 = props[secondindex].bbox  # [minr, maxr),[minc, maxc)

        x1, y1, z1 = props[firstindex].coords[0]
        x2, y2, z2 = props[secondindex].coords[0]

        imgs[region == region[x1, y1, z1]] = 65535
        imgs[region == region[x2, y2, z2]] = 65535
        imgs[imgs != 65535] = 0
        imgs[imgs == 65535] = 1
        bbox = [[min(minx1, minx2), max(maxx1, maxx2)], [min(miny1, miny2), max(maxy1, maxy2)],
                [min(minz1, minz2), max(maxz1, maxz2)]]
    return imgs,bbox

def get_crop_model():
    modelpath = 'seg_module/lung_cropping/model_lobe.pkl'
    tmp_state_dict = torch.load(modelpath, map_location='cpu')['weight']
    state_dict = {k.replace('module.', ''):v for k, v in tmp_state_dict.items()}
    crop_model = DeepLab()
    crop_model.load_state_dict(state_dict)
    crop_model.cuda()
    crop_model.eval()

    return crop_model

def learning_based_lung_cropping(CT, bs=36):
    CT = CT.transpose(1, 2, 0) # h, w, d
    h, w, d = CT.shape
    if h != 512 or w != 512:
        CT = CT.astype(np.float32)
        CT = sktsf.resize(CT, (512, 512, d), anti_aliasing=True)

    model = get_crop_model()
    data = CT
    imgs = []
    imgs1 = []
    for i in range(data.shape[2]):
        im = copy.deepcopy(data[..., i])
        wcenter = -500
        wwidth = 1500
        minvalue = (2 * wcenter - wwidth) / 2.0 + 0.5
        maxvalue = (2 * wcenter + wwidth) / 2.0 + 0.5

        dfactor = 255.0 / (maxvalue - minvalue)

        zo = np.ones(im.shape) * minvalue
        Two55 = np.ones(im.shape) * maxvalue
        im = np.where(im < minvalue, zo, im)
        im = np.where(im > maxvalue, Two55, im)
        im = ((im - minvalue) * dfactor).astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        imgs1.append(im)

        im = im / 255.
        im -= (0.485, 0.456, 0.406)
        im /= (0.229, 0.224, 0.225)
        im = torch.tensor(im).float().permute(2,0,1).unsqueeze(0)
        imgs.append(im)

    res = []
    m_batch = len(imgs) // bs
    if len(imgs) % bs != 0:
        m_batch += 1

    with torch.no_grad():
        for j in range(m_batch):
            x = copy.deepcopy(imgs[j*bs : (j+1)*bs])

            x = torch.cat(x, dim=0).cuda()
            _ , y = model(x)
            y = torch.softmax(y, dim=1)
            y = torch.argmax(y, dim=1).detach().cpu().numpy()
            #print("\033[1;32mBatch Pre Shape :{}\033[0m\n".format(y.shape))
            for one in range(y.shape[0]):
                res.append(y[one].astype(np.uint8))

    tmp = np.stack([im.astype(np.uint8) for im in res],axis=0)
    dilatedtmp = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i, im in enumerate(tmp):
        im = copy.deepcopy(im)
        dilated = cv2.dilate(im, kernel, 5)
        dilatedtmp.append(dilated)
    dilatedtmp = np.stack(dilatedtmp,axis = 0)
    dilatedtmp, bbox = refine_lung_region(dilatedtmp)
    z1, z2 = bbox[0]
    y1, y2 = bbox[1]
    x1, x2 = bbox[2]

    CT = CT[y1:y2, x1:x2, z1:z2]
    CT = np.transpose(CT, (2, 0, 1)) # d, h, w

    return CT