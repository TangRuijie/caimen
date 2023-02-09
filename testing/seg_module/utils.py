import numpy as np
from skimage import transform as sktsf

# 窗宽窗位变换
# 输入：原生CT及目标窗宽窗位
# 输出：0-255的经窗宽窗位转换的CT
def wc_transform(img, wc=None, ww=None):
    if wc != None:
        wmin = (wc*2 - ww) // 2
        wmax = (wc*2 + ww) // 2
    else:
        wmin = img.min()
        wmax = img.max()
    dfactor = 255.0 / (wmax - wmin)
    img = np.where(img < wmin, wmin, img)
    img = np.where(img > wmax, wmax, img)
    img = (img - wmin) * dfactor
    img = img.astype(np.float32)
    d, _, _ = img.shape
    img = sktsf.resize(img, (d, 256, 256), anti_aliasing=True)

    return img