import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import random
import imageio
import numpy as np
import pdb

def rgb2y(rgb):
    return np.dot(rgb[...,:3], [65.738/256, 129.057/256, 25.064/256]) + 16

def _augment(x):
    """random flip and rotate aumentation""" 
    aug_idx = random.randint(0,7)

    if (aug_idx>>2)&1 == 1:
        # transpose
        x = x.transpose((1, 0, 2)).copy()
    if (aug_idx>>1)&1 == 1:
        # vertical flip
        x = x[::-1, :, :].copy()
    if aug_idx&1 == 1:
        # horizontal flip
        x = x[:, ::-1, :].copy()

    return x

def get_imgs_fn(file_name, path):
    return imageio.imread(path + file_name)

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=192, hrg=192, is_random=is_random)
    x = _augment(x)
    return x

def downsample_fn(x, scale=4):
    # Downsample then interpolate
    h, w = x.shape[0:2]
    hs, ws = h//scale, w//scale

    x = imresize(x, size=[hs, ws], interp='bicubic', mode=None)
    x = imresize(x, size=[h, w], interp='bicubic', mode=None)
    return x

def transpose(xs):
    for i in range(len(xs)):
        xs[i] = xs[i].transpose(2, 0, 1)
    return xs

def update_tensorboard(epoch, tb, img_idx, lr, sr, hr):
    [lr, sr, hr] = transpose([lr, sr, hr])

    if epoch == 20: #first validation
        tb.add_image(str(img_idx) + '_LR', lr, 0)
        tb.add_image(str(img_idx) + '_HR', hr, 0)
    tb.add_image(str(img_idx) + '_SR', sr, epoch)

def compute_PSNR(out, lbl):
    out = rgb2y(out)
    lbl = rgb2y(lbl)
    out = out.clip(0, 255).round()
    lbl = lbl.clip(0, 255).round()
    diff = out - lbl
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20*np.log10(255/rmse)
    return psnr

def normalize(xs):
    for i in range(len(xs)):
        xs[i] = xs[i]/255
        xs[i] = xs[i].astype(np.float32)
    return xs

def restore(xs):
    for i in range(len(xs)):
        xs[i] = xs[i]*255
        xs[i] = xs[i].clip(0, 255).round()
        xs[i] = xs[i].astype(np.uint8)
    return xs

