import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def rgb2y(rgb):
    return np.dot(rgb[...,:3], [65.738/256, 129.057/256, 25.064/256]) + 16

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=192, hrg=192, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    h, w = x.shape[0:2]
    x = imresize(x, size=[h//4, w//4], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def update_tensorboard(epoch, tb, img_idx, lr, sr, hr):
    if epoch < 2:
        tb.add_image(str(img_idx) + '_LR', lr, epoch)
        tb.add_image(str(img_idx) + '_HR', hr, epoch)
    tb.add_image(str(img_idx) + '_SR', sr, epoch)

def compute_PSNR(out, lbl):
    out = rgb2y(out)
    lbl = rgb2y(lbl)
    diff = out - lbl
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20*np.log10(255/rmse)
    return psnr

def normalize(x):
    #for i in range(len(xs)):
        #xs[i] = xs[i]/127.5 - 1
    #return xs
    return x/127.5 - 1

def restore(xs):
    for i in range(len(xs)):
        xs[i] = (xs[i] + 1)*127.5
        xs[i] = xs[i].clip(0, 255).round()
        xs[i] = xs[i].astype(np.uint8)
    return xs

