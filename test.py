from __future__ import division
import argparse
import numpy as np
import time, math, glob
import scipy.misc
import os
import imageio
import pdb
import tensorlayer as tl
import tensorflow as tf
from model import *
from utils import *

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_path", type=str, default="checkpoint/FEQE/model.ckpt", help="model path")
parser.add_argument('--save_path', type=str, default='results')
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")

parser.add_argument('--downsample_type', type=str, default='desubpixel')
parser.add_argument('--upsample_type', type=str, default='subpixel')
parser.add_argument('--conv_type', type=str, default='default')
parser.add_argument('--body_type', type=str, default='resnet')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of convolution feats')
parser.add_argument('--n_blocks', type=int, default=20,
                    help='number of residual block if body_type=resnet')
parser.add_argument('--n_groups', type=int, default=0,
                    help='number of residual group if body_type=res_in_res')
parser.add_argument('--n_convs', type=int, default=0,
                    help='number of conv layers if body_type=conv')
parser.add_argument('--n_squeezes', type=int, default=0,
                    help='number of squeeze blocks if body_type=squeeze')

parser.add_argument('--scale', type=int, default=4)

args = parser.parse_args()

print('############################################################')
print('# Image Super Resolution - PIRM2018 - TEAM_ALEX            #')
print('# Implemented by Thang Vu, thangvubk@gmail.com             #')
print('############################################################')
print('')
print('_____________YOUR SETTINGS_____________')
for arg in vars(args):
    print("%20s: %s" %(str(arg), str(getattr(args, arg))))
print('')

def main():
    #==================Data==================================
    print('Loading data...')
    test_hr_path = os.path.join('./data/test_benchmark', args.dataset)
    hr_paths = glob.glob(os.path.join(test_hr_path, '*.png'))
    hr_paths.sort()

    #=================Model===================================
    print('Loading model...')
    t_lr = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    t_hr = tf.placeholder('float32', [1, None, None, 3], name='label_image')

    opt = {
        'n_feats': args.n_feats,
        'n_blocks': args.n_blocks,
        'n_groups': args.n_groups,
        'n_convs': args.n_convs,
        'n_squeezes': args.n_squeezes,
        'downsample_type': args.downsample_type,
        'upsample_type': args.upsample_type,
        'conv_type': args.conv_type,
        'body_type': args.body_type,
        'scale': args.scale
    }
    t_sr = FEQE(t_lr, opt)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)

    #=================result=================================
    save_path = os.path.join(args.save_path, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    psnr_avr = 0
    for i, _ in enumerate(hr_paths):
        print('processing image %d' %i)
        hr_org = imageio.imread(hr_paths[i])
        lr = downsample_fn(hr_org)
        [hr, lr] = normalize([hr_org, lr])

        lr = lr[np.newaxis, :, :, :]
        hr = hr[np.newaxis, :, :, :]

        [sr] = sess.run([t_sr], {t_lr: lr, t_hr: hr})
        sr = np.squeeze(sr)

        [sr] = restore([sr])
        
        sr = sr[args.scale:-args.scale, args.scale:-args.scale]
        hr_org = hr_org[args.scale:-args.scale, args.scale:-args.scale]
        psnr_avr += compute_PSNR(sr, hr_org)
        scipy.misc.imsave(os.path.join(save_path, os.path.basename(hr_paths[i])), sr)

    print('Average PSNR: %.4f' %(psnr_avr/len(hr_paths)))
    print('Finish')

if __name__ == '__main__':
    main()
