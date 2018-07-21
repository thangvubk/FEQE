#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pdb

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
checkpoint = config.checkpoint

ni = int(np.sqrt(batch_size))

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    tl.files.exists_or_mkdir(checkpoint)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=16)

 
    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [None, None, None, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [None, None, None, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)

    net_g.print_params(False)
    net_g.print_layers()

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    ####========================== DEFINE TRAIN OPS ==========================###
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    loss = tf.losses.absolute_difference(t_target_image, net_g.outputs)

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    ###=========================Tensorboard=============================###
    writer = SummaryWriter(checkpoint)
    best_val_psnr, best_epoch = -1, -1

    ###========================= Training ====================###
    for epoch in range(1, n_epoch + 1):
        ## update learning rate
        if epoch == 1:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)
        if epoch % decay_every == 0:
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        epoch_time = time.time()
        num_batches = len(train_hr_imgs)//batch_size
        running_loss = 0

        for idx in tqdm(range(num_batches)):
            b_imgs_192 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_48 = tl.prepro.threading_data(b_imgs_192, fn=downsample_fn)
            ## update G
            loss_val, _ = sess.run([loss, g_optim], {t_image: b_imgs_48, t_target_image: b_imgs_192})
            running_loss += loss_val
        log = "[*] Epoch: [%2d/%2d], loss: %.8f" % (epoch, n_epoch, running_loss/num_batches)
        print(log)

        writer.add_scalar('Loss', running_loss/num_batches, epoch)
        
        #=============Valdating==================#
        if (epoch % 20 == 0):
            print('Validating...')
            val_psnr = 0
            for idx in tqdm(range(len(valid_hr_imgs))):

                hr = normalize(valid_hr_imgs[idx])
                lr = downsample_fn(hr)
                
                hr_ex = np.expand_dims(hr, axis=0)
                lr_ex = np.expand_dims(lr, axis=0)
                
                loss_val, sr_ex = sess.run([loss, net_g_test.outputs], {t_image: lr_ex, t_target_image: hr_ex})
                sr = np.squeeze(sr_ex)
                
                [lr, sr, hr] = restore([lr, sr, hr])
                update_tensorboard(epoch, writer, idx, lr, sr, hr)
                val_psnr += compute_PSNR(hr, sr)
            
            val_psnr = val_psnr/len(valid_hr_imgs)
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_epoch = epoch
                print('Saving new best model')
                tl.files.save_npz(net_g.all_params, name=checkpoint + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
            print('Validate PSNR: %.4fdB. Best: %.4fdB at epoch %d' %(val_psnr, best_val_psnr, epoch))
            writer.add_scalar('Validate PSNR', val_psnr, epoch)
            writer.add_scalar('Best val PSNR', best_val_psnr, epoch)

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
