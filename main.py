#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import utils2
import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g
from utils import *
from config import config, log_config
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pdb
import argparse
import vgg
from models import *
import random
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--sample_type', type=str, default='subpixel')
parser.add_argument('--conv_type', type=str, default='default')
parser.add_argument('--body_type', type=str, default='resnet')
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--n_blocks', type=int, default=14)
parser.add_argument('--n_groups', type=int, default=0)
parser.add_argument('--n_convs', type=int, default=0)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--train_path', type=str, default='./data/DIV2K_train_HR')
parser.add_argument('--valid_path', type=str, default='./data/DIV2K_valid_HR_9')
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--alpha_mse', type=float, default=1)
parser.add_argument('--alpha_vgg', type=float, default=0)
parser.add_argument('--alpha_color', type=float, default=0)
parser.add_argument('--visualize', type=lambda x: (str(x).lower() == 'true'), default=True)
args = parser.parse_args()

###====================== HYPER-PARAMETERS ===========================###
batch_size = args.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
checkpoint = args.checkpoint
PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def train():
    ## create folders to save trained model
    tl.files.exists_or_mkdir(checkpoint)

    ###====================== PRE-LOAD DATA ===========================###
    train_hq_npy = os.path.join(config.TRAIN.hq_img_path, 'train_hq.npy')
    train_lq_npy = os.path.join(config.TRAIN.lq_img_path, 'train_lq.npy')
    valid_hq_npy = os.path.join(config.VALID.hq_img_path, 'valid_hq.npy')
    valid_lq_npy = os.path.join(config.VALID.lq_img_path, 'valid_lq.npy')
    visual_lq_npy = os.path.join('data_enhance/HD_res', 'visual_lq.npy')

    if os.path.exists(train_hq_npy) and os.path.exists(train_lq_npy) and os.path.exists(valid_hq_npy) and os.path.exists(valid_lq_npy):
        train_hq_imgs = np.load(train_hq_npy)
        train_lq_imgs = np.load(train_lq_npy)
        valid_hq_imgs = np.load(valid_hq_npy)
        valid_lq_imgs = np.load(valid_lq_npy)
        visual_lq_imgs = np.load(visual_lq_npy)
    else:
        print('Data bins is not created. Creating data bins')
        train_hq_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hq_img_path, regx='.*.jpg', printable=False))
        train_lq_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lq_img_path, regx='.*.jpg', printable=False))
        valid_hq_img_list = sorted(tl.files.load_file_list(path=config.VALID.hq_img_path, regx='.*.jpg', printable=False))
        valid_lq_img_list = sorted(tl.files.load_file_list(path=config.VALID.lq_img_path, regx='.*.jpg', printable=False))
        visual_lq_img_list = sorted(tl.files.load_file_list(path='data_enhance/HD_res', regx='.*.png', printable=False))



        train_hq_imgs = np.array(tl.vis.read_images(train_hq_img_list, path=config.TRAIN.hq_img_path, n_threads=32))
        train_lq_imgs = np.array(tl.vis.read_images(train_lq_img_list, path=config.TRAIN.lq_img_path, n_threads=32))
        valid_hq_imgs = np.array(tl.vis.read_images(valid_hq_img_list, path=config.VALID.hq_img_path, n_threads=16))
        valid_lq_imgs = np.array(tl.vis.read_images(valid_lq_img_list, path=config.VALID.lq_img_path, n_threads=16))
        visual_lq_imgs = np.array(tl.vis.read_images(visual_lq_img_list, path='data_enhance/HD_res', n_threads=16))

        np.save(train_hq_npy, train_hq_imgs)
        np.save(train_lq_npy, train_lq_imgs)
        np.save(valid_hq_npy, valid_hq_imgs)
        np.save(valid_lq_npy, valid_lq_imgs)
        np.save(visual_lq_npy, visual_lq_imgs)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_lq = tf.placeholder('float32', [None, 100, 100, 3], name='t_lq')
    t_hq = tf.placeholder('float32', [None, 100, 100, 3], name='t_hq')

    t_lq_vis = tf.placeholder('float32', [1, None, None, 3], name='t_lq_vis')

    opt = {
        'n_feats': args.n_feats,
        'n_blocks': args.n_blocks,
        'n_groups': args.n_groups,
        'n_convs': args.n_convs,
        'sample_type': args.sample_type,
        'conv_type': args.conv_type,
        'body_type': args.body_type,
        'scale': args.scale
    }
    t_sr = SRGAN_g(t_lq, opt)
    t_sr_vis = SRGAN_g(t_lq_vis, opt, reuse=True)
    #t_sr = resnet_8_32(t_lq)

    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total number of trainable parameters: %d" % total_parameters)

    ####========================== DEFINE TRAIN OPS ==========================###
    #l1_loss = tl.cost.absolute_difference_error(t_sr, t_hq, is_mean=True)
    t_mse_loss = args.alpha_mse*tl.cost.mean_squared_error(t_sr, t_hq, is_mean=True)
    # 2) content loss

    with tf.variable_scope('vgg_loss'):
        CONTENT_LAYER = 'relu5_4'
        vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
        enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(t_sr * 255))
        dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(t_hq * 255))

        t_vgg_loss = args.alpha_vgg*tl.cost.mean_squared_error(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER], is_mean=True) if args.alpha_vgg != 0 else tf.constant(0.0)
    # 3) color loss

    with tf.variable_scope('color_loss'):
        enhanced_blur = utils2.blur(t_sr)
        dslr_blur = utils2.blur(t_hq)
        t_color_loss = args.alpha_color*tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size) if args.alpha_color != 0 else tf.constant(0.0)

    # final loss
    t_loss = t_mse_loss + t_vgg_loss + t_color_loss

    with tf.variable_scope('l1_regularizer'):
        l2 = 0
        if args.weight_decay != 0:
            for w in tl.layers.get_variables_with_name('Generator', True, True):
                l2 += tf.contrib.layers.l2_regularizer(args.weight_decay)(w)
    t_loss = t_loss + l2

    g_vars = tl.layers.get_variables_with_name('Generator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(t_loss, var_list=g_vars)

    #============================PSNR============================================
    # average the entire batch to avoid Inf PSNR in some patches
    with tf.variable_scope('PSNR'):
        loss_mse = tf.reduce_sum(tf.pow(t_hq - t_sr, 2))/(100*100*3)/batch_size
        t_psnr = 20 * log10(1.0 / tf.sqrt(loss_mse))

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    saver = tf.train.Saver()
    if args.pretrained_model != '':
        saver.restore(sess, args.pretrained_model)

    ###=========================Tensorboard=============================###
    writer = SummaryWriter(os.path.join(checkpoint, 'result'))
    tf.summary.FileWriter(os.path.join(checkpoint, 'graph'), sess.graph)
    best_psnr, best_epoch = -1, -1

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
        num_batches = len(train_hq_imgs)//batch_size
        running_loss = np.zeros(4)

        # Ids for shuffling
        ids = np.random.permutation(len(train_hq_imgs))

        for idx in tqdm(range(num_batches)):
            aug_idx = random.randint(0,7)
            hq = tl.prepro.threading_data(train_hq_imgs[ids[idx*batch_size:(idx+1)*batch_size]], fn=augment, aug_idx=aug_idx)
            lq = tl.prepro.threading_data(train_lq_imgs[ids[idx*batch_size:(idx+1)*batch_size]], fn=augment, aug_idx=aug_idx)
            [lq, hq] = normalize([lq, hq])

            ## update G
            loss, mse_loss, vgg_loss, color_loss, _ = sess.run([t_loss, t_mse_loss, t_vgg_loss, t_color_loss, g_optim], {t_lq: lq, t_hq: hq})
            running_loss += [loss, mse_loss, vgg_loss, color_loss]
        avr_loss = running_loss/num_batches
        print("[*] Epoch: [%2d/%2d], loss: %.8f. MSE: %.4f. VGG: %.4f. Color: %.4f" \
              % (epoch, n_epoch, avr_loss[0], avr_loss[1], avr_loss[2], avr_loss[3]))

        writer.add_scalar('Total', avr_loss[0], epoch)
        writer.add_scalar('MSE', avr_loss[1], epoch)
        writer.add_scalar('VGG', avr_loss[2], epoch)
        writer.add_scalar('Color', avr_loss[3], epoch)

        running_loss = 0
        
        if args.visualize:
            print('Visualize...')
            for idx in tqdm(range(len(visual_lq_imgs))):
                lq = visual_lq_imgs[idx]

                [lq] = normalize([lq])
                lq_ex = np.expand_dims(lq, axis=0)

                sr_ex = sess.run(t_sr_vis, {t_lq_vis: lq_ex})
                sr = np.squeeze(sr_ex)
                
                [lq, sr] = restore([lq, sr])
                update_tensorboard(epoch, writer, idx, lq, sr, sr)

        #=============Valdating==================#
        running_loss = np.zeros(4)
        if (epoch % args.eval_every == 0):
            print('Validating...')
            val_psnr = 0
            num_batches = len(valid_hq_imgs)//batch_size
            for idx in tqdm(range(num_batches)):
                #if idx == 100: break
                hq = valid_hq_imgs[idx*batch_size: (idx+1)*batch_size]
                lq = valid_lq_imgs[idx*batch_size: (idx+1)*batch_size]

                [lq, hq] = normalize([lq, hq])
                psnr, loss, mse_loss, vgg_loss, color_loss = sess.run([t_psnr, t_loss, t_mse_loss, t_vgg_loss, t_color_loss], {t_lq: lq, t_hq: hq})

                running_loss += [loss, mse_loss, vgg_loss, color_loss]
                val_psnr += psnr #compute_PSNR(hr, sr)

            #saver.save(sess, os.path.join(checkpoint, 'model_{}.ckpt'.format(epoch)))

            val_psnr = val_psnr/num_batches
            avr_loss = running_loss/num_batches
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                print('Saving new best model')
                saver.save(sess, os.path.join(checkpoint, 'model.ckpt'))
            print('Validate psnr: %.4fdB. Best: %.4fdB at epoch %d' %(val_psnr, best_psnr, best_epoch))
            writer.add_scalar('Val PSNR', val_psnr, epoch)
            writer.add_scalar('Val Total', avr_loss[0], epoch)
            writer.add_scalar('Val MSE', avr_loss[1], epoch)
            writer.add_scalar('Val VGG', avr_loss[2], epoch)
            writer.add_scalar('Val Color', avr_loss[3], epoch)

if __name__ == '__main__':
    train()

