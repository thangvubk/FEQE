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
parser.add_argument('--n_blocks', type=int, default=16)
parser.add_argument('--n_groups', type=int, default=0)
parser.add_argument('--n_convs', type=int, default=0)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--train_path', type=str, default='./data/DIV2K_train_HR')
parser.add_argument('--valid_path', type=str, default='./data/DIV2K_valid_HR_9')
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--loss', type=str, default='l1')
parser.add_argument('--weight_decay', type=float, default=0)
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
    t_lq = tf.placeholder('float32', [None, None, None, 3], name='t_lq')
    t_hq = tf.placeholder('float32', [None, None, None, 3], name='t_hq')

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
    

    if args.loss == 'l1':
        loss = tl.cost.absolute_difference_error(t_sr, t_hq, is_mean=True)
    else:
        # 2) content loss

        CONTENT_LAYER = 'relu5_4'
        vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
        enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(t_sr * 255))
        dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(t_hq * 255))

        content_size = utils2._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
        loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

        # 3) color loss

        enhanced_blur = utils2.blur(t_sr)
        dslr_blur = utils2.blur(t_hq)

        loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)

        # 4) total variation loss

        batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
        tv_y_size = utils2._tensor_size(t_sr[:,1:,:,:])
        tv_x_size = utils2._tensor_size(t_sr[:,:,1:,:])
        y_tv = tf.nn.l2_loss(t_sr[:,1:,:,:] - t_sr[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(t_sr[:,:,1:,:] - t_sr[:,:,:batch_shape[2]-1,:])
        loss_tv = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

        # final loss
        w_content = 10
        w_color = 0.5
        w_tv = 2000
        loss = w_content * loss_content + w_color * loss_color + w_tv * loss_tv

    with tf.variable_scope('l1_regularizer'):
        l2 = 0
        if args.weight_decay != 0:
            for w in tl.layers.get_variables_with_name('Generator', True, True):
                l2 += tf.contrib.layers.l2_regularizer(args.weight_decay)(w)
    loss = loss + l2

    g_vars = tl.layers.get_variables_with_name('Generator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=g_vars)

    #=============PSNR and SSIM================================================
    #t_psnr = tf.image.psnr(t_sr, t_hq, max_val=1.0)
    with tf.variable_scope('PSNR'):
        loss_mse = tf.reduce_sum(tf.pow(t_hq - t_sr, 2))/(100*100*3)/batch_size
        t_psnr = 20 * log10(1.0 / tf.sqrt(loss_mse))

    #t_ssim = tf.image.ssim_multiscale(t_sr, t_hq, max_val=1.0)  

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    saver = tf.train.Saver()
    if args.pretrained_model != '':
        saver.restore(sess, args.pretrained_model)
    #train_writer = tf.summary.FileWriter('test_tb', sess.graph)

    ###=========================Tensorboard=============================###
    writer = SummaryWriter(os.path.join(checkpoint, 'result'))
    tf.summary.FileWriter(os.path.join(checkpoint, 'graph'), sess.graph)
    best_psnr, best_epoch = -1, -1

    ###========================= Training ====================###
    for epoch in range(1, n_epoch + 1):
        ## array_imgs = np.random.permutation(160471)
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
        running_loss = 0

        #==========Ids for shuffling====================
        ids = np.random.permutation(len(train_hq_imgs))

        for idx in tqdm(range(num_batches)):
            aug_idx = random.randint(0,7)
            hq = tl.prepro.threading_data(train_hq_imgs[ids[idx*batch_size:(idx+1)*batch_size]], fn=augment, aug_idx=aug_idx)
            lq = tl.prepro.threading_data(train_lq_imgs[ids[idx*batch_size:(idx+1)*batch_size]], fn=augment, aug_idx=aug_idx)
            #hq = train_hq_imgs[ids[idx*batch_size:(idx+1)*batch_size]] ### use random array
            #lq = train_lq_imgs[ids[idx*batch_size:(idx+1)*batch_size]]
            ###check the type of hq,lq, delete later
            #print("Type hq and lq: {} and {}".format(type(hq),type(lq)))

            [lq, hq] = normalize([lq, hq])

            ## update G
            loss_val, _ = sess.run([loss, g_optim], {t_lq: lq, t_hq: hq})
            running_loss += loss_val
        log = "[*] Epoch: [%2d/%2d], loss: %.8f" % (epoch, n_epoch, running_loss/num_batches)
        print(log)

        writer.add_scalar('Loss', running_loss/num_batches, epoch)

        running_loss = 0
        if (epoch % args.eval_every == 0):
            print('Visualize...')
            num_batches = len(visual_lq_imgs)//4
            for idx in tqdm(range(num_batches)):
                #if idx == 100: break
                lq = visual_lq_imgs[0:4]

                [lq, hq] = normalize([lq, lq])

                #hq_ex = np.expand_dims(hq, axis=0)
                #lq_ex = np.expand_dims(lq, axis=0)

                sr = sess.run(t_sr, {t_lq: lq})
                #print(idx, psnr)
                #sr = np.squeeze(sr_ex)

                if idx == 0:
                    for i in range(4):
                        [lq_i, sr_i, hq_i] = [lq[i], sr[i], hq[i]]
                        [lq_i, sr_i, hq_i] = restore([lq_i, sr_i, hq_i])
                        update_tensorboard(epoch, writer, i-4, lq_i, sr_i, hq_i)


        #=============Valdating==================#
        running_loss = 0
        if (epoch % args.eval_every == 0):
            print('Validating...')
            val_psnr = 0
            num_batches = len(valid_hq_imgs)//batch_size
            for idx in tqdm(range(num_batches)):
                #if idx == 100: break
                hq = valid_hq_imgs[idx*batch_size: (idx+1)*batch_size]
                lq = valid_lq_imgs[idx*batch_size: (idx+1)*batch_size]

                [lq, hq] = normalize([lq, hq])

                #hq_ex = np.expand_dims(hq, axis=0)
                #lq_ex = np.expand_dims(lq, axis=0)

                psnr, loss_val, sr = sess.run([t_psnr, loss, t_sr], {t_lq: lq, t_hq: hq})
                #print(idx, psnr)
                #sr = np.squeeze(sr_ex)

                if idx == 0:
                    for i in range(10):
                        [lq_i, sr_i, hq_i] = [lq[i], sr[i], hq[i]]
                        [lq_i, sr_i, hq_i] = restore([lq_i, sr_i, hq_i])
                        update_tensorboard(epoch, writer, i, lq_i, sr_i, hq_i)

                val_psnr += psnr #compute_PSNR(hr, sr)
                running_loss += loss_val


            val_psnr = val_psnr/num_batches
            avr_loss = running_loss/num_batches
            print('fasdf', val_psnr)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                print('Saving new best model')
                saver.save(sess, os.path.join(checkpoint, 'model.ckpt'))
            print('Validate psnr: %.4fdB. Best: %.4fdB at epoch %d' %(val_psnr, best_psnr, best_epoch))
            writer.add_scalar('Validate PSNR', val_psnr, epoch)
            writer.add_scalar('Validation Loss', avr_loss, epoch)

if __name__ == '__main__':
    train()

