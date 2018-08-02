#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g
from utils import *
from config import config, log_config
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pdb
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--sample_type', type=str, default='subpixel')
parser.add_argument('--conv_type', type=str, default='default')
parser.add_argument('--body_type', type=str, default='resnet')
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--n_blocks', type=int, default=32)
parser.add_argument('--n_groups', type=int, default=0)
parser.add_argument('--n_convs', type=int, default=0)
parser.add_argument('--n_squeezes', type=int, default=0)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--eval_every', type=int, default=20)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--train_path', type=str, default='./data/DIV2K_train_HR')
parser.add_argument('--valid_path', type=str, default='./data/DIV2K_valid_HR_9')
parser.add_argument('--phase', type=str, default='train')
args = parser.parse_args()

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = args.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
checkpoint = args.checkpoint

def train():
    ## create folders to save trained model
    tl.files.exists_or_mkdir(checkpoint)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_npy = os.path.join(args.train_path, 'train_hr.npy')
    valid_hr_npy = os.path.join(args.valid_path, 'valid_hr.npy')
    valid_lr_npy = os.path.join(args.valid_path, 'X{}_valid_lr.npy'.format(args.scale))
    
    if os.path.exists(train_hr_npy) and os.path.exists(valid_hr_npy) and os.path.exists(valid_lr_npy):
        train_hr_imgs = np.load(train_hr_npy)
        valid_hr_imgs = np.load(valid_hr_npy)
        valid_lr_imgs = np.load(valid_lr_npy)
    else:
        print('Data bins is not created. Creating data bins')
        train_hr_img_list = sorted(tl.files.load_file_list(path=args.train_path, regx='.*.png', printable=False))
        valid_hr_img_list = sorted(tl.files.load_file_list(path=args.valid_path, regx='.*.png', printable=False))
        train_hr_imgs = np.array(tl.vis.read_images(train_hr_img_list, path=args.train_path, n_threads=32))
        valid_hr_imgs = np.array(tl.vis.read_images(valid_hr_img_list, path=args.valid_path, n_threads=16))
        valid_lr_imgs = tl.prepro.threading_data(valid_hr_imgs, fn=downsample_fn, scale=args.scale)
        np.save(train_hr_npy, train_hr_imgs)
        np.save(valid_hr_npy, valid_hr_imgs)
        np.save(valid_lr_npy, valid_lr_imgs)

 
    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_lr = tf.placeholder('float32', [None, None, None, 3], name='t_lr')
    t_hr = tf.placeholder('float32', [None, None, None, 3], name='t_hr')


    opt = {
        'n_feats': args.n_feats,
        'n_blocks': args.n_blocks,
        'n_groups': args.n_groups,
        'n_convs': args.n_convs,
        'n_squeezes': args.n_squeezes,
        'sample_type': args.sample_type,
        'conv_type': args.conv_type,
        'body_type': args.body_type,
        'scale': args.scale
    }
    t_sr = SRGAN_g(t_lr, opt)


    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total number of trainable parameters: %d" % total_parameters)

    ####========================== DEFINE TRAIN OPS ==========================###
    t_loss = tl.cost.absolute_difference_error(t_sr, t_hr, is_mean=True)

    g_vars = tl.layers.get_variables_with_name('Generator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(t_loss, var_list=g_vars)


    #=============PSNR and SSIM================================================
    t_psnr = tf.image.psnr(t_sr, t_hr, max_val=1.0)
    t_ssim = tf.image.ssim_multiscale(t_sr, t_hr, max_val=1.0) 

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    if args.phase == 'pretrain':
        body_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator/body'))
    else: 
        global_saver = tf.train.Saver()
        if args.pretrained_model != '':
            body_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator/body'))
            body_saver.restore(sess, args.pretrained_model)

    ###=========================Tensorboard=============================###
    writer = SummaryWriter(os.path.join(checkpoint, 'result'))
    tf.summary.FileWriter(os.path.join(checkpoint, 'graph'), sess.graph)
    best_score, best_epoch = -1, -1

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

        # ids to shuffle batches
        ids = np.random.permutation(len(train_hr_imgs))

        epoch_time = time.time()
        num_batches = len(train_hr_imgs)//batch_size
        running_loss = 0

        for i in tqdm(range(num_batches)):
            hr = tl.prepro.threading_data(train_hr_imgs[ids[i*batch_size:(i+1)*batch_size]], fn=crop_sub_imgs_fn, is_random=True)
            lr = tl.prepro.threading_data(hr, fn=downsample_fn, scale=args.scale)
            [lr, hr] = normalize([lr, hr])

            ## update G
            loss, _ = sess.run([t_loss, g_optim], {t_lr: lr, t_hr: hr})
            running_loss += loss
        log = "[*] Epoch: [%2d/%2d], loss: %.8f" % (epoch, n_epoch, running_loss/num_batches)
        print(log)

        writer.add_scalar('Loss', running_loss/num_batches, epoch)
        

        #=============Valdating==================#
        running_loss = 0
        if (epoch % args.eval_every == 0):
            print('Validating...')
            val_psnr = 0
            val_ssim = 0
            score = 0
            for i in tqdm(range(len(valid_hr_imgs))):

                hr = valid_hr_imgs[i]
                lr = valid_lr_imgs[i]

                [lr, hr] = normalize([lr, hr])
                
                hr_ex = np.expand_dims(hr, axis=0)
                lr_ex = np.expand_dims(lr, axis=0)
                
                psnr, ssim, loss, sr_ex = sess.run([t_psnr, t_ssim, t_loss, t_sr], 
                                                   {t_lr: lr_ex, t_hr: hr_ex})
                sr = np.squeeze(sr_ex)
                
                #[lr, sr, hr] = restore([lr, sr, hr])
                update_tensorboard(epoch, writer, i, lr, sr, hr)

                val_psnr += psnr #compute_PSNR(hr, sr)
                val_ssim += ssim
                running_loss += loss

                # score referred to https://github.com/aiff22/ai-challenge
                score += (psnr-26.5) + (ssim-0.94)*100
                

            
            val_psnr = val_psnr/len(valid_hr_imgs)
            val_ssim = val_ssim/len(valid_hr_imgs)
            score = score/len(valid_hr_imgs)
            avr_loss = running_loss/len(valid_hr_imgs)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                print('Saving new best model')

                if args.phase == 'pretrain':
                    body_saver.save(sess, os.path.join(checkpoint, 'body.ckpt'))
                else:
                    global_saver.save(sess, os.path.join(checkpoint, 'model.ckpt'))
            print('Validate score: %.4fdB. Best: %.4fdB at epoch %d' %(score, best_score, best_epoch))
            writer.add_scalar('Validate PSNR', val_psnr, epoch)
            writer.add_scalar('Validate SSIM', val_ssim, epoch)
            writer.add_scalar('Validate score', score, epoch)
            writer.add_scalar('Best val score', best_score, epoch)
            writer.add_scalar('Validation Loss', avr_loss, epoch)

if __name__ == '__main__':
    train()

