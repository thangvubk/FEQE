#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import math
from pixel_deshuffle import DeSubpixelConv2d

def init(in_feats, kernel_size=3):
    std = 1./math.sqrt(in_feats*(kernel_size**2))
    return tf.random_uniform_initializer(-std, std)

def conv(n, in_feats, out_feats, kernel_sizes=(3, 3), strides=(1, 1), act=None, conv_type='default', name='conv'):
    with tf.variable_scope(name):
        if conv_type == 'default':
            n = Conv2d(n, out_feats, kernel_sizes, strides, act=act, W_init=init(in_feats), b_init=init(in_feats))

        elif conv_type == 'depth_wise':
            n = DepthwiseConv2d(n, kernel_sizes, strides, act=tf.nn.relu, W_init=init(in_feats), 
                                b_init=init(in_feats), name='depthwise')
            n = Conv2d(n, out_feats, (1, 1), (1, 1), act=act, 
                       W_init=init(in_feats, kernel_sizes[0]), 
                       b_init=init(in_feats, kernel_sizes[0]), name='conv')

        else:
            raise Exception('Unknown conv type', conv_type)
    return n

def downsample(n, n_feats, conv_type='default', sample_type='subpixel'):
    if sample_type == 'subpixel':
        n = conv(n, 3, n_feats//4, act=None, conv_type=conv_type, name='head/conv1')
        n = DeSubpixelConv2d(n, 2, name='pixel_deshuffle/1')
        n = conv(n, n_feats, n_feats//4, act=None, conv_type=conv_type, name='head/pds1')
        n = DeSubpixelConv2d(n, 2, name='pixel_deshuffle/2')

    elif sample_type == 'deconv':
        n = conv(n, 3, n_feats, strides=(2, 2), act=tf.nn.relu, name='head/deconv1')
        n = conv(n, n_feats, n_feats, strides=(2, 2), act=tf.nn.relu, name='head/deconv2')

    else: 
        raise Exception('Unknown sample_type', sample_type)
    return n

def upsample(n, n_feats, conv_type='default', sample_type='subpixel'):
    if sample_type == 'subpixel':
        n = conv(n, n_feats, n_feats*4, act=None, conv_type=conv_type, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, name='pixelshufflerx2/1')

        n = conv(n, n_feats, n_feats*4, act=None, conv_type=conv_type, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, name='pixelshufflerx2/2')

    elif sample_type == 'deconv':
        n = DeConv2d(n, n_feats, n_feats, strides=(2, 2), act=tf.nn.relu, W_init=init(), b_init=init())
        n = DeConv2d(n, n_feats, n_feats, strides=(2, 2), act=tf.nn.relu, W_init=init(), b_init=init())
    else:
        raise Exception('Unknown sample_type', sample_type)
    return n

def res_block(n, n_feats, conv_type='default', name='res_block'):
    with tf.variable_scope(name):
        res = conv(n, n_feats, n_feats, act=tf.nn.relu, conv_type=conv_type, name='conv1')
        res = conv(res, n_feats, n_feats, act=None, conv_type=conv_type, name='conv2')
        n = ElementwiseLayer([n, res], tf.add, name='res_add')
    return n

def res_group(x, n_feats, n_blocks, conv_type='default', name='res_group'):
    with tf.variable_scope(name):
        res = x
        for i in range(n_blocks):
            res = res_block(res, n_feats, conv_type=conv_type, name='res_block%d' %i)
        x = ElementwiseLayer([x, res], tf.add, name='add')
    return x

def SRGAN_g(t_image, opt):

    sample_type = opt['sample_type']
    conv_type = opt['conv_type']
    n_feats = opt['n_feats']
    n_blocks = opt['n_blocks']
    n_groups = opt['n_groups']

    with tf.variable_scope("SRGAN_g") as vs:
        # normalize input (0, 1) -> (-127.5, 127.5)
        t_image = (t_image-0.5)*255
        x = InputLayer(t_image, name='in')

        #===========Downsample==============
        x = downsample(x, n_feats, conv_type, sample_type)
        res = x

        #============Residual=================
        for i in range(n_groups):
            res = res_group(res, n_feats, n_blocks, conv_type=conv_type, name='res_block%d' %i)

        res = conv(res, n_feats, n_feats, act=None, conv_type=conv_type, name='end_res')
        x = ElementwiseLayer([x, res], tf.add, name='add')
        
        #=============Upsample==================
        x = upsample(x, n_feats, conv_type, sample_type)

        x = conv(x, n_feats, 3, act=None, conv_type=conv_type, name='out')
        outputs = x.outputs/255 + 0.5
        return outputs


