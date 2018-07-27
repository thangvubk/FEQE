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

class NormalizeLayer(Layer):
    def __init__(self, prev_layer, mean, std, name='normalize_layer'):
        Layer.__init__(self, prev_layer=prev_layer, name=name)

        self.inputs = prev_layer.outputs
        self.outputs = (self.inputs - mean)*std

        self.all_layers.append(self.outputs)

class RestoreLayer(Layer):
    def __init__(self,prev_layer, mean, std, name='restore_layer'):
        Layer.__init__(self, prev_layer=prev_layer, name=name)

        self.inputs = prev_layer.outputs
        self.outputs = (self.inputs/std) + mean

        self.all_layers.append(self.outputs)

def conv(x, in_feats, out_feats, kernel_sizes=(3, 3), strides=(1, 1), act=None, conv_type='default', name='conv'):
    with tf.variable_scope(name):
        if conv_type == 'default':
            x = Conv2d(x, out_feats, kernel_sizes, strides, act=act, W_init=init(in_feats), b_init=init(in_feats))

        elif conv_type == 'depth_wise':
            x = DepthwiseConv2d(x, kernel_sizes, strides, act=tf.nn.relu, W_init=init(in_feats), 
                                b_init=init(in_feats), name='depthwise')
            x = Conv2d(x, out_feats, (1, 1), (1, 1), act=act, 
                       W_init=init(in_feats, kernel_sizes[0]), 
                       b_init=init(in_feats, kernel_sizes[0]), name='conv')

        else:
            raise Exception('Unknown conv type', conv_type)
    return x

def downsample(x, n_feats, scale=4, conv_type='default', sample_type='subpixel', name='downsample'):
    with tf.variable_scope(name):
        if sample_type == 'subpixel':
            assert scale == 2 or scale == 4

            # pretrain on scale 2 then finetune of scale 4
            x = conv(x, 3, n_feats//4, act=None, conv_type=conv_type, name='conv1')
            x = DeSubpixelConv2d(x, 2, name='pixel_deshuffle1')
            if scale == 4:
                x = conv(x, n_feats, n_feats//4, act=None, conv_type=conv_type, name='conv2')
                x = DeSubpixelConv2d(x, 2, name='pixel_deshuffle2')

        elif sample_type == 'deconv':
            x = conv(x, 3, n_feats, strides=(2, 2), act=tf.nn.relu, name='conv1_stride2')
            x = conv(x, n_feats, n_feats, strides=(2, 2), act=tf.nn.relu, name='conv2_stride2')

        elif sample_type == 'none':
            x = conv(x, 3, n_feats, act=tf.nn.relu, name='conv')

        else: 
            raise Exception('Unknown sample_type', sample_type)
    return x

def upsample(x, n_feats, scale=4, conv_type='default', sample_type='subpixel', name='upsample'):
    with tf.variable_scope(name):
        if sample_type == 'subpixel':
            assert scale == 2 or scale == 4

            x = conv(x, n_feats, n_feats*4, act=None, conv_type=conv_type, name='conv1')
            x = SubpixelConv2d(x, scale=2, n_out_channel=None, name='pixelshuffle1')# /1
            if scale == 4:
                x = conv(x, n_feats, n_feats*4, act=None, conv_type=conv_type, name='conv2')
                x = SubpixelConv2d(x, scale=2, n_out_channel=None, name='pixelshuffle2')

        elif sample_type == 'deconv':
            x = DeConv2d(x, n_feats, n_feats, strides=(2, 2), act=tf.nn.relu, W_init=init(), b_init=init())
            x = DeConv2d(x, n_feats, n_feats, strides=(2, 2), act=tf.nn.relu, W_init=init(), b_init=init())

        elif sample_type == 'none':
            x = conv(x, n_feats, n_feats, act=tf.nn.relu, name='conv')

        else:
            raise Exception('Unknown sample_type', sample_type)
    return x

def res_block(x, n_feats, conv_type='default', name='res_block'):
    with tf.variable_scope(name):
        res = conv(x, n_feats, n_feats, act=tf.nn.relu, conv_type=conv_type, name='conv1')
        res = conv(res, n_feats, n_feats, act=None, conv_type=conv_type, name='conv2')
        x = ElementwiseLayer([x, res], tf.add, name='res_add')
    return x

def res_group(x, n_feats, n_blocks, conv_type='default', name='res_group'):
    with tf.variable_scope(name):
        res = x
        for i in range(n_blocks):
            res = res_block(res, n_feats, conv_type=conv_type, name='res_block%d' %i)
        x = ElementwiseLayer([x, res], tf.add, name='add')
    return x

def body(res, n_feats, n_groups, n_blocks, n_convs, body_type='resnet', conv_type='defaults', name='body'):
    with tf.variable_scope(name):
        if body_type == 'resnet':
            for i in range(n_blocks):
                res = res_block(res, n_feats, conv_type=conv_type, name='res_block%d' %i)
        elif body_type == 'res_in_res':
            for i in range(n_groups):
                res = res_group(res, n_feats, n_blocks, conv_type=conv_type, name='res_group%d' %i)
        elif body_type == 'conv':
            for i in range(n_convs):
                res = conv(res, n_feats, n_feats, conv_type=conv_type, name='conv%d' %i)
        else:
            raise Exception('Unknown body type', body_type)
        
        res = conv(res, n_feats, n_feats, act=None, conv_type=conv_type, name='res_lastconv')
    return res

def SRGAN_g(t_bicubic, opt):

    sample_type = opt['sample_type'] 
    conv_type   = opt['conv_type']
    body_type   = opt['body_type']

    n_feats     = opt['n_feats']
    n_blocks    = opt['n_blocks']
    n_groups    = opt['n_groups']
    n_convs     = opt['n_convs']

    scale       = opt['scale']

    with tf.variable_scope('Generator') as vs:
        # normalize input (0, 1) -> (-127.5, 127.5)
        #t_image = (t_image - 0.5)*255
        x = InputLayer(t_bicubic, name='in')
        x = NormalizeLayer(x, 0.5, 255)
        g_skip = x

        #===========Downsample==============
        x = downsample(x, n_feats, scale, conv_type, sample_type)
        res = x

        #============Residual=================
        res = body(res, n_feats, n_groups, n_blocks, n_convs, body_type, conv_type)
        x = ElementwiseLayer([x, res], tf.add, name='add_res')
        
        #=============Upsample==================
        x = upsample(x, n_feats, scale, conv_type, sample_type)

        x = conv(x, n_feats, 3, act=None, conv_type=conv_type, name='global_res')
        x = ElementwiseLayer([x, g_skip], tf.add, name='add_global_res')

        #outputs = x.outputs/255 + 0.5
        x = RestoreLayer(x, 0.5, 255)
        outputs = tf.clip_by_value(x.outputs, 0, 1)
        return outputs


