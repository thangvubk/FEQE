#! /usr/bin/python

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import math
from pixel_deshuffle import DeSubpixelConv2d
import numpy as np
import scipy.io

IMAGE_MEAN = np.array([123.68 ,  116.779,  103.939])

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

class Bicubic(Layer):
    def __init__(self,prev_layer, name='bicubic'):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        shape = tf.shape(self.inputs)
        h, w = shape[1], shape[2]
        self.outputs = tf.image.resize_images(self.inputs, [h//4, w//4], tf.image.ResizeMethod.BICUBIC)
        self.all_layers.append(self.outputs)

def conv(x, in_feats, out_feats, kernel_sizes=(3, 3), strides=(1, 1), act=None, conv_type='default', name='conv'):
    with tf.variable_scope(name):
        if conv_type == 'default':
            x = Conv2d(x, out_feats, kernel_sizes, strides, act=act, 
                       W_init=init(in_feats, kernel_sizes[0]), 
                       b_init=init(in_feats, kernel_sizes[0]))

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
        if sample_type == 'desubpixel':
            assert scale == 2 or scale == 4
            if scale == 2:
                x = conv(x, 3, n_feats//4, (1, 1), act=None, conv_type=conv_type, name='conv')
                x = DeSubpixelConv2d(x, 2, name='pixel_deshuffle')
            else:
                x = DeSubpixelConv2d(x, 2, name='pixel_deshuffle1')
                x = conv(x, 12, n_feats//4, (1, 1), act=None, conv_type=conv_type, name='conv2')
                x = DeSubpixelConv2d(x, 2, name='pixel_deshuffle2')

        elif sample_type == 'conv_s2':
            x = conv(x, 3, n_feats, (1, 1), strides=(2, 2), act=tf.nn.relu, name='conv1_stride2')
            x = conv(x, n_feats, n_feats, (1, 1), strides=(2, 2), act=tf.nn.relu, name='conv2_stride2')

        elif sample_type == 'bicubic':
            x = RestoreLayer(x, 0.5, 255)
            x = Bicubic(x)
            x = NormalizeLayer(x, 0.5, 255)
            x = conv(x, 3, n_feats, (1, 1), act=tf.nn.relu, name='conv1')

        elif sample_type == 'pooling':
            x = MaxPool2d(x, (2, 2))
            x = conv(x, 12, n_feats, (1, 1), act=None, conv_type=conv_type, name='conv')
            x = MaxPool2d(x, (2, 2))

        elif sample_type == 'none':
            x = conv(x, 3, n_feats, act=tf.nn.relu, name='conv')

        else: 
            raise Exception('Unknown sample_type', sample_type)
    return x

def upsample(x, n_feats, scale=4, conv_type='default', sample_type='subpixel', name='upsample'):
    with tf.variable_scope(name):
        if sample_type == 'subpixel':
            assert scale == 2 or scale == 4
            if scale == 2:
                x = conv(x, n_feats, 3*4, (1, 1), act=None, conv_type=conv_type, name='conv')
                x = SubpixelConv2d(x, scale=2, n_out_channel=None, name='pixelshuffle')
            else:
                x = conv(x, n_feats, n_feats*4, (1, 1), act=None, conv_type=conv_type, name='conv1')
                x = SubpixelConv2d(x, scale=2, n_out_channel=None, name='pixelshuffle1')# /1
                x = conv(x, n_feats, 3*4, (1, 1), act=None, conv_type=conv_type, name='conv2')
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

def fire(x, n_feats, conv_type, name):
    with tf.variable_scope(name):
        res = conv(x, n_feats, n_feats//4, (1, 1), act=tf.nn.relu, conv_type=conv_type, name='conv1')
        res_11 = conv(res, n_feats//4, n_feats//2, (1, 1), act=tf.nn.relu, conv_type=conv_type, name='conv2')
        res_33 = conv(res, n_feats//4, n_feats//2, act=tf.nn.relu, conv_type=conv_type, name='conv3')
        res = ConcatLayer([res_11, res_33], 3, name='concat1')

        res = conv(x, n_feats, n_feats//4, (1, 1), act=tf.nn.relu, conv_type=conv_type, name='conv4')
        res_11 = conv(res, n_feats//4, n_feats//2, (1, 1), act=tf.nn.relu, conv_type=conv_type, name='conv5')
        res_33 = conv(res, n_feats//4, n_feats//2, act=tf.nn.relu, conv_type=conv_type, name='conv6')
        res = ConcatLayer([res_11, res_33], 3, name='concat2')
        x = ElementwiseLayer([x, res], tf.add, name='add')

    return x

def body(res, n_feats, n_groups, n_blocks, n_convs, n_squeezes, body_type='resnet', conv_type='default', name='body'):
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
        elif body_type == 'squeeze':
            for i in range(n_squeezes):
                res = fire(res, n_feats, conv_type, name='fire%d' %i) 
        else:
            raise Exception('Unknown body type', body_type)
        
        res = conv(res, n_feats, n_feats, act=None, conv_type=conv_type, name='res_lastconv')
    return res

def FEQE(t_bicubic, opt):

    #############Option Mutual Exclusive###############
    # body_type=resnet:         n_blocks is required
    # body_type='res_in_res':   n _blocks and n_groups are required
    # body_type='conv':         n_convs is required
    # body_type='squeeze':      n_squeezes is required
    
    downsample_type = opt['downsample_type']
    upsample_type = opt['upsample_type'] 
    conv_type   = opt['conv_type']
    body_type   = opt['body_type']

    n_feats     = opt['n_feats']
    n_blocks    = opt['n_blocks']
    n_groups    = opt['n_groups']
    n_convs     = opt['n_convs']
    n_squeezes  = opt['n_squeezes']

    scale       = opt['scale']

    with tf.variable_scope('Generator') as vs:
        # normalize input (0, 1) -> (-127.5, 127.5)
        #t_image = (t_image - 0.5)*255
        x = InputLayer(t_bicubic, name='in')
        x = NormalizeLayer(x, 0.5, 255)
        g_skip = x

        #===========Downsample==============
        x = downsample(x, n_feats, scale, conv_type, downsample_type)

        #============Residual=================
        x = body(x, n_feats, n_groups, n_blocks, n_convs, n_squeezes, body_type, conv_type)
        
        #=============Upsample==================
        x = upsample(x, n_feats, scale, conv_type, upsample_type)

        #x = conv(x, n_feats, 3, act=None, conv_type=conv_type, name='global_res')
        x = ElementwiseLayer([x, g_skip], tf.add, name='add_global_res')

        #outputs = x.outputs/255 + 0.5
        x = RestoreLayer(x, 0.5, 255)
        outputs = tf.clip_by_value(x.outputs, 0, 1)

        return outputs

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def vgg19(path_to_vgg_net, input_image):

    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(path_to_vgg_net)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        layer_type = name[:4]
        if layer_type == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
        elif layer_type == 'pool':
            current = _pool_layer(current)
        net[name] = current

    return net

def preprocess(image):
    return image - IMAGE_MEAN

