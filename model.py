#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import math
from pixel_deshuffle import DeSubpixelConv2d

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py


def SRGAN_g(t_image):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    #w_init = tf.random_normal_initializer(stddev=0.02)
    #b_init = None  # tf.constant_initializer(value=0.0)
    #g_init = tf.random_normal_initializer(1., 0.02)
    #stddev = 1./(64*3*3)
    #w_init = tf.random_uniform_initializer(-stddev, stddev)
    #b_init = tf.random_uniform_initializer(-stddev, stddev)
    def init(in_channels=64):
        std = 1./math.sqrt(in_channels*3*3) # kernel size 3x3
        #print(in_channels, std) 
        return tf.random_uniform_initializer(-std, std)
        #return tf.keras.initializers.he_uniform()

    with tf.variable_scope("SRGAN_g") as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        # normalize input (0, 1) -> (-127.5, 127.5)
        t_image = (t_image-0.5)*255
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(3), b_init=init(3), name='n64s1/c')
        n = DeSubpixelConv2d(n, 2, name='pixel_deshuffle/1')
        n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(16), b_init=init(16), name='n64s1/c')
        n = DeSubpixelConv2d(n, 2, name='pixel_deshuffle/2')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=init(), b_init=init(), name='n64s1/c1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='n64s1/c2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='n64s1/c/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='out')
        outputs = n.outputs/255 + 0.5
        return outputs

def SRGAN_g1(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    #w_init = tf.random_normal_initializer(stddev=0.02)
    #b_init = None  # tf.constant_initializer(value=0.0)
    #g_init = tf.random_normal_initializer(1., 0.02)
    def init(in_channels=64):
        std = 1./math.sqrt(in_channels*3*3) # kernel size 3x3
        #print(in_channels, std) 
        return tf.random_uniform_initializer(-std, std)
        #return tf.keras.initializers.he_uniform()
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=init(3), name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=init(), b_init=init(), name='n64s1/c1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='n64s1/c2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='n64s1/c/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=init(), b_init=init(), name='out')
        return n



