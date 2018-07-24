import tensorflow as tf

def resnet_12_64(input_image):

    with tf.variable_scope("generator"):

        W1 = weight_variable([9, 9, 3, 64], name="W1"); b1 = bias_variable([64], name="b1");
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2"); b2 = bias_variable([64], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3"); b3 = bias_variable([64], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # residual 2

        W4 = weight_variable([3, 3, 64, 64], name="W4"); b4 = bias_variable([64], name="b4");
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 64, 64], name="W5"); b5 = bias_variable([64], name="b5");
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

        # residual 3

        W6 = weight_variable([3, 3, 64, 64], name="W6"); b6 = bias_variable([64], name="b6");
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7"); b7 = bias_variable([64], name="b7");
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5

        # residual 4

        W8 = weight_variable([3, 3, 64, 64], name="W8"); b8 = bias_variable([64], name="b8");
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9"); b9 = bias_variable([64], name="b9");
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9)) + c7

        # Convolutional

        W10 = weight_variable([3, 3, 64, 64], name="W10"); b10 = bias_variable([64], name="b10");
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        W11 = weight_variable([3, 3, 64, 64], name="W11"); b11 = bias_variable([64], name="b11");
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)

        # Final

        W12 = weight_variable([9, 9, 64, 3], name="W12"); b12 = bias_variable([3], name="b12");
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced

def resnet_8_32(input_image):

    with tf.variable_scope("generator"):

        W1 = weight_variable([9, 9, 3, 32], name="W1"); b1 = bias_variable([32], name="b1");
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 32, 32], name="W2"); b2 = bias_variable([32], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 32, 32], name="W3"); b3 = bias_variable([32], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # residual 2

        W4 = weight_variable([3, 3, 32, 32], name="W4"); b4 = bias_variable([32], name="b4");
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 32, 32], name="W5"); b5 = bias_variable([32], name="b5");
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

        # Convolutional

        W6 = weight_variable([3, 3, 32, 32], name="W6"); b6 = bias_variable([32], name="b6");
        c6 = tf.nn.relu(conv2d(c5, W6) + b6)

        W7 = weight_variable([3, 3, 32, 32], name="W7"); b7 = bias_variable([32], name="b7");
        c7 = tf.nn.relu(conv2d(c6, W7) + b7)

        # Final

        W8 = weight_variable([9, 9, 32, 3], name="W8"); b8 = bias_variable([3], name="b8");
        enhanced = tf.nn.tanh(conv2d(c7, W8) + b8) * 0.58 + 0.5

    return enhanced


def resnet_6_16(input_image):

    with tf.variable_scope("generator"):

        W1 = weight_variable([9, 9, 3, 16], name="W1"); b1 = bias_variable([16], name="b1");
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual

        W2 = weight_variable([3, 3, 16, 16], name="W2"); b2 = bias_variable([16], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 16, 16], name="W3"); b3 = bias_variable([16], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # Convolutional

        W4 = weight_variable([3, 3, 16, 16], name="W4"); b4 = bias_variable([16], name="b4");
        c4 = tf.nn.relu(conv2d(c3, W4) + b4)

        W5 = weight_variable([3, 3, 16, 16], name="W5"); b5 = bias_variable([16], name="b5");
        c5 = tf.nn.relu(conv2d(c4, W5) + b5)

        # Final

        W6 = weight_variable([9, 9, 16, 3], name="W6"); b6 = bias_variable([3], name="b6");
        enhanced = tf.nn.tanh(conv2d(c5, W6) + b6) * 0.58 + 0.5

    return enhanced


def srcnn(image_):

    with tf.variable_scope("generator"):

        weights = {
          'w1': tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=1e-3), name='w1'),
          'w2': tf.Variable(tf.random_normal([5, 5, 64, 32], stddev=1e-3), name='w2'),
          'w3': tf.Variable(tf.random_normal([5, 5, 32, 3], stddev=1e-3), name='w3')
        }

        biases = {
          'b1': tf.Variable(tf.zeros([64]), name='b1'),
          'b2': tf.Variable(tf.zeros([32]), name='b2'),
          'b3': tf.Variable(tf.zeros([1]), name='b3')
        }

        conv1 = tf.nn.relu(tf.nn.conv2d(image_, weights['w1'], strides=[1,1,1,1], padding='SAME') + biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='SAME') + biases['b2'])
        conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='SAME') + biases['b3']

    return tf.nn.tanh(conv3) * 0.58 + 0.5


def vgg_19(image_):

    with tf.variable_scope("shared_model"):

        gray_image = tf.image.rgb_to_grayscale(image_)

        conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64])
        conv_00_b = tf.get_variable("conv_00_b", [64])
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(gray_image, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

        for i in range(18):

            conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64])
            conv_b = tf.get_variable("conv_%02d_b" % (i+1), [64])

            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))

        conv_w = tf.get_variable("conv_20_w", [3,3,64,1])
        conv_b = tf.get_variable("conv_20_b", [1])

        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)
        tensor = tf.add(tensor, gray_image)

        tensor_colored = image_ - tf.image.grayscale_to_rgb(gray_image)
        tensor_colored += tf.image.grayscale_to_rgb(tensor)

    return tensor_colored


def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

