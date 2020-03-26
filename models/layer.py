import numpy as np
import tensorflow as tf


def grayscale_to_rainbow(image):
    with tf.variable_scope('garyscale_to_rainbow'):
        H = tf.squeeze((1. - image) * 2. / 3., axis=-1)
        SV = tf.ones(H.get_shape())
        HSV = tf.stack([H, SV, SV], axis=3)
        RGB = tf.image.hsv_to_rgb(HSV)
    return RGB


def get_num_channels(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the number of channels of x
    """
    return int(x.get_shape()[-1])


def get_spatial_rank(x):
    """
    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the spatial rank of the tensor i.e. the number of spatial dimensions between batch_size and num_channels
    """
    return len(x.get_shape()) - 2


# Weight initialization (Xavier's init)
def xavier_initializer_convolution(shape, dist='uniform', lambda_initializer=True):
    s = len(shape) - 2
    num_activations = np.prod(shape[:s]) * np.sum(shape[s:])  # input_activations + output_activations
    if dist == 'uniform':
        lim = np.sqrt(6. / num_activations)
        if lambda_initializer:
            return np.random.uniform(-lim, lim, shape).astype(np.float32)
        else:
            return tf.random_uniform(shape, minval=-lim, maxval=lim)
    if dist == 'normal':
        stddev = np.sqrt(3. / num_activations)
        if lambda_initializer:
            return np.random.normal(0, stddev, shape).astype(np.float32)
        else:
            tf.truncated_normal(shape, mean=0, stddev=stddev)
    raise ValueError('Distribution must be either "uniform" or "normal".')


def constant_initializer(value, shape, lambda_initializer=True):
    if lambda_initializer:
        return np.full(shape, value).astype(np.float32)
    else:
        return tf.constant(value, tf.float32, shape)


def convolution(x, filter_, padding='SAME', strides=None, dilation_rate=None, initializer='XAVIER'):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter_))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter_[-1]))
    return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b


def deconvolution(x, filter_, output_shape, strides, padding='SAME'):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter_))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter_[-2]))
    spatial_rank = get_spatial_rank(x)
    if spatial_rank == 2:
        return tf.nn.conv2d_transpose(x, w, output_shape, strides, padding) + b
    if spatial_rank == 3:
        return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b
    raise ValueError('Only 2D and 3D images supported.')


def conv_bn_relu_drop(input_, filter_, strides=None, relu=None, norm=None, is_train=True, drop=0, replace=1,
                      resnet=True):
    with tf.variable_scope('conv_bn_relu_drop'):
        x = input_
        for i in range(replace):
            with tf.variable_scope('conv_' + str(i + 1)):
                x = convolution(x, filter_, strides=strides)
                if resnet and i == replace - 1:
                    x = x + input_
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                  training=is_train)
                if relu is not None:
                    x = relu(x)
                x = tf.nn.dropout(x, keep_prob=1 - drop)
        return x


def down_conv_bn_relu(x, filter_, strides, relu=None, norm=None, is_train=True):
    with tf.variable_scope('down_conv_bn_relu'):
        x = convolution(x, filter_, strides=strides)
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_train)
        if relu is not None:
            x = relu(x)
        return x


def up_conv_bn_relu(x, filter_, strides, output_shape, relu=None, norm=None, is_train=True):
    with tf.variable_scope('up_conv_bn_relu'):
        x = deconvolution(x, filter_, output_shape, strides=strides)
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_train)
        if relu is not None:
            x = relu(x)
        return x


def concat_conv_bn_relu_drop(input_, feature, filter_, strides, relu=None, norm=None, is_train=True, drop=0, replace=1,
                             resnet=True):
    with tf.variable_scope('concat_conv_bn_relu_drop'):
        x = tf.concat((input_, feature), axis=-1)
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_train)
        for i in range(replace):
            with tf.variable_scope('conv_' + str(i + 1)):
                if i == 0:
                    _filter = [_f if _i != len(filter_) - 2 else _f * 2 for _i, _f in enumerate(filter_)]
                    x = convolution(x, _filter)
                else:
                    x = convolution(x, filter_, strides=strides)
                    input_ = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                           training=is_train)
                if resnet and i == replace - 1:
                    x = x + input_
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                  training=is_train)
                if relu is not None:
                    x = relu(x)
                x = tf.nn.dropout(x, keep_prob=1 - drop)
        return x


def logits2predict(logits):
    with tf.name_scope('predicted_label'):
        pred_op = tf.argmax(logits, axis=-1, name='prediction')
        return pred_op


def logits2softmax(logits):
    with tf.name_scope('softmax_label'):
        softmax_op = tf.nn.softmax(logits, name='softmax')
        return softmax_op
