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
    return int(x.get_shape()[-1])


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


def conv3d(x, filter_, padding='SAME', strides=None, dilation_rate=None, initializer='XAVIER'):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter_))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter_[-1]))
    return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b


def deconv3d(x, filter_, output_shape, strides, padding='SAME'):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter_))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter_[-2]))
    return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b


def conv_bn_relu_drop(input_, filter_, strides=None, relu=None, norm=None, is_train=True, drop=0, replace=1,
                      resnet=True):
    with tf.variable_scope('conv_bn_relu_drop'):
        x = input_
        for i in range(replace):
            with tf.variable_scope('conv_' + str(i + 1)):
                x = conv3d(x, filter_, strides=strides)
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
        x = conv3d(x, filter_, strides=strides)
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_train)
        if relu is not None:
            x = relu(x)
        return x


def up_conv_bn_relu(x, filter_, strides, output_shape, relu=None, norm=None, is_train=True):
    with tf.variable_scope('up_conv_bn_relu'):
        x = deconv3d(x, filter_, output_shape, strides=strides)
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_train)
        if relu is not None:
            x = relu(x)
        return x


def concat_conv_bn_relu_drop(input_, feature, filter_, strides, relu=None, norm=None, is_train=True, drop=0, replace=1,
                             resnet=True):
    with tf.variable_scope('concat_conv_bn_relu_drop'):
        x = tf.concat((input_, feature), axis=-1)
        with tf.variable_scope('conv_' + str(1)):
            x = conv3d(x, [filter_[0], filter_[1], filter_[2], filter_[3] * 2, filter_[4]])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              training=is_train)
            if relu is not None:
                x = relu(x)
            x = tf.nn.dropout(x, keep_prob=1 - drop)
        for i in range(1, replace):
            with tf.variable_scope('conv_' + str(i + 1)):
                x = conv3d(x, filter_, strides=strides)
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


def vnet(x, init_channel, net_configs, relu=None, norm=None, is_train=True, drop=0, n_classes=2):
    # drop = drop if is_train else 0.0
    with tf.variable_scope('vnet/input_layer'):
        x = tf.tile(x, [1, 1, 1, 1, init_channel])
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_train)
    features = {}
    for level_name, items in net_configs.items():
        with tf.variable_scope('vnet/' + level_name):
            for sub_name, _configs in items.items():
                n_channels = get_num_channels(x)
                if 'conv_block' == sub_name:
                    filter_ = _configs['kernel'] + [n_channels, n_channels]
                    x = conv_bn_relu_drop(x, filter_, _configs['strides'], relu, norm, is_train, drop,
                                          _configs['replace'], _configs['resnet'])
                    features[level_name] = x
                if 'down_block' == sub_name:
                    filter_ = _configs['kernel'] + [n_channels, n_channels * 2]
                    x = down_conv_bn_relu(x, filter_, _configs['strides'], relu, norm, is_train)
                if 'up_block' == sub_name:
                    filter_ = _configs['kernel'] + [n_channels // 2, n_channels]
                    _shape = tf.shape(features[_configs['output_shape']])
                    x = up_conv_bn_relu(x, filter_, _configs['strides'], _shape, relu, norm, is_train)
                if 'concat_conv_block' == sub_name:
                    filter_ = _configs['kernel'] + [n_channels, n_channels]
                    feature = features[_configs['feature']]
                    x = concat_conv_bn_relu_drop(x, feature, filter_, _configs['strides'], relu, norm, is_train,
                                                 drop, _configs['replace'], _configs['resnet'])
    with tf.variable_scope('vnet/output_layer'):
        x = conv3d(x, [1, 1, 1, init_channel, n_classes])
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=is_train)
    return x


def logits2predict(logits):
    with tf.name_scope('predicted_label'):
        pred_op = tf.argmax(logits, axis=-1, name='prediction')
        return pred_op


def logits2softmax(logits):
    with tf.name_scope('softmax_label'):
        softmax_op = tf.nn.softmax(logits, name='softmax')
        return softmax_op


def get_loss(logits, target, num_classes, axis=(1, 2, 3), weighted=True, smooth=1e-5):
    with tf.name_scope('loss'):
        target = tf.cast(tf.one_hot(target[:, :, :, :, 0], depth=num_classes), dtype=tf.float32)
        inse = tf.reduce_sum(logits * target, axis=axis, name='intersection')
        l = tf.reduce_sum(logits, axis=axis, name='left')
        r = tf.reduce_sum(target, axis=axis, name='right')
        if weighted:
            w = 1 / (tf.reduce_sum(target * target, axis=axis) + smooth)
            dice = tf.reduce_sum(2. * w * inse + smooth, axis=-1) / tf.reduce_sum(w * (l + r + smooth), axis=-1)
            dice = tf.reduce_mean(dice, name='dice_coe')
        else:
            dice = (2 * inse + smooth) / (l + r + smooth)
            dice = tf.reduce_mean(dice, name='dice_coe')
        return dice


def get_metrics(pred_op, target, num_classes):
    with tf.name_scope('metrics'):
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.expand_dims(pred_op, -1), tf.cast(target, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        label_one_hot = tf.one_hot(target[:, :, :, :, 0], depth=num_classes, name='label_one_hot')
        pred_one_hot = tf.one_hot(pred_op[:, :, :, :], depth=num_classes, name='pred_one_hot')
        for i in range(1, num_classes):
            tp, tp_op = tf.metrics.true_positives(label_one_hot[:, :, :, :, i], pred_one_hot[:, :, :, :, i],
                                                  name="true_positives_" + str(i))
            tn, tn_op = tf.metrics.true_negatives(label_one_hot[:, :, :, :, i], pred_one_hot[:, :, :, :, i],
                                                  name="true_negatives_" + str(i))
            fp, fp_op = tf.metrics.false_positives(label_one_hot[:, :, :, :, i], pred_one_hot[:, :, :, :, i],
                                                   name="false_positives_" + str(i))
            fn, fn_op = tf.metrics.false_negatives(label_one_hot[:, :, :, :, i], pred_one_hot[:, :, :, :, i],
                                                   name="false_negatives_" + str(i))
            with tf.name_scope('sensitivity'):
                sensitivity_op = tf.divide(tf.cast(tp_op, tf.float32), tf.cast(tf.add(tp_op, fn_op), tf.float32))
            with tf.name_scope('specificity'):
                specificity_op = tf.divide(tf.cast(tn_op, tf.float32), tf.cast(tf.add(tn_op, fp_op), tf.float32))
            with tf.name_scope('dice'):
                dice_op = 2. * tp_op / (2. * tp_op + fp_op + fn_op)
            return accuracy, sensitivity_op, specificity_op, dice_op


# parametric leaky relu
def prelu(x):
    with tf.variable_scope('prelu'):
        alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype,
                                initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


# Batch Normalization
def normalizationlayer(x, is_train, depth=None, height=None, width=None, norm_type=None, G=16, esp=1e-5, scope=None):
    """
    :param x:input data with shap of[batch,height,width,channel]
    :param is_train:flag of normalizationlayer,True is training,False is Testing
    :param height:in some condition,the data height is in Runtime determined,such as through deconv layer and conv2d
    :param width:in some condition,the data width is in Runtime determined
    :param depth:
    :param norm_type:normalization type:support"batch","group","None"
    :param G:in group normalization,channel is seperated with group number(G)
    :param esp:Prevent divisor from being zero
    :param scope:normalizationlayer scope
    :return:
    """
    with tf.name_scope(scope + norm_type):
        if norm_type == None:
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_train=is_train)
        elif norm_type == "group":
            # tranpose:[bs,z,h,w,c]to[bs,c,z,h,w]following the paper
            x = tf.transpose(x, [0, 4, 1, 2, 3])
            N, C, Z, H, W = x.get_shape().as_list()
            G = min(G, C)
            if H == None and W == None and Z == None:
                Z, H, W = depth, height, width
            x = tf.reshape(x, [-1, G, C // G, Z, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4, 5], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            gama = tf.get_variable(scope + norm_type + 'group_gama', [C], initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable(scope + norm_type + 'group_beta', [C], initializer=tf.constant_initializer(0.0))
            gama = tf.reshape(gama, [1, C, 1, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1, 1])
            output = tf.reshape(x, [-1, C, Z, H, W]) * gama + beta
            # tranpose:[bs,c,z,h,w]to[bs,z,h,w,c]following the paper
            output = tf.transpose(output, [0, 2, 3, 4, 1])
        return output
