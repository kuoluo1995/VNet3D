import datetime
import shutil

import math
import tensorflow as tf

from models.tools.activation_function import get_activation_fn
from models.tools.layer import grayscale_to_rainbow, logits2predict, logits2softmax, get_num_channels, conv_bn_relu_drop, \
    down_conv_bn_relu, up_conv_bn_relu, concat_conv_bn_relu_drop, convolution
from models.tools.loss_function import get_loss_op
from models.tools.metrics_function import get_metrics

net_configs = {
    'encoder_1': {'conv_block': {'kernel': [5, 5, 5], 'strides': None, 'replace': 1, 'resnet': True},
                  'down_block': {'kernel': [2, 2, 2], 'strides': [2, 2, 2], 'replace': 1, 'resnet': False}},
    'encoder_2': {'conv_block': {'kernel': [5, 5, 5], 'strides': None, 'replace': 2, 'resnet': True},
                  'down_block': {'kernel': [2, 2, 2], 'strides': [2, 2, 2], 'replace': 1, 'resnet': False}},
    'encoder_3': {'conv_block': {'kernel': [5, 5, 5], 'strides': None, 'replace': 3, 'resnet': True},
                  'down_block': {'kernel': [2, 2, 2], 'strides': [2, 2, 2], 'replace': 1, 'resnet': False}},
    'encoder_4': {'conv_block': {'kernel': [5, 5, 5], 'strides': None, 'replace': 3, 'resnet': True},
                  'down_block': {'kernel': [2, 2, 2], 'strides': [2, 2, 2], 'replace': 1, 'resnet': False}},
    'bottom': {'conv_block': {'kernel': [5, 5, 5], 'strides': None, 'replace': 3, 'resnet': True}},
    'decoder_4': {'up_block': {'kernel': [2, 2, 2], 'strides': [1, 2, 2, 2, 1], 'output_shape': 'encoder_4'},
                  'concat_conv_block': {'kernel': [5, 5, 5], 'strides': None, 'feature': 'encoder_4', 'replace': 3,
                                        'resnet': True}},
    'decoder_3': {'up_block': {'kernel': [2, 2, 2], 'strides': [1, 2, 2, 2, 1], 'output_shape': 'encoder_3'},
                  'concat_conv_block': {'kernel': [5, 5, 5], 'strides': None, 'feature': 'encoder_3', 'replace': 3,
                                        'resnet': True}},
    'decoder_2': {'up_block': {'kernel': [2, 2, 2], 'strides': [1, 2, 2, 2, 1], 'output_shape': 'encoder_2'},
                  'concat_conv_block': {'kernel': [5, 5, 5], 'strides': None, 'feature': 'encoder_2', 'replace': 2,
                                        'resnet': True}},
    'decoder_1': {'up_block': {'kernel': [2, 2, 2], 'strides': [1, 2, 2, 2, 1], 'output_shape': 'encoder_1'},
                  'concat_conv_block': {'kernel': [5, 5, 5], 'strides': None, 'feature': 'encoder_1', 'replace': 1,
                                        'resnet': True}}
}


def get_logits(x, init_channel, net_configs, relu=None, norm=None, is_train=True, drop=0, n_classes=2):
    # drop = drop if is_train else 0.0
    with tf.variable_scope('vnet'):
        with tf.variable_scope('input_layer'):
            x = tf.tile(x, [1, 1, 1, 1, init_channel])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              training=is_train)
        features = {}
        for level_name, items in net_configs.items():
            with tf.variable_scope(level_name):
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
        with tf.variable_scope('output_layer'):
            x = convolution(x, [1, 1, 1, init_channel, n_classes])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              training=is_train)
        return x


class Vnet3dModule:
    def __init__(self, batch_size, depth, height, width, channels, init_filter, num_classes, drop=0, relu='prelu',
                 loss_name="weighted_sorensen"):
        self.batch_size = batch_size
        self.depth = depth
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.drop = drop

        self.image_ph = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.depth, self.channels],
                                       name='image')
        self.label_ph = tf.placeholder(tf.int32, shape=[None, self.height, self.width, self.depth, self.channels],
                                       name='label')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.drop_ph = tf.placeholder(tf.float32, name='drop')
        self.relu = get_activation_fn(relu)
        logits = get_logits(self.image_ph, init_filter, net_configs, relu=self.relu, is_train=self.is_train,
                            drop=self.drop_ph, n_classes=self.num_classes)
        self.pred_op = logits2predict(logits)
        self.softmax_op = logits2softmax(logits)
        self.loss_op = get_loss_op(loss_name, logits_=logits, softmax_=self.softmax_op, labels=self.label_ph,
                                   num_classes=num_classes)
        self.accuracy_op, self.sensitivity_op, self.specificity_op, self.dice_op = get_metrics(self.pred_op,
                                                                                               self.label_ph,
                                                                                               self.num_classes)
        self.global_epoch_op = tf.Variable(0, name='global_epoch', trainable=False, dtype=tf.int32)
        with tf.name_scope('epoch_add'):
            self.global_epoch_inc = self.global_epoch_op.assign(self.global_epoch_op + 1)

    def build_summary(self, transpose=(3, 1, 2, 0)):
        # with tf.variable_scope('summary'):
        image_summary = list()
        depth = self.depth
        for _b in range(self.batch_size):
            for _c in range(self.channels):
                with tf.name_scope('image_log'):
                    image_log = tf.transpose(tf.cast(self.image_ph[_b:_b + 1, ..., _c] * 255, dtype=tf.uint8),
                                             transpose)
                image_summary.append(tf.summary.image('image_b{}_c{}'.format(_b, _c), image_log, max_outputs=depth))
                with tf.name_scope('softmax_log'):
                    softmax_log = grayscale_to_rainbow(tf.transpose(self.softmax_op[_b:_b + 1, ..., _c], transpose))
                    softmax_log = tf.cast(softmax_log * 255, dtype=tf.uint8)
                image_summary.append(tf.summary.image("softmax_b{}_c{}".format(_b, _c), softmax_log, max_outputs=depth))
            with tf.name_scope('label_log'):
                label_log = tf.cast(self.label_ph[_b:_b + 1, ..., 0] * math.floor(255 / (self.num_classes - 1)),
                                    dtype=tf.uint8)
                label_log = tf.transpose(label_log, transpose)
            image_summary.append(tf.summary.image('label_b{}'.format(_b), label_log, max_outputs=depth))
            with tf.name_scope('pred_log'):
                pred_log = tf.cast(self.pred_op[_b:_b + 1, ...] * math.floor(255 / (self.num_classes - 1)),
                                   dtype=tf.uint8)
                pred_log = tf.transpose(pred_log, transpose)
            image_summary.append(tf.summary.image('pred_b{}'.format(_b), pred_log, max_outputs=depth))
        self.image_summary = tf.summary.merge(image_summary)

        train_summary = list()
        train_summary.append(tf.summary.scalar('train/loss', self.loss_op))
        train_summary.append(tf.summary.scalar('train/accuracy', self.accuracy_op))
        train_summary.append(tf.summary.scalar('train/sensitivity', self.sensitivity_op))
        train_summary.append(tf.summary.scalar('train/specificity', self.specificity_op))
        train_summary.append(tf.summary.scalar('train/dice', self.dice_op))
        train_summary.append(tf.summary.scalar('train/learning_rate', self.learning_rate))
        self.train_summary = tf.summary.merge(train_summary)

        eval_summary = list()
        self.eval_loss_ph = tf.placeholder(tf.float32, shape=None, name='eval_loss')
        eval_summary.append(tf.summary.scalar('eval/loss', self.eval_loss_ph))
        self.eval_sensitivity_ph = tf.placeholder(tf.float32, shape=None, name='eval_sensitivity')
        eval_summary.append(tf.summary.scalar('eval/sensitivity', self.eval_sensitivity_ph))
        self.eval_specificity_ph = tf.placeholder(tf.float32, shape=None, name='eval_specificity')
        eval_summary.append(tf.summary.scalar('eval/specificity', self.eval_specificity_ph))
        self.eval_dice_ph = tf.placeholder(tf.float32, shape=None, name='eval_dice')
        eval_summary.append(tf.summary.scalar('eval/dice', self.eval_dice_ph))
        self.eval_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='eval_accuracy')
        eval_summary.append(tf.summary.scalar('eval/accuracy', self.eval_accuracy_ph))
        self.eval_summary = tf.summary.merge(eval_summary)

    def train(self, sess, train_generator, eval_generator, model_dir, logs_path, init_learning_rate, decay_steps,
              decay_factor, train_epochs=10000, train_steps=31, eval_steps=30, is_restore=False):
        with tf.name_scope("learning_rate"):
            self.learning_rate = tf.train.exponential_decay(init_learning_rate, self.global_epoch_op, decay_steps,
                                                            decay_factor, staircase=False)
        with tf.name_scope('training'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss=self.loss_op, global_step=self.global_epoch_op)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op, update_ops])
        self.build_summary()
        sess.run(tf.initializers.global_variables())
        best_saver = tf.train.Saver(max_to_keep=3)
        if is_restore:
            if (model_dir / 'model-latest').exists():
                latest_ckpt = tf.train.latest_checkpoint(str(model_dir), latest_filename='model-latest')
                best_saver.restore(sess, latest_ckpt)
        else:
            if model_dir.exists():
                shutil.rmtree(str(model_dir))
            if logs_path.exists():
                shutil.rmtree(str(logs_path))
        model_dir.mkdir(parents=True, exist_ok=True)
        logs_path.mkdir(parents=True, exist_ok=True)
        # summary writer for tensorboard
        summary_writer = tf.summary.FileWriter(str(logs_path), graph=tf.get_default_graph())

        best_ac = 0
        train_iterator = train_generator.get_next()
        eval_iterator = eval_generator.get_next()
        for epoch in range(self.global_epoch_op.eval(session=sess), train_epochs):
            print("{}: Epoch {} starts...".format(datetime.datetime.now(), epoch + 1))
            # get new batch
            best_accuracy = train_ac = train_dice = train_sensitivity = train_specificity = train_loss = 0
            best_image, best_label = None, None
            sess.run(train_generator.initializer)
            for step in range(train_steps):
                _image, _label = sess.run(train_iterator)
                sess.run(tf.local_variables_initializer())
                _, _dice, _loss, _sensitivity, _specificity, _ac = sess.run(
                    [train_op, self.dice_op, self.loss_op, self.sensitivity_op, self.specificity_op, self.accuracy_op],
                    feed_dict={self.image_ph: _image, self.label_ph: _label, self.is_train: True,
                               self.drop_ph: self.drop})
                train_ac += _ac
                train_dice += _dice
                train_sensitivity += _sensitivity
                train_specificity += _specificity
                train_loss += _loss
                if _dice >= best_accuracy:
                    best_accuracy, best_image, best_label = _dice, _image, _label
            print("{}: Training of epoch {} complete, loss:{},ac:{},dice:{}".format(datetime.datetime.now(), epoch + 1,
                                                                                    train_loss / train_steps,
                                                                                    train_ac / train_steps,
                                                                                    train_dice / train_steps))
            summary = sess.run(self.image_summary, feed_dict={self.image_ph: best_image, self.label_ph: best_label,
                                                              self.is_train: False, self.drop_ph: 0})
            summary_writer.add_summary(summary, epoch)
            summary = sess.run(self.train_summary, feed_dict={self.loss_op: train_loss / train_steps,
                                                              self.accuracy_op: train_ac / train_steps,
                                                              self.dice_op: train_dice / train_steps,
                                                              self.sensitivity_op: train_sensitivity / train_steps,
                                                              self.specificity_op: train_specificity / train_steps})
            summary_writer.add_summary(summary, epoch)
            summary_writer.flush()

            sess.run(eval_generator.initializer)
            eval_loss = eval_ac = eval_dice = eval_sensitivity = eval_specificity = 0
            for step in range(eval_steps):
                _image, _label = sess.run(eval_iterator)
                sess.run(tf.local_variables_initializer())
                _dice, _loss, _ac, _sensitivity, _specificity = sess.run(
                    [self.dice_op, self.loss_op, self.accuracy_op, self.sensitivity_op, self.specificity_op],
                    feed_dict={self.image_ph: _image, self.label_ph: _label, self.is_train: False, self.drop_ph: 0})
                eval_loss += _loss
                eval_ac += _ac
                eval_dice += _dice
                eval_sensitivity += _sensitivity
                eval_specificity += _specificity
            print("{}: Evaling of epoch {} complete, loss:{},ac:{},dice:{}".format(datetime.datetime.now(), epoch + 1,
                                                                                   eval_loss / eval_steps,
                                                                                   eval_ac / eval_steps,
                                                                                   eval_dice / eval_steps))
            summary = sess.run(self.eval_summary, feed_dict={self.eval_dice_ph: eval_dice / eval_steps,
                                                             self.eval_accuracy_ph: eval_ac / eval_steps,
                                                             self.eval_specificity_ph: eval_specificity / eval_steps,
                                                             self.eval_sensitivity_ph: eval_sensitivity / eval_steps,
                                                             self.eval_loss_ph: eval_loss / eval_steps})
            summary_writer.add_summary(summary, epoch)
            summary_writer.flush()
            if train_dice + eval_dice >= best_ac:
                best_ac = train_ac + eval_ac
                save_path = best_saver.save(sess, str(model_dir / 'model'), global_step=epoch)
                print("{}: Best model saved in {}".format(datetime.datetime.now(), save_path))
            self.global_epoch_inc.op.run()
        summary_writer.close()

    def predict(self, sess, test_generator, model_dir, eval_steps=30):
        pass
