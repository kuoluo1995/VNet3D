import datetime
import shutil

import math
import nibabel as nib
import numpy as np
import tensorflow as tf
from dataset.nf_dataloader import get_patch_list, get_patch_all
from models.layer import vnet, prelu, grayscale_to_rainbow

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


def _logits2predict(logits):
    with tf.name_scope('predicted_label'):
        pred_op = tf.argmax(logits, axis=-1, name='prediction')
        return pred_op


def _logits2softmax(logits):
    with tf.name_scope('softmax_label'):
        softmax_op = tf.nn.softmax(logits, name='softmax')
        return softmax_op


def _get_loss(logits, target, num_classes, axis=(1, 2, 3), weighted=True, smooth=1e-5):
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


def _get_metrics(pred_op, target, num_classes):
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


class Vnet3dModule(object):
    def __init__(self, batch_size, depth, height, width, channels, init_filter, num_classes, drop=0, relu='prelu',
                 loss_name="dice coefficient"):
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
        logits = vnet(self.image_ph, init_filter, net_configs, relu=eval(relu), is_train=self.is_train, drop=self.drop,
                      n_classes=self.num_classes)
        self.pred_op = _logits2predict(logits)
        self.softmax_op = _logits2softmax(logits)
        self.loss_op = _get_loss(self.softmax_op, self.label_ph, num_classes)
        self.accuracy_op, self.sensitivity_op, self.specificity_op, self.dice_op = _get_metrics(self.pred_op,
                                                                                                self.label_ph,
                                                                                                self.num_classes)
        self.global_epoch = tf.Variable(0, name='global_epoch', trainable=False, dtype=tf.int32)
        with tf.name_scope('epoch_add'):
            self.global_epoch_inc = self.global_epoch.assign(self.global_epoch + 1)

    def build_summary(self, transpose=(3, 1, 2, 0)):
        # with tf.variable_scope('summary'):
        image_summary = list()
        for _b in range(self.batch_size):
            for _c in range(self.channels):
                with tf.name_scope('image_log'):
                    image_log = tf.transpose(tf.cast(self.image_ph[_b:_b + 1, :, :, :, _c] * 255, dtype=tf.uint8),
                                             transpose)
                image_summary.append(
                    tf.summary.image('image_b{}_c{}'.format(_b, _c), image_log, max_outputs=self.depth))
                with tf.name_scope('softmax_log'):
                    softmax_log = grayscale_to_rainbow(tf.transpose(self.softmax_op[_b:_b + 1, :, :, :, _c], transpose))
                    softmax_log = tf.cast(softmax_log * 255, dtype=tf.uint8)
                image_summary.append(
                    tf.summary.image("softmax_b{}_c{}".format(_b, _c), softmax_log, max_outputs=self.depth))
            with tf.name_scope('label_log'):
                label_log = tf.transpose(
                    tf.cast(self.label_ph[_b:_b + 1, :, :, :, 0] * math.floor(255 / (self.num_classes - 1)),
                            dtype=tf.uint8), transpose)
            image_summary.append(tf.summary.image('label_b{}'.format(_b), label_log, max_outputs=self.depth))
            with tf.name_scope('pred_log'):
                pred_log = tf.transpose(
                    tf.cast(self.pred_op[_b:_b + 1, :, :, :] * math.floor(255 / (self.num_classes - 1)),
                            dtype=tf.uint8), transpose)
            image_summary.append(tf.summary.image('pred_b{}'.format(_b), pred_log, max_outputs=self.depth))
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
        self.eval_loss = tf.placeholder(tf.float32, shape=None, name='eval_loss')
        eval_summary.append(tf.summary.scalar('eval/loss', self.eval_loss))
        self.eval_sensitivity = tf.placeholder(tf.float32, shape=None, name='eval_sensitivity')
        eval_summary.append(tf.summary.scalar('eval/sensitivity', self.eval_sensitivity))
        self.eval_specificity = tf.placeholder(tf.float32, shape=None, name='eval_specificity')
        eval_summary.append(tf.summary.scalar('eval/specificity', self.eval_specificity))
        self.eval_dice = tf.placeholder(tf.float32, shape=None, name='eval_dice')
        eval_summary.append(tf.summary.scalar('eval/dice', self.eval_dice))
        self.eval_accuracy = tf.placeholder(tf.float32, shape=None, name='eval_accuracy')
        eval_summary.append(tf.summary.scalar('eval/accuracy', self.eval_dice))
        self.eval_summary = tf.summary.merge(eval_summary)

    def train(self, sess, train_generator, eval_generator, model_dir, logs_path, init_learning_rate, decay_steps,
              decay_factor, train_epochs=10000, train_steps=31, eval_steps=30, is_restore=False):
        with tf.name_scope("learning_rate"):
            self.learning_rate = tf.train.exponential_decay(init_learning_rate, self.global_epoch, decay_steps,
                                                            decay_factor, staircase=False)
        with tf.name_scope('training'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss=self.loss_op)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op, update_ops])
        self.build_summary()
        sess.run(tf.initializers.global_variables())
        best_saver = tf.train.Saver(max_to_keep=3)
        if is_restore:
            if (model_dir / 'model').exists():
                latest_ckpt = tf.train.latest_checkpoint(str(model_dir), latest_filename='model-latest')
                best_saver.restore(sess, latest_ckpt)
        else:
            if model_dir.exists():
                shutil.rmtree(str(model_dir))
            model_dir.mkdir(parents=True, exist_ok=True)
            if logs_path.exists():
                shutil.rmtree(str(logs_path))
            logs_path.mkdir(parents=True, exist_ok=True)

        summary_writer = tf.summary.FileWriter(str(logs_path), graph=tf.get_default_graph())

        best_ac = 0
        train_iterator = train_generator.get_next()
        eval_iterator = eval_generator.get_next()
        for i in range(self.global_epoch.eval(session=sess), train_epochs):
            # get new batch
            best_accuracy = train_ac = train_sensitivity = train_specificity = train_dice = train_lost = 0
            best_xs, best_ys = None, None
            sess.run(train_generator.initializer)
            for j in range(train_steps):
                batch_xs, batch_ys = sess.run(train_iterator)
                sess.run(tf.local_variables_initializer())
                _, _dice, _lost, _sensitivity, _specificity, _ac = sess.run(
                    [train_op, self.dice_op, self.loss_op, self.sensitivity_op, self.specificity_op, self.accuracy_op],
                    feed_dict={self.image_ph: batch_xs, self.label_ph: batch_ys, self.is_train: True})
                train_dice += _dice
                train_lost += _lost
                train_sensitivity += _sensitivity
                train_specificity += _specificity
                train_ac += _ac
                if _dice > best_accuracy:
                    best_accuracy, best_xs, best_ys = _dice, batch_xs, batch_ys
            print("{}: Training of epoch {} complete, loss:{},ac:{},dice:{}".format(datetime.datetime.now(), i,
                                                                                    train_lost / train_steps,
                                                                                    train_ac / train_steps,
                                                                                    train_dice / train_steps))
            summary = sess.run(self.image_summary, feed_dict={self.image_ph: best_xs, self.label_ph: best_ys,
                                                              self.is_train: False})
            summary_writer.add_summary(summary, i)
            summary = sess.run(self.train_summary, feed_dict={self.loss_op: train_lost / train_steps,
                                                              self.accuracy_op: train_ac / train_steps,
                                                              self.dice_op: train_dice / train_steps,
                                                              self.sensitivity_op: train_sensitivity / train_steps,
                                                              self.specificity_op: train_specificity / train_steps})
            summary_writer.add_summary(summary, i)

            sess.run(eval_generator.initializer)
            eval_dice = eval_lost = eval_ac = eval_sensitivity = eval_specificity = 0
            for j in range(eval_steps):
                batch_xs, batch_ys = sess.run(eval_iterator)
                sess.run(tf.local_variables_initializer())
                _dice, _lost, _ac, _sensitivity, _specificity = sess.run(
                    [self.dice_op, self.loss_op, self.accuracy_op, self.sensitivity_op, self.specificity_op],
                    feed_dict={self.image_ph: batch_xs, self.label_ph: batch_ys, self.is_train: False})
                eval_ac += _ac
                eval_dice += _dice
                eval_lost += _lost
                eval_sensitivity += _sensitivity
                eval_specificity += _specificity
            print("{}: Evaling of epoch {} complete, loss:{},ac:{},dice:{}".format(datetime.datetime.now(), i,
                                                                                   eval_lost / eval_steps,
                                                                                   eval_ac / eval_steps,
                                                                                   eval_dice / eval_steps))
            summary = sess.run(self.eval_summary, feed_dict={self.eval_dice: eval_dice / eval_steps,
                                                             self.eval_accuracy: train_ac / eval_steps,
                                                             self.eval_specificity: eval_specificity / eval_steps,
                                                             self.eval_sensitivity: eval_sensitivity / eval_steps,
                                                             self.eval_loss: eval_lost / eval_steps})
            summary_writer.add_summary(summary, i)
            summary_writer.flush()
            if train_dice + eval_dice >= best_ac:
                best_ac = train_ac + eval_ac
                save_path = best_saver.save(sess, str(model_dir / 'model'), global_step=i)
                print("{}: Best model saved in file".format(datetime.datetime.now()))
            self.global_epoch_inc.op.run()
        summary_writer.close()

    def predict_patch(self, eval_generator, model_path, image_path, eval_steps=30):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=3)
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(init)
        self.restore_training(model_path, saver, sess)
        depth = self.depth
        height = self.height
        width = self.width
        for j in range(eval_steps):
            x, y, len_depth, len_height, len_width, pad_depth, pad_height, pad_width, image_fold, _x, _y = next(
                eval_generator)
            patch_xs, patch_ys = get_patch_list(x, y, len_depth, len_height, len_width, depth, height,
                                                width)
            patch_pres = np.zeros((len_depth * depth, len_height * height, len_width * width))
            for _d_i in range(len_depth):
                for _h_i in range(len_height):
                    for _w_i in range(len_width):
                        accuracy, cost, p = sess.run([self.accuracy, self.loss_op, self.pred_op],
                                                     feed_dict={self.image_ph: np.array([patch_xs[_d_i][_h_i][_w_i]]),
                                                                self.label_ph: np.array([patch_ys[_d_i][_h_i][_w_i]]),
                                                                self.is_train: 0, self.drop: 1})
                        patch_pres[_d_i * depth:(_d_i + 1) * depth,
                        _h_i * height:(_h_i + 1) * height,
                        _w_i * width:(_w_i + 1) * width] = p[0, :, :, :, 0]
            patch_pres = patch_pres[:-pad_depth, :-pad_height, :-pad_width]
            (image_path / image_fold).mkdir(parents=True, exist_ok=True)
            patch_pres = self._predict2nii(_y.affine, patch_pres, (2, 1, 0))
            nib.save(patch_pres, str(image_path / image_fold / 'prediction.nii'))
            nib.save(_x, str(image_path / image_fold / 'volume.nii'))
            nib.save(_y, str(image_path / image_fold / 'label.nii'))
            print('{}/{}'.format(j, eval_steps))

    def predict_all(self, eval_generator, model_path, image_path, eval_steps=30):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=3)
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(init)
        self.restore_training(model_path, saver, sess)
        for j in range(eval_steps):
            x, y, num_slice, len_height, len_width, pad_height, pad_width, image_fold, _x, _y = next(eval_generator)
            patch_xs, patch_ys = get_patch_all(x, y, self.depth)
            _accuracy, _cost, batch_preds = sess.run([self.accuracy, self.loss_op, self.pred_op],
                                                     feed_dict={self.image_ph: np.array([patch_xs]),
                                                                self.label_ph: np.array([patch_ys]), self.is_train: 0,
                                                                self.drop: 1})
            patch_pres = batch_preds[0, :, :, :, :]
            patch_pres = self._predict2nii(_y.affine, patch_pres, (2, 1, 0))
            nib.save(patch_pres, image_path + '/' + image_fold + '/prediction.nii')
            nib.save(_x, image_path + '/' + image_fold + '/volume.nii')
            nib.save(_y, image_path + '/' + image_fold + '/label.nii')
            print('{}/{}'.format(j, eval_steps))

    def _predict2nii(self, _affine, _nii, _transpose):
        patch_pres = np.transpose(_nii, _transpose)
        patch_pres = nib.Nifti1Image(patch_pres, _affine)
        return patch_pres

    def restore_training(self, model_path, saver, sess):
        print("\nReading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print('Checkpoint file: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            n_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print('Loading success, global training epoch is: {}\n'.format(n_epoch))
            return n_epoch
        else:
            print('No checkpoint file found.\n')
            return
