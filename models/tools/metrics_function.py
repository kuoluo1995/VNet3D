import tensorflow as tf


def get_metrics(pred_op, target, num_classes):
    with tf.name_scope('metrics'):
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.expand_dims(pred_op, -1), tf.cast(target, dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tp, tp_op = tf.metrics.true_positives(target, pred_op, name="true_positives")
        tn, tn_op = tf.metrics.true_negatives(target, pred_op, name="true_negatives")
        fp, fp_op = tf.metrics.false_positives(target, pred_op, name="false_positives")
        fn, fn_op = tf.metrics.false_negatives(target, pred_op, name="false_negatives")
        with tf.name_scope('sensitivity'):
            sensitivity_op = tf.divide(tf.cast(tp_op, tf.float32), tf.cast(tf.add(tp_op, fn_op), tf.float32))
        with tf.name_scope('specificity'):
            specificity_op = tf.divide(tf.cast(tn_op, tf.float32), tf.cast(tf.add(tn_op, fp_op), tf.float32))
        with tf.name_scope('dice'):
            dice_op = 2. * tp_op / (2. * tp_op + fp_op + fn_op)
        return accuracy, sensitivity_op, specificity_op, dice_op
