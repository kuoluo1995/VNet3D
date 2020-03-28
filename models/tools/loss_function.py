import numpy as np
import tensorflow as tf

from models.tools.layer import get_spatial_rank


def get_loss_op(loss_name, logits_, softmax_, labels, num_classes, **kwargs):
    with tf.name_scope('loss'):
        _axis = np.arange(1, get_spatial_rank(labels) + 1)
        # labels = tf.cast(tf.one_hot(labels[..., 0], depth=num_classes), dtype=tf.float32)
        if loss_name == "xent":
            loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(labels, squeeze_dims=[len(_axis) + 1]),
                                                        logits=logits_))
        elif loss_name == 'weighted_corss_entropy':
            class_weights = tf.constant([1.0, 1.0])
            # deduce weights for batch samples based on their true label
            one_hot_labels = tf.one_hot(tf.squeeze(labels, squeeze_dims=[len(_axis) + 1]), depth=num_classes)

            weights = tf.reduce_sum(class_weights * one_hot_labels, axis=-1)
            # compute your (unweighted) softmax cross entropy loss
            logits_masked = logits_
            unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_masked,
                                                                             labels=tf.squeeze(labels, squeeze_dims=[
                                                                                 len(_axis) + 1]))
            # apply the weights, relying on broadcasting of the multiplication
            weighted_loss = unweighted_loss * weights
            # reduce the result to get your final loss
            loss_op = tf.reduce_mean(weighted_loss)
        elif loss_name == "sorensen":
            sorensen = dice_coe(softmax_, tf.cast(labels, dtype=tf.float32), loss_type='sorensen', axis=_axis)
            loss_op = 1. - sorensen
        elif (loss_name == "weighted_sorensen"):
            sorensen = dice_coe(softmax_, tf.cast(labels, dtype=tf.float32), loss_type='sorensen', axis=_axis,
                                weighted=True)
            loss_op = 1. - sorensen
        elif (loss_name == "jaccard"):
            jaccard = dice_coe(softmax_, tf.cast(labels, dtype=tf.float32), loss_type='jaccard', axis=_axis)
            loss_op = 1. - jaccard
        elif (loss_name == "weighted_jaccard"):
            jaccard = dice_coe(softmax_, tf.cast(labels, dtype=tf.float32), loss_type='jaccard', axis=_axis,
                               weighted=True)
            loss_op = 1. - jaccard
        elif (loss_name == "mixed_sorensen"):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_))
            sorensen = dice_coe(softmax_, tf.cast(labels, dtype=tf.float32), loss_type='sorensen', axis=_axis)
            loss_op = (1. - sorensen) + xent
        elif (loss_name == "mixed_weighted_sorensen"):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_))
            sorensen = dice_coe(softmax_, tf.cast(labels, dtype=tf.float32), loss_type='sorensen', axis=_axis,
                                weighted=True)
            loss_op = (1. - sorensen) + xent
        elif (loss_name == "mixed_jaccard"):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_))
            jaccard = dice_coe(softmax_, tf.cast(labels, dtype=tf.float32), loss_type='jaccard', axis=_axis)
            loss_op = (1. - jaccard) + xent
        elif (loss_name == "mixed_weighted_jaccard"):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_))
            jaccard = dice_coe(softmax_, tf.cast(labels, dtype=tf.float32), loss_type='jaccard', axis=_axis,
                               weighted=True)
            loss_op = (1. - jaccard) + xent
        else:
            raise Exception("Invalid loss function")
        return loss_op


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), weighted=False, smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    weighted : bool
        Boolean option for generalized dice loss
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> import tensorlayer as tl
    >>> outputs = tl.act.pixel_wise_softmax(outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    inse = tf.reduce_sum(output * target, axis=axis, name='intersection')
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis, name='left')
        r = tf.reduce_sum(target * target, axis=axis, name='right')
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis, name='left')
        r = tf.reduce_sum(target, axis=axis, name='right')
    else:
        raise Exception("Unknown loss_type")

    if weighted:
        w = 1 / (tf.reduce_sum(target * target, axis=axis) + smooth)
        dice = tf.reduce_sum(2. * w * inse + smooth, axis=-1) / tf.reduce_sum(w * (l + r + smooth), axis=-1)
        dice = tf.reduce_mean(dice, name='dice_coe')
    else:
        # old axis=[0,1,2,3]
        # dice = 2 * (inse) / (l + r)
        # epsilon = 1e-5
        # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
        # new haodong
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice, name='dice_coe')

    return dice
