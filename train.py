import os

import tensorflow as tf
from pathlib import Path

from dataset import NiftiDataset3D
from dataloaders.nf_tf_dataloader import get_nii2_list, nift_generator, get_nii_list
# from models.vnet3d import Vnet3dModule
from models.vnet3d2 import Vnet3dModule

tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 1  # Size of batch
depth = 16  # Number of layers in data patch
height = 256  # Size of a data patch
width = 256  # Size of a data patch
input_channels = ['volume']
label_channels = ['segmentation']
classes = [0, 1]

init_filter = 16
keep_prob = 0.7
num_epoch = 100000
checkpoint_dir = Path('temp/checkpoints') / 'vnet_intnet'  # Directory where to write checkpoint
log_dir = Path('temp/tensorboard_logs') / 'vnet_intnet'  # Directory where to write training and testing event logs
is_restore = False

learning_rate = 0.001  # Initial learning rate
decay_factor = 0.99  # Exponential decay learning rate factor
decay_steps = 100  # Number of epoch before applying one learning rate decay

train_transforms = [
    # NiftiDataset.Normalization(),
    # NiftiDataset3D.ExtremumNormalization(0.1),
    NiftiDataset3D.ManualNormalization(min_window=0, max_window=900),
    # NiftiDataset3D.StatisticalNormalization(2.5),
    # NiftiDataset3D.Reorient((0, 1, 2)),
    # NiftiDataset3D.Resample([1.5, 1.5, 10]),
    NiftiDataset3D.Padding((height, width, depth)),
    NiftiDataset3D.RandomCrop((height, width, depth), is_train=True, drop_ratio=0.01, min_pixel=30),
    # NiftiDataset.BSplineDeformation(randomness=2),
    NiftiDataset3D.RandomFlip((True, True, True)),
    NiftiDataset3D.RandomNoise()
]

eval_transforms = [
    # NiftiDataset.Normalization(),
    # NiftiDataset3D.ExtremumNormalization(0.1),
    NiftiDataset3D.ManualNormalization(min_window=0, max_window=900),
    # NiftiDataset3D.StatisticalNormalization(2.5),
    # NiftiDataset3D.Reorient((0, 1, 2)),
    # NiftiDataset3D.Resample([1.5, 1.5, 10]),
    NiftiDataset3D.Padding((height, width, depth)),
    NiftiDataset3D.RandomCrop((height, width, depth), is_train=False, drop_ratio=0, min_pixel=30)
]


def train():
    dataset = get_nii_list('/home/yf/PythonProject/MIS/data/NF/nii_NF')
    # dataset = get_nii2_list('E:/Dataset/Neurofibromatosis/source/test')
    dataset = list(dataset.values())
    train_dataset_size = len(dataset) * 9 // 10
    train_dataset = dataset[:train_dataset_size]
    eval_dataset = dataset[train_dataset_size:]

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        train_generator = nift_generator(train_dataset, train_transforms, batch_size, input_channels, label_channels,
                                         classes, is_train=True)
        eval_generator = nift_generator(eval_dataset, eval_transforms, batch_size, input_channels, label_channels,
                                        classes, is_train=True)
        model = Vnet3dModule(batch_size, depth, height, width, len(input_channels), init_filter, len(classes),
                             keep_prob, activation_type='prelu', loss_name='sorensen')
        model.build_graph(is_train=True, init_learning_rate=learning_rate, decay_steps=decay_steps,
                          decay_factor=decay_factor)
        with tf.Session(config=config_proto) as sess:
            model.train(sess, train_generator, eval_generator, checkpoint_dir, log_dir, train_epochs=num_epoch,
                        # train_steps=1, eval_steps=1)
                        train_steps=len(train_dataset) // batch_size, eval_steps=len(eval_dataset) // batch_size)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not is_restore:
        if tf.gfile.Exists(str(checkpoint_dir)):  # checkpoint_dir.exists():
            tf.gfile.DeleteRecursively(str(checkpoint_dir))  # shutil.rmtree(str(checkpoint_dir))

        if tf.gfile.Exists(str(log_dir)):  # log_dir.exists():
            tf.gfile.DeleteRecursively(str(log_dir))  # shutil.rmtree(str(log_dir))
    tf.gfile.MakeDirs(str(checkpoint_dir))  # checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tf.gfile.MakeDirs(str(log_dir))  # log_dir.mkdir(parents=True, exist_ok=True)
    train()
