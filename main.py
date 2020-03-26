import os

import tensorflow as tf
from pathlib import Path

from dataset import NiftiDataset3D
from dataloaders.nf_tf_dataloader import get_nii2_list, nift_generator, get_nii_list
from models.vnet3d import Vnet3dModule

tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 2
depth = 16
height = 256
width = 256
input_channels = ['volume']
label_channels = ['segmentation']

init_filter = 16
classes = [0, 1]
model_dir = Path('temp/checkpoints')
logs_path = Path('temp/tensorboard_logs') / 'vnet'

learning_rate = 1e-2
decay_factor = 0.99
decay_steps = 100

drop = 0.3
num_epoch = 99999

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
    with tf.Session(config=config_proto) as sess:
        train_generator = nift_generator(train_dataset, train_transforms, batch_size, input_channels, label_channels,
                                         classes, is_train=True)
        eval_generator = nift_generator(eval_dataset, eval_transforms, batch_size, input_channels, label_channels,
                                        classes, is_train=True)
        model = Vnet3dModule(batch_size, depth, height, width, len(input_channels), init_filter, len(classes), drop)
        model.train(sess, train_generator, eval_generator, model_dir, logs_path, learning_rate, decay_steps,
                    decay_factor, train_epochs=num_epoch,
                    # train_steps=1, eval_steps=1)
                    train_steps=len(train_dataset) // batch_size, eval_steps=len(eval_dataset) // batch_size)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()
