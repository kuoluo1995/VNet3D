import os

import tensorflow as tf
from pathlib import Path

from dataset import NiftiDataset3D
from dataset.nf_tf_dataloader import get_nii_list, get_nii2_list, nift_generator
from models.vnet3d import Vnet3dModule

tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 1
depth = 16
height = 256
width = 256
input_channels = ['volume']
label_channels = ['segmentation']

spacing = [1.5, 1.5, 10]

init_filter = 16
classes = [0, 1]
model_dir = Path('_checkpoints')
logs_path = Path('tensorboard_logs') / 'vnet'

learning_rate = 1e-2
decay_factor = 0.99
decay_steps = 100

drop = 0.01
num_epoch = 99999

train_transforms = [
    # NiftiDataset3D.Resample(spacing),
    # NiftiDataset3D.Reorient((0, 1, 2)),
    NiftiDataset3D.ManualNormalization(0, 1000),
    NiftiDataset3D.Padding((height, width, depth)),
    NiftiDataset3D.RandomCrop((height, width, depth), True, 0.01, 30),
    NiftiDataset3D.RandomFlip((True, True, True)),
    NiftiDataset3D.RandomNoise()
]

eval_transforms = [
    # NiftiDataset3D.Resample(spacing),
    # NiftiDataset3D.Reorient((0, 1, 2)),
    NiftiDataset3D.ManualNormalization(0, 1000),
    NiftiDataset3D.Padding((height, width, depth)),
    NiftiDataset3D.RandomCrop((height, width, depth), False, 0, 30)
]


def train():
    # dataset = get_nii_list('/home/yf/PythonProject/MIS/data/NF/nii_NF')
    dataset = get_nii2_list('E:/Dataset/Neurofibromatosis/source/test')
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
                                        classes, is_train=False)
        model = Vnet3dModule(batch_size, depth, height, width, len(input_channels), init_filter, len(classes), drop)
        model.train(sess, train_generator, eval_generator, model_dir, logs_path, learning_rate, decay_steps,
                    decay_factor, train_epochs=num_epoch,
                    # train_steps=1, eval_steps=1)
                    train_steps=len(train_dataset) // batch_size, eval_steps=len(eval_dataset) // batch_size)


# def predict():
#     dataset = get_nii_list('/home/mgh3dqi/zjw/MIS/data/NF/nii_NF')
#     # dataset = get_nii2_list('E:/Dataset/Neurofibromatosis/source/label')
#     dataset = list(dataset.values())
#     train_dataset_size = len(dataset) * 9 // 10
#     eval_dataset = dataset[train_dataset_size:]
#     eval_generator = generator_volume(eval_dataset, batch_size=batch_size, image_depth=depth, image_height=height,
#                                       image_width=width, image_channel=1)
#     Vnet3d = Vnet3dModule(batch_size, depth, height, width, len(input_channels), len(classes), drop)
#     image_path = Path('./nf_results/')
#     model_path = Path('./checkpoints/nf/vnet/best')
#     Vnet3d.predict_patch(eval_generator, str(model_path), image_path, len(eval_dataset))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()
    # predict()
