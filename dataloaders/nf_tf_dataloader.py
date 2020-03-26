import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  # plt 用于显示图片
from pathlib import Path

from dataset import NiftiDataset3D


def get_nii_list(_path):
    dataset = {}
    nii_path = Path(_path)
    for _iter in nii_path.iterdir():
        name = _iter.name
        _type, _key = name.split('-')
        if _key not in dataset.keys():
            dataset[_key] = {}
        dataset[_key][_type] = str(_iter)
    return dataset


def get_nii2_list(_path):
    dataset = {}
    nii_path = Path(_path)
    for _iter in nii_path.iterdir():
        name = _iter.name
        dataset[name] = {}
        dataset[name]['volume'] = str(_iter / 'stir.nii')
        dataset[name]['segmentation'] = str(_iter / 'label.nii')
    return dataset


def nift_generator(data_list, data_transforms, batch_size, input_channels, label_channels, classes, is_train=True):
    with tf.device('/cpu:0'):
        with tf.name_scope('{}_pipeline'.format('train' if is_train else 'test')):
            Dataset = NiftiDataset3D.NiftiDataset(data_list, input_channels, label_channels, classes, data_transforms,
                                                  is_train)
            dataset = Dataset.get_dataset()
            dataset = dataset.shuffle(buffer_size=3)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            iterator = dataset.make_initializable_iterator()
    return iterator


if __name__ == '__main__':
    batch_size = 1
    depth = 16
    height = 256
    width = 256
    input_channels = ['volume']
    label_channels = ['segmentation']
    spacing = [1.5, 1.5, 10]
    classes = [0, 1]
    train_transforms = [
        NiftiDataset3D.ManualNormalization(0, 1000),
        # NiftiDataset3D.Resample(spacing),
        # NiftiDataset3D.Reorient((0, 1, 2)),
        NiftiDataset3D.Padding((height, width, depth)),
        NiftiDataset3D.RandomCrop((height, width, depth), True, 0.01, 30),
        NiftiDataset3D.RandomFlip((True, True, True)),
        NiftiDataset3D.RandomNoise()
    ]
    dataset = get_nii2_list('E:/Dataset/Neurofibromatosis/source/test')
    dataset = list(dataset.values())
    train_dataset_size = len(dataset) * 9 // 10
    train_dataset = dataset
    eval_dataset = dataset[train_dataset_size:]

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Session(config=config_proto) as sess:
        train_generator = nift_generator(train_dataset, train_transforms, batch_size, input_channels, label_channels,
                                         classes, is_train=True)
        sess.run(tf.initializers.global_variables())
        next_element_train = train_generator.get_next()
        sess.run(train_generator.initializer)
        image, label = sess.run(next_element_train)
        for _d in range(depth):
            plt.axis('off')  # 不显示坐标轴
            _image = np.transpose(image[0, :, :, _d, 0])
            plt.imshow(_image, cmap='gray')  # 显示图片
            plt.show()
            _label = np.transpose(label[0, :, :, _d, 0])
            plt.imshow(_label * 255, cmap='gray')  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.show()
