import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt  # plt 用于显示图片

from dataset.nf_dataloader import get_nii_list, generator, get_nii2_list, generator_volume, nift_generator
from models.vnet3d import Vnet3dModule

tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 1
depth = 16
height = 256
width = 256
input_channels = ['volume']
label_channels = ['segmentation']

spacing = [1.5, 1.5, 9]

init_filter = 16
classes = [0, 1]
model_dir = Path('_checkpoints')
logs_path = Path('tensorboard_logs') / 'vnet'

learning_rate = 1e-2
decay_steps = 100
decay_factor = 0.99

drop = 0.01
num_epoch = 99999


def train():
    dataset = get_nii_list('/home/yf/PythonProject/MIS/data/NF/nii_NF')
    # dataset = get_nii2_list('E:/Dataset/Neurofibromatosis/source/test')
    dataset = list(dataset.values())
    train_dataset_size = len(dataset) * 9 // 10
    train_dataset = dataset[:train_dataset_size]
    eval_dataset = dataset[train_dataset_size:]
    # train_generator = generator(train_dataset, batch_size=batch_size, depth=depth, height=height, width=width,
    #                             channels=len(channels), is_random=True)  # , transpose=(1, 2, 0)
    # eval_generator = generator(eval_dataset, batch_size=batch_size, depth=depth, height=height, width=width,
    #                            channels=len(channels), is_random=False)  # , transpose=(1, 2, 0)
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Session(config=config_proto) as sess:
        train_generator = nift_generator(train_dataset, spacing, batch_size, height, width, depth, input_channels,
                                         label_channels, classes, is_train=True)
        # sess.run(tf.initializers.global_variables())
        # next_element_train = train_generator.get_next()
        # sess.run(train_generator.initializer)
        # image, label = sess.run(next_element_train)
        # for _d in range(depth):
        #     plt.axis('off')  # 不显示坐标轴
        #     _image = np.transpose(image[0, :, :, _d, 0])
        #     plt.imshow(_image, cmap='gray')  # 显示图片
        #     plt.show()
        #     _label = np.transpose(label[0, :, :, _d, 0])
        #     plt.imshow(_label, cmap='gray')  # 显示图片
        #     plt.axis('off')  # 不显示坐标轴
        #     plt.show()
        eval_generator = nift_generator(eval_dataset, spacing, batch_size, height, width, depth, input_channels,
                                        label_channels, classes, is_train=False)
        model = Vnet3dModule(batch_size, depth, height, width, len(input_channels), init_filter, len(classes), drop)
        model.train(sess, train_generator, eval_generator, model_dir, logs_path, learning_rate, decay_steps,
                    decay_factor, train_epochs=num_epoch,
                    # train_steps=1, eval_steps=1)
                    train_steps=len(train_dataset) // batch_size, eval_steps=len(eval_dataset) // batch_size)


def predict():
    dataset = get_nii_list('/home/mgh3dqi/zjw/MIS/data/NF/nii_NF')
    # dataset = get_nii2_list('E:/Dataset/Neurofibromatosis/source/label')
    dataset = list(dataset.values())
    train_dataset_size = len(dataset) * 9 // 10
    eval_dataset = dataset[train_dataset_size:]
    eval_generator = generator_volume(eval_dataset, batch_size=batch_size, image_depth=depth, image_height=height,
                                      image_width=width, image_channel=1)
    Vnet3d = Vnet3dModule(batch_size, depth, height, width, len(input_channels), len(classes), drop)
    image_path = Path('./nf_results/')
    model_path = Path('./checkpoints/nf/vnet/best')
    Vnet3d.predict_patch(eval_generator, str(model_path), image_path, len(eval_dataset))


def test_generator():
    dataset = get_nii_list('/home/yf/PythonProject/MIS/data/NF/nii_NF')
    # dataset = get_nii2_list('E:/Dataset/Neurofibromatosis/source/label')
    dataset = list(dataset.values())
    train_dataset_size = len(dataset) * 9 // 10
    train_dataset = dataset[:train_dataset_size]
    train_generator = generator(train_dataset, batch_size=batch_size, depth=depth, height=height, width=width,
                                channels=len(input_channels), is_random=True)  # , transpose=(1, 2, 0)
    Path('test_generator').mkdir(parents=True, exist_ok=True)
    for i in range(1):
        batch_xs, batch_ys = next(train_generator)
        np.savez('test_generator/{}'.format(i), batch_xs=batch_xs, batch_ys=batch_ys)
        # for _d in range(depth):
        #     plt.axis('off')  # 不显示坐标轴
        #     plt.imshow(batch_xs[1, :, :, _d, 0], cmap='gray')  # 显示图片
        #     plt.show()
        #     plt.imshow(batch_ys[1, :, :, _d, 0], cmap='gray')  # 显示图片
        #     plt.axis('off')  # 不显示坐标轴
        #     plt.show()
        # print(11)


def show_test_generator():
    for i in range(1):
        value = np.load('./test_generator/{}.npz'.format(i))
        batch_xs, batch_ys = value['batch_xs'], value['batch_ys']
        for _d in range(depth):
            plt.axis('off')  # 不显示坐标轴
            plt.imshow(batch_xs[1, :, :, _d, 0], cmap='gray')  # 显示图片
            plt.show()
            plt.imshow(batch_ys[1, :, :, _d, 0], cmap='gray')  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.show()
        print(11)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # test_generator()
    # show_test_generator()
    train()
    # predict()
