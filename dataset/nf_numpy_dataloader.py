import matplotlib.pyplot as plt  # plt 用于显示图片
import nibabel as nib
import numpy as np
from pathlib import Path

from dataset.nf_tf_dataloader import get_nii2_list


def _get_data_patch(x, y, depth, height, width, channels, is_random):
    shape = y.shape
    k = (shape[0] - depth) // 2
    tumor_indexes = np.where(y > 0)
    if len(tumor_indexes) > 0 and len(tumor_indexes[1]) > 0 and len(tumor_indexes[2]) > 0:
        height_min = np.min(tumor_indexes[1])
        width_min = np.min(tumor_indexes[2])
        height_max = np.max(tumor_indexes[1])
        width_max = np.max(tumor_indexes[2])
    else:
        height_min = 0
        width_min = 0
        height_max = shape[1] - height
        width_max = shape[2] - width

    height_range_min = np.max((0, np.min((height_min, shape[1] - height))))
    width_range_min = np.max((0, np.min((width_min, shape[2] - width))))

    height_range_max = np.min((shape[1] - height, np.max((height_max - height, 0))))
    width_range_max = np.min((shape[2] - width, np.max((width_max - height, 0))))

    if height_range_min == height_range_max:
        height_range_min = np.max((0, height_range_min - 1))
        height_range_max = np.min((height_range_max + 1, shape[1] - height))
    if width_range_min == width_range_max:
        width_range_min = np.max((0, width_range_min - 1))
        width_range_max = np.min((width_range_max + 1, shape[2] - width))

    if height_range_max < height_range_min:
        temp = height_range_max
        height_range_max = height_range_min
        height_range_min = temp
    if width_range_max < width_range_min:
        temp = width_range_min
        width_range_min = width_range_max
        width_range_max = temp
    if is_random:
        j = np.random.randint(height_range_min, height_range_max)
        i = np.random.randint(width_range_min, width_range_max)
    else:
        j = height_range_min
        i = width_range_min
    x = x[k:k + depth, j:j + height, i:i + width, np.newaxis]
    y = y[k:k + depth, j:j + height, i:i + width, np.newaxis]
    return x, y


def generator(_dataset, batch_size, depth, height, width, channels, is_random, transpose=(2, 1, 0)):
    dataset_size = len(_dataset)
    _i = 0
    batch_xs = np.empty(shape=(batch_size, depth, height, width, channels), dtype=np.float)
    batch_ys = np.empty(shape=(batch_size, depth, height, width, channels), dtype=np.float)
    while True:
        for _j in range(batch_size):
            if _i == 0 and is_random:
                np.random.shuffle(_dataset)
            x = nib.load(_dataset[_i]['volume']).get_data()
            x = np.transpose(x, transpose)
            y = nib.load(_dataset[_i]['segmentation']).get_data()
            y = np.transpose(y, transpose)
            x, y = _get_data_patch(x, y, depth, height, width, channels, is_random)
            batch_xs[_j] = x
            batch_ys[_j] = y
        batch_xs = np.clip(batch_xs, 0, 1000) / 1000
        batch_ys[batch_ys > 0] = 1
        # batch_ys[batch_ys <= 0] = 0
        # _indexs = [w for w, _slice in enumerate(y) if np.max(_slice) > 0]
        # if is_random:
        #     batch_indexs[_j] = _indexs[np.random.randint(0, len(_indexs))] if len(_indexs) > 0 else 0
        # else:
        #     batch_indexs[_j] = _indexs[0] if len(_indexs) > 0 else 0
        _i = (_i + 1) % dataset_size
        if np.max(batch_ys) > 0:
            yield np.transpose(batch_xs, (0, 2, 3, 1, 4)), np.transpose(batch_ys, (0, 2, 3, 1, 4))


def generator_volume(_dataset, batch_size, image_depth, image_height, image_width, image_channel, transpose=(2, 1, 0)):
    dataset_size = len(_dataset)
    for i in range(dataset_size):
        x = nib.load(_dataset[i]['volume']).get_data()
        x = np.transpose(x, transpose)
        _shape = x.shape
        pad_depth = image_depth - _shape[0] % image_depth
        pad_height = image_height - _shape[1] % image_height
        pad_width = image_width - _shape[2] % image_width
        x = np.pad(x, ((0, pad_depth), (0, pad_height), (0, pad_width)), 'constant')
        len_depth = x.shape[0] // image_depth
        len_height = x.shape[1] // image_height
        len_width = x.shape[2] // image_width
        y = nib.load(_dataset[i]['segmentation']).get_data()
        y = np.transpose(y, transpose)
        y = np.pad(y, ((0, pad_depth), (0, pad_height), (0, pad_width)), 'constant')
        yield x, y, len_depth, len_height, len_width, pad_depth, pad_height, pad_width, \
              Path(_dataset[i]['volume']).name.split('.')[0], nib.load(_dataset[i]['volume']), nib.load(
            _dataset[i]['segmentation'])


def get_patch_all(x, y, image_depth):
    patch_xs = x[0:image_depth, :, :, np.newaxis]
    patch_ys = y[0:image_depth, :, :, np.newaxis]
    return patch_xs, patch_ys


def get_patch_list(x, y, len_depth, len_height, len_width, image_depth, image_height, image_width):
    patch_xs = np.zeros((len_depth, len_height, len_width, image_depth, image_height, image_width, 1))
    patch_ys = np.zeros((len_depth, len_height, len_width, image_depth, image_height, image_width, 1))
    for _d in range(len_depth):
        for _h in range(len_height):
            for _w in range(len_width):
                patch_xs[_d, _h, _w] = (
                    x[_d * image_depth:(_d + 1) * image_depth, _h * image_height:(_h + 1) * image_height,
                    _w * image_width:(_w + 1) * image_width, np.newaxis])
                patch_ys[_d, _h, _w] = (
                    y[_d * image_depth:(_d + 1) * image_depth, _h * image_height:(_h + 1) * image_height,
                    _w * image_width:(_w + 1) * image_width, np.newaxis])
    patch_xs = np.clip(np.array(patch_xs), 0, 1000) / 1000
    patch_ys = np.array(patch_ys)
    patch_ys[patch_ys > 0] = 1.0
    patch_ys[patch_ys <= 0] = 0.0
    return patch_xs, patch_ys


def show_test_generator(depth):
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
    dataset = get_nii2_list('E:/Dataset/Neurofibromatosis/source/test')
    dataset = list(dataset.values())
    train_dataset_size = len(dataset) * 9 // 10
    train_dataset = dataset  # dataset[:train_dataset_size]
    eval_dataset = dataset[train_dataset_size:]
    train_generator = generator(train_dataset, batch_size=1, depth=16, height=256, width=256,
                                channels=1, is_random=True)  # , transpose=(1, 2, 0)
    eval_generator = generator(eval_dataset, batch_size=1, depth=16, height=256, width=256,
                               channels=1, is_random=False)  # , transpose=(1, 2, 0)
