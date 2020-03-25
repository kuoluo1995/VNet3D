from pathlib import Path
import numpy as np

import SimpleITK as sitk


def get_nii_list(_path):
    dataset = {}
    nii_path = Path(_path)
    for _iter in nii_path.iterdir():
        name = _iter.name
        _key = name.split('.')[0]
        _type = name.split('.')[1]
        if 'segmentation' in _key:
            _type = 'segmentation_' + _type
            _key = name.split('_')[0]
        if _key not in dataset.keys():
            dataset[_key] = {}
        dataset[_key][_type] = str(_iter)
    return dataset


def _get_data_patch(x, y, image_depth, image_height, image_width, image_channel):
    shape = y.shape
    k = (shape[0] - image_depth) // 2
    tumor_indexes = np.where(y >= 1.0)
    if len(tumor_indexes) > 0:
        height_min = np.min(tumor_indexes[1])
        width_min = np.min(tumor_indexes[2])
        height_max = np.max(tumor_indexes[1])
        width_max = np.max(tumor_indexes[2])
    else:
        height_min = 0
        width_min = 0
        height_max = shape[1] - image_height
        width_max = shape[2] - image_width

    height_range_min = np.max((0, np.min((height_min, shape[1] - image_height))))
    width_range_min = np.max((0, np.min((width_min, shape[2] - image_width))))

    height_range_max = np.min((shape[1] - image_height, np.max((height_max - image_height, 0))))
    width_range_max = np.min((shape[2] - image_width, np.max((width_max - image_height, 0))))

    if height_range_min == height_range_max:
        height_range_min = np.max((0, height_range_min - 1))
        height_range_max = np.min((height_range_max + 1, shape[1] - image_height))
    if width_range_min == width_range_max:
        width_range_min = np.max((0, width_range_min - 1))
        width_range_max = np.min((width_range_max + 1, shape[2] - image_width))

    if height_range_max < height_range_min:
        temp = height_range_max
        height_range_max = height_range_min
        height_range_min = temp
    if width_range_max < width_range_min:
        temp = width_range_min
        width_range_min = width_range_max
        width_range_max = temp

    j = np.random.randint(height_range_min, height_range_max)
    i = np.random.randint(width_range_min, width_range_max)
    x = x[k:k + 64, j:j + image_height, i:i + image_width, np.newaxis]
    y = y[k:k + 64, j:j + image_height, i:i + image_width, np.newaxis]
    return x, y


def generator(_dataset, batch_size, image_depth, image_width, image_height, image_channel):
    dataset_size = len(_dataset)
    i = 0
    batch_xs = np.empty(shape=(batch_size, image_depth, image_height, image_width, image_channel), dtype=np.float)
    batch_ys = np.empty(shape=(batch_size, image_depth, image_height, image_width, image_channel), dtype=np.float)
    batch_indexs = np.empty(shape=batch_size, dtype=np.int64)
    while True:
        for j in range(batch_size):
            if i == 0:
                np.random.shuffle(_dataset)
            x = sitk.ReadImage(_dataset[i]['mhd'])
            x = sitk.GetArrayFromImage(x)
            x = np.concatenate((x, x, x))
            y = sitk.ReadImage(_dataset[i]['segmentation_mhd'])
            y = sitk.GetArrayFromImage(y)
            y = np.concatenate((y, y, y))
            x, y = _get_data_patch(x, y, image_depth, image_height, image_width, image_channel)
            batch_xs[j] = x
            batch_ys[j] = y
            _indexs = [j for j, _slice in enumerate(y) if np.max(_slice) > 0]

            batch_indexs[j] = _indexs[np.random.randint(0, len(_indexs))] if len(_indexs) > 0 else 0
            i = (i + 1) % dataset_size
        batch_xs = np.clip(batch_xs, 0.0, 1000.0) / 1000
        batch_ys = np.clip(batch_ys, 0.0, 1.0)
        yield batch_xs, batch_ys, batch_indexs
