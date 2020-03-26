import math
import multiprocessing
import random
import numpy as np
import SimpleITK as sitk
import tensorflow as tf


class NiftiDataset:
    def __init__(self, data_list, input_channels, label_channels, classes, transforms, is_train=False):
        # Init membership variables
        self.data_list = data_list
        self.input_channels = input_channels
        self.label_channels = label_channels[0]
        self.classes = classes
        self.transforms = transforms
        self.is_train = is_train

    def get_dataset(self):
        self.data_size = len(self.data_list)
        case_list = np.arange(0, self.data_size)
        dataset = tf.data.Dataset.from_tensor_slices(case_list)
        dataset = dataset.map(lambda case: tuple(tf.py_func(self.input_parser, [case], [tf.float32, tf.int32])),
                              num_parallel_calls=multiprocessing.cpu_count())
        self.dataset = dataset
        return self.dataset

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        return reader.Execute()

    def input_parser(self, case_index):
        case = self.data_list[case_index]
        image_paths = []
        for channel in self.input_channels:
            image_paths.append(case[channel])

        # read image and label
        images = []
        for channel in range(len(image_paths)):
            image = self.read_image(image_paths[channel])
            images.append(image)

        for channel in range(len(images)):
            # check header same
            same_size = images[channel].GetSize() == images[0].GetSize()
            same_spacing = images[channel].GetSpacing() == images[0].GetSpacing()
            same_direction = images[channel].GetDirection() == images[0].GetDirection()
            if same_size and same_spacing and same_direction:
                continue
            else:
                raise Exception('Header info inconsistent: {}'.format(image_paths[channel]))

        label = sitk.Image(images[0].GetSize(), sitk.sitkUInt8)
        label.SetOrigin(images[0].GetOrigin())
        label.SetSpacing(images[0].GetSpacing())
        label.SetDirection(images[0].GetDirection())

        if self.is_train:
            label_ = self.read_image(case[self.label_channels])
            # check header same
            same_size = label_.GetSize() == images[0].GetSize()
            same_spacing = label_.GetSpacing() == images[0].GetSpacing()
            same_direction = label_.GetDirection() == images[0].GetDirection()
            if not (same_size and same_spacing and same_direction):
                raise Exception('Header info inconsistent: {}'.format(case['volume']))

            thresholdFilter = sitk.BinaryThresholdImageFilter()
            thresholdFilter.SetOutsideValue(0)
            thresholdFilter.SetInsideValue(1)

            castImageFilter = sitk.CastImageFilter()
            castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
            for channel in range(len(self.classes)):
                thresholdFilter.SetLowerThreshold(self.classes[channel])
                thresholdFilter.SetUpperThreshold(self.classes[channel])
                one_hot_label_image = thresholdFilter.Execute(label_)
                multiFilter = sitk.MultiplyImageFilter()
                one_hot_label_image = multiFilter.Execute(one_hot_label_image, channel)
                # cast one_hot_label to sitkUInt8
                one_hot_label_image = castImageFilter.Execute(one_hot_label_image)
                one_hot_label_image.SetSpacing(images[0].GetSpacing())
                one_hot_label_image.SetDirection(images[0].GetDirection())
                one_hot_label_image.SetOrigin(images[0].GetOrigin())
                addFilter = sitk.AddImageFilter()
                label = addFilter.Execute(label, one_hot_label_image)

        sample = {'image': images, 'label': label}
        if self.transforms:
            for transform in self.transforms:
                try:
                    sample = transform(sample)
                except:
                    print("Dataset preprocessing error: {}".format(case))
                    exit()

        # convert sample to tf tensors
        for channel in range(len(sample['image'])):
            image_np_ = sitk.GetArrayFromImage(sample['image'][channel])
            image_np_ = np.asarray(image_np_, np.float32)
            # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
            image_np_ = np.transpose(image_np_, (1, 2, 0))
            if channel == 0:
                image_np = image_np_[:, :, :, np.newaxis]
            else:
                image_np = np.append(image_np, image_np_[:, :, :, np.newaxis], axis=-1)

        label_np = sitk.GetArrayFromImage(sample['label'])
        label_np = np.asarray(label_np, np.int32)
        # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
        label_np = np.transpose(label_np, (1, 2, 0))
        label_np = label_np[:, :, :, np.newaxis]
        return image_np / 255, label_np


class Normalization:
    def __init__(self, min_output, max_output):
        self.name = 'Normalization'
        self.min_output = float(min_output)
        self.max_output = float(max_output)

    def __call__(self, sample):
        resacleFilter = sitk.RescaleIntensityImageFilter()
        resacleFilter.SetOutputMinimum(self.min_output)
        resacleFilter.SetOutputMaximum(self.max_output)
        image, label = sample['image'], sample['label']
        image = resacleFilter.Execute(image)

        return {'image': image, 'label': label}


class StatisticalNormalization:
    """
    Normalize an image by mapping intensity with intensity distribution
    """

    def __init__(self, sigma, pre_norm=False):
        self.name = 'StatisticalNormalization'
        assert isinstance(sigma, float)
        self.sigma = sigma
        self.pre_norm = pre_norm

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        for image_channel in range(len(image)):
            if self.pre_norm:
                normalFilter = sitk.NormalizeImageFilter()
                image[image_channel] = normalFilter.Execute(image[image_channel])

            statisticsFilter = sitk.StatisticsImageFilter()
            statisticsFilter.Execute(image[image_channel])

            intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
            intensityWindowingFilter.SetOutputMaximum(255)
            intensityWindowingFilter.SetOutputMinimum(0)
            intensityWindowingFilter.SetWindowMaximum(
                statisticsFilter.GetMean() + self.sigma * statisticsFilter.GetSigma())
            intensityWindowingFilter.SetWindowMinimum(
                statisticsFilter.GetMean() - self.sigma * statisticsFilter.GetSigma())

            image[image_channel] = intensityWindowingFilter.Execute(image[image_channel])

        return {'image': image, 'label': label}


class ExtremumNormalization:
    """
    Normalize an image by mapping intensity with intensity maximum and minimum
    """

    def __init__(self, percent=0.05):
        self.name = 'ExtremumNormalization'
        assert isinstance(percent, float)
        self.percent = percent

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        for image_channel in range(len(image)):
            statisticsFilter = sitk.StatisticsImageFilter()
            statisticsFilter.Execute(image[image_channel])

            intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
            intensityWindowingFilter.SetOutputMaximum(255)
            intensityWindowingFilter.SetOutputMinimum(0)
            windowMax = (statisticsFilter.GetMaximum() - statisticsFilter.GetMinimum()) * (1 - self.percent) + \
                        statisticsFilter.GetMinimum()
            windowMin = (statisticsFilter.GetMaximum() - statisticsFilter.GetMinimum()) * self.percent + \
                        statisticsFilter.GetMinimum()
            intensityWindowingFilter.SetWindowMaximum(windowMax)
            intensityWindowingFilter.SetWindowMinimum(windowMin)

            image[image_channel] = intensityWindowingFilter.Execute(image[image_channel])

        return {'image': image, 'label': label}


class ManualNormalization:
    def __init__(self, min_window, max_window):
        self.name = 'ManualNormalization'
        self.min_window = float(min_window)
        self.max_window = float(max_window)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        intersity_window_filter = sitk.IntensityWindowingImageFilter()
        intersity_window_filter.SetOutputMinimum(0)
        intersity_window_filter.SetOutputMaximum(255)
        intersity_window_filter.SetWindowMinimum(self.min_window)
        intersity_window_filter.SetWindowMaximum(self.max_window)
        for _c in range(len(image)):
            image[_c] = intersity_window_filter.Execute(image[_c])
        return {'image': image, 'label': label}


class Reorient:
    def __init__(self, order):
        self.name = 'Reoreient'
        assert isinstance(order, (int, tuple))
        assert len(order) == 3
        self.order = order

    def __call__(self, sample):
        reorientFilter = sitk.PermuteAxesImageFilter()
        reorientFilter.SetOrder(self.order)
        image = reorientFilter.Execute(sample['image'])
        label = reorientFilter.Execute(sample['label'])

        return {'image': image, 'label': label}


class RandomFlip(object):
    def __init__(self, axes):
        self.name = 'Flip'
        assert len(axes) > 0 and len(axes) <= 3
        self.axes = axes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        _axes = np.where(np.random.random(3) > 0.5, True, False)
        axes = list()
        for i, _a in enumerate(self.axes):
            if _a and _axes[i]:
                axes.append(True)
            else:
                axes.append(False)
        flipFilter = sitk.FlipImageFilter()
        flipFilter.SetFlipAxes(axes)
        for image_channel in range(len(image)):
            image[image_channel] = flipFilter.Execute(image[image_channel])
        label = flipFilter.Execute(label)

        return {'image': image, 'label': label}


class Invert:
    """Invert the image intensity from 0-255
    """

    def __init__(self):
        self.name = 'Invert'

    def __call__(self, sample):
        invertFilter = sitk.InvertIntensityImageFilter()
        image = invertFilter.Execute(sample['image'], 255)
        label = sample['label']

        return {'image': image, 'label': label}


class Resample:
    def __init__(self, voxel_size):
        self.name = 'Resample'
        self.voxel_size = voxel_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        for _c in range(len(image)):
            old_spacing = image[_c].GetSpacing()
            old_size = image[_c].GetSize()

            new_spacing = self.voxel_size

            new_size = []
            for i in range(3):
                new_size.append(int(math.ceil(old_spacing[i] * old_size[i] / new_spacing[i])))
            new_size = tuple(new_size)

            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size)
            # resample on image
            resampler.SetOutputOrigin(image[_c].GetOrigin())
            resampler.SetOutputDirection(image[_c].GetDirection())
            image[_c] = resampler.Execute(image[_c])

        # resample on segmentation
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputOrigin(label.GetOrigin())
        resampler.SetOutputDirection(label.GetDirection())
        label = resampler.Execute(label)

        return {'image': image, 'label': label}


class Padding:
    def __init__(self, output_size):
        self.name = 'Padding'
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image[0].GetSize()

        if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (
                size_old[2] >= self.output_size[2]):
            return sample
        else:
            output_size = list(self.output_size)
            if size_old[0] > self.output_size[0]:
                output_size[0] = size_old[0]
            if size_old[1] > self.output_size[1]:
                output_size[1] = size_old[1]
            if size_old[2] > self.output_size[2]:
                output_size[2] = size_old[2]
            output_size = tuple(output_size)

            for _c in range(len(image)):
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(image[_c].GetSpacing())
                resampler.SetSize(output_size)
                # resample on image
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetOutputOrigin(image[_c].GetOrigin())
                resampler.SetOutputDirection(image[_c].GetDirection())
                image[_c] = resampler.Execute(image[_c])

            # resample on label
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetOutputOrigin(label.GetOrigin())
            resampler.SetOutputDirection(label.GetDirection())
            label = resampler.Execute(label)

            return {'image': image, 'label': label}


class RandomCrop:
    def __init__(self, output_size, is_train, drop_ratio=0.1, min_pixel=1):
        self.name = 'Random Crop'
        self.output_size = output_size
        self.is_train = is_train
        self.drop_ratio = drop_ratio
        self.min_pixel = min_pixel

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image[0].GetSize()
        size_new = self.output_size
        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
        binaryThresholdFilter.SetLowerThreshold(1)
        binaryThresholdFilter.SetUpperThreshold(255)
        binaryThresholdFilter.SetInsideValue(1)
        binaryThresholdFilter.SetOutsideValue(0)
        label_ = binaryThresholdFilter.Execute(label)

        while not contain_label:
            # get the start crop coordinate in ijk
            if size_old[0] <= size_new[0]:
                start_i = 0
            else:
                start_i = np.random.randint(0, size_old[0] - size_new[0])
            if size_old[1] <= size_new[1]:
                start_j = 0
            else:
                start_j = np.random.randint(0, size_old[1] - size_new[1])
            if size_old[2] <= size_new[2]:
                start_k = 0
            else:
                start_k = np.random.randint(0, size_old[2] - size_new[2])
            roiFilter.SetIndex([start_i, start_j, start_k])
            label_crop = roiFilter.Execute(label_)
            statFilter = sitk.StatisticsImageFilter()
            statFilter.Execute(label_crop)
            # will iterate until a sub volume containing label is extracted
            # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
            # if statFilter.GetSum()/pixel_count<self.min_ratio:
            if statFilter.GetSum() < self.min_pixel:
                contain_label = self.drop(self.drop_ratio)  # has some probabilty to contain patch with empty label
            else:
                contain_label = True
        for _c in range(len(image)):
            image[_c] = roiFilter.Execute(image[_c])
        label = roiFilter.Execute(label)

        return {'image': image, 'label': label}

    def drop(self, probability):
        return random.random() <= probability


class RandomNoise(object):
    def __init__(self):
        self.name = 'Random Noise'

    def __call__(self, sample):
        self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
        self.noiseFilter.SetMean(0)
        self.noiseFilter.SetStandardDeviation(0.1)

        image, label = sample['image'], sample['label']

        for _c in range(len(image)):
            image[_c] = self.noiseFilter.Execute(image[_c])

        return {'image': image, 'label': label}


class BSplineDeformation:
    """
    Image deformation with a sparse set of control points to control a free form deformation.
    Details can be found here:
    https://simpleitk.github.io/SPIE2018_COURSE/spatial_transformations.pdf
    https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html

    Args:
        randomness (int,float): BSpline deformation scaling factor, default is 10.
    """

    def __init__(self, randomness=10):
        self.name = 'BSpline Deformation'

        assert isinstance(randomness, (int, float))
        if randomness > 0:
            self.randomness = randomness
        else:
            raise RuntimeError('Randomness should be non zero values')

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        spline_order = 3
        domain_physical_dimensions = [image.GetSize()[0] * image.GetSpacing()[0],
                                      image.GetSize()[1] * image.GetSpacing()[1],
                                      image.GetSize()[2] * image.GetSpacing()[2]]

        bspline = sitk.BSplineTransform(3, spline_order)
        bspline.SetTransformDomainOrigin(image.GetOrigin())
        bspline.SetTransformDomainDirection(image.GetDirection())
        bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
        bspline.SetTransformDomainMeshSize((10, 10, 10))

        # Random displacement of the control points.
        originalControlPointDisplacements = np.random.random(len(bspline.GetParameters())) * self.randomness
        bspline.SetParameters(originalControlPointDisplacements)

        image = sitk.Resample(image, bspline)
        label = sitk.Resample(label, bspline)
        return {'image': image, 'label': label}

    def NormalOffset(self, size, sigma):
        s = np.random.normal(0, size * sigma / 2, 100)  # 100 sample is good enough
        return int(round(random.choice(s)))
