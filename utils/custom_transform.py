from __future__ import division
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, imus, intrinsics):
        for t in self.transforms:
            images, imus, intrinsics = t(images, imus, intrinsics)
        return images, imus, intrinsics
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, imus, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, imus, intrinsics
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'mean: {}, '.format(self.mean)
        format_string += 'std: {})\n'.format(self.std)
        return format_string

class ToTensor(object):
    def __call__(self, images, imus, gts):
        tensors = []
        for im in images:
            tensors.append(TF.to_tensor(im)- 0.5)
        tensors = torch.stack(tensors, 0)
        return tensors, imus, gts
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Resize(object):
    def __init__(self, size=(256, 512)):
        self.size = size

    def __call__(self, images, imus, gts):
        tensors = [TF.resize(im, size=self.size) for im in images]
        tensors = torch.stack(tensors, 0)
        return tensors, imus, gts
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'img_h: {}, '.format(self.size[0])
        format_string += 'img_w: {})'.format(self.size[1])
        return format_string

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, images, imus, gts):
        if random.random() < self.p:
            tensors = [TF.hflip(im) for im in images]
            tensors = torch.stack(tensors, 0)
            # Adjust imus and target poses according to horizontal flips
            imus[:, 1], imus[:, 3], imus[:, 5] = -imus[:, 1], -imus[:, 3], -imus[:, 5]
            gts[:, 1], gts[:, 2], gts[:, 3] = -gts[:, 1], -gts[:, 2], -gts[:, 3]
        else:
            tensors = images
        return tensors, imus, gts
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p: {})'.format(self.p)
        return format_string

class RandomColorAug(object):
    def __init__(self, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2], p=0.5):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2
        self.p = p
    def __call__(self, images, imus, gts):
        if random.random() < self.p:
            images = images + 0.5
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            
            # randomly shift gamma
            img_aug = images ** random_gamma
            
            # randomly shift brightness
            img_aug = img_aug * random_brightness
            
            # randomly shift color
            for i in range(3):
                img_aug[:, i, :, :] *= random_colors[i]
            
            # saturate
            img_aug = torch.clamp(img_aug, 0, 1) - 0.5

        else:
            img_aug = images

        return img_aug, imus, gts
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'gamma: {}-{}, '.format(self.gamma_low, self.gamma_high)
        format_string += 'brightness: {}-{}, '.format(self.brightness_low, self.brightness_high)
        format_string += 'color shift: {}-{}, '.format(self.color_low, self.color_high)
        format_string += 'p: {})'.format(self.p)
        return format_string
