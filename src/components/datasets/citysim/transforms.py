import numpy as np
import torch
from .constants import _LOCATIONS_

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor:

    def __call__(self, sample):
        # if 'event' in sample.keys():
        for domain in ['event', 'disparity', 'depth']:
            for location in _LOCATIONS_:
                if isinstance(sample[domain][location], np.ndarray):
                    sample[domain][location] = torch.from_numpy(sample[domain][location])

        # if 'disparity' in sample.keys():
        # sample['disparity'] = torch.from_numpy(sample['disparity'])

        return sample


class RandomCrop():
    def __init__(self,  crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample):
        if 'event' in sample:
            ori_height, ori_width = sample['event']['left'].shape[-2:]
        else:
            raise NotImplementedError

        assert self.crop_height <= ori_height and self.crop_width <= ori_width

        offset_x = np.random.randint(ori_width - self.crop_width + 1)
        offset_y = np.random.randint(ori_height - self.crop_height + 1)

        # if 'event' in sample.keys():
        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width
        for domain in ['event', 'disparity', 'depth']:
            for location in _LOCATIONS_ :
                sample[domain][location] = sample[domain][location][:, start_y:end_y, start_x:end_x]

        # if 'disparity' in sample.keys():
        # start_y, end_y = offset_y, offset_y + self.crop_height
        # start_x, end_x = offset_x, offset_x + self.crop_width
        #
        # sample['disparity'] = sample['disparity'][:, start_y:end_y, start_x:end_x]
        return sample



class RandomVerticalFlip:

    def __call__(self, sample):
        if np.random.random() < 0.5:
            for domain in ['event', 'disparity', 'depth']:
                for location in _LOCATIONS_:
                    sample[domain][location] = np.copy(np.flip(sample[domain][location], axis=1))


        return sample
