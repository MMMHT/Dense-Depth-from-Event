import os
import time

import cv2
import h5py
import numpy as np
import torch

from . import transforms
from .constants import _LOCATIONS_

from .utils import calibration
from src.components.datasets.mvsec.utils.flow_module import Flow
from src.components.datasets.mvsec.utils.grid import ev2grid
import src.components.datasets.mvsec.Preprocessing as Preprocessing
import src.components.datasets.mvsec.constants as CONSTANTS

class MVSEC_Indoorflying_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, folder, subset, mode, max_disparity_pred, crop_height, crop_width, use_pre_data, events_num = 30000, max_interval_cnt = 20, max_interval = 300):

        self.debug = False
        self.use_half = False
        self.use_pre_data = use_pre_data
        self.root_dir = root_dir
        self.subset = str(subset)
        self.dataset = str(subset[:-1])
        self.subset_sel = str(subset[-1:])
        self.mode = mode
        self.max_num_event = events_num
        self.interval_s = 0.2
        self.max_disparity_pred = max_disparity_pred
        self.disparity_limit = CONSTANTS._DISPARITY_LIMIT_[self.subset]
        self.essen = self.disparity_limit / self.max_disparity_pred

        self.crop_height = crop_height
        self.crop_width = crop_width

        self._LOCATION_ = _LOCATIONS_

        self.cal = calibration.Calibration(self.dataset)
        self.flow = Flow(self.cal)

        self.resolution = self.flow.resolution

        self.data_path = os.path.join(root_dir, folder, subset)
        self._len = len(os.listdir(self.data_path))

        if mode in ['train']:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(crop_height=crop_height, crop_width=crop_width),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        elif mode in ['test', 'val']:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise NotImplementedError
    def __depth_to_disparity(self, depth):
        MULTIPY_FACTOR = 1
        disparity = MULTIPY_FACTOR * self.flow.Pfx * self.flow.baseline / (depth + 1e-7)
        invalid_nan = torch.isnan(disparity)
        invalid_inf = (disparity == float('inf'))
        invalid = invalid_nan | invalid_inf
        invalid_range = (disparity < 0) | (disparity > 255.0)
        invalid = invalid | invalid_range
        disparity[invalid] = 0
        return disparity

    def __getitem__(self, index):

        # get the h5 file
        data = h5py.File(os.path.join(self.data_path, '{}.h5'.format(index)), 'r')

        output = {}

        for location in self._LOCATION_:
            # store the params
            output['params'] = {}
            output['params']['f_x_baseline'] = self.flow.Pfx * self.flow.baseline
            output['params']['essen'] = torch.tensor(self.essen).float()
            output['params']['max_disparity_pred'] = torch.tensor(self.max_disparity_pred).float()
            output['params']['max_disp'] = torch.tensor(self.essen * self.max_disparity_pred).float()
            output['params']['p'] = self.flow.P

            output['event'] = {}
            output['disparity'] = {}
            output['depth'] = {}

            # get the data
            output['event'][location] = np.array(data[location] ['events'])
            output['depth'][location]  = np.array(data[location] ['depth'])
            output['disparity'][location] = np.array(data[location] ['disparity'])

        return output

    def __len__(self):
        return self._len


def get_dataloader(args, dataset_cfg, dataloader_cfg, is_distributed=False):

    dataset_sequences = list()
    subfolder_list = os.listdir(os.path.join(args.data_root, dataset_cfg.FOLDER))
    subfolder_list.sort()
    for subfolder in subfolder_list:
        if dataset_cfg.PARAMS.mode == 'train':
            if subfolder not in CONSTANTS.Training_set:
                continue
        elif dataset_cfg.PARAMS.mode == 'test':
            if subfolder not in CONSTANTS.Testing_set:
                continue
        print('loading {}'.format(subfolder))
        dataset_sequences.append(MVSEC_Indoorflying_Dataset(root_dir=args.data_root, folder=dataset_cfg.FOLDER, subset = subfolder, **dataset_cfg.PARAMS))


    loder =  torch.utils.data.ConcatDataset(dataset_sequences)
    dataloader = globals()[dataloader_cfg.NAME](dataset=loder,
                                                dataloader_cfg=dataloader_cfg,
                                                num_workers=args.num_workers,
                                                is_distributed=is_distributed,
                                                world_size=args.world_size if is_distributed else None,)

    return dataloader

def get_multi_epochs_dataloader(dataset, dataloader_cfg, num_workers, is_distributed, world_size):

    if is_distributed:
        raise  NotImplementedError
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              **dataloader_cfg.PARAMS)


    return dataloader





