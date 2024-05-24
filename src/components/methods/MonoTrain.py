import time

import torch
import os
import torch.nn.functional as F
import torch.distributed as dist


from tqdm import tqdm
from collections import OrderedDict

from src.utils.metric import (AverageMeter, EndPointError, NPixelError, RootMeanSquareError,
                          AbsoluteRelativeError, RootMeanSquareErrorLog, RelativeSquaredError,Accuracy, DepthError)
from src.utils import visualizer



def evaluation(model, data_loader, logger, device, sequence_name):
    # print('eva')
    model.eval()

    log_dict = OrderedDict([

        ('absrel', AbsoluteRelativeError(average_by='image', string_format='%6.3lf')),
        ('sqrel', RelativeSquaredError(average_by='image', string_format='%6.3lf')),
        ('RMSE', RootMeanSquareError(average_by='image', string_format='%6.3lf')),
        ('RMSElog', RootMeanSquareErrorLog(average_by='image', string_format='%6.3lf')),
        ('Acc1.25', Accuracy(threshold=1.25, average_by='image', string_format='%6.3lf')),
        ('Acc1.25^2', Accuracy(threshold=1.25 ** 2, average_by='image', string_format='%6.3lf')),
        ('Acc1.25^3', Accuracy(threshold=1.25 ** 3, average_by='image', string_format='%6.3lf')),
        ('AAD10', DepthError(average_by = 'image', string_format='%6.3lf')),
        ('AAD20', DepthError(average_by='image', string_format='%6.3lf')),
        ('AAD30', DepthError(average_by='image', string_format='%6.3lf')),
        ('EPE', EndPointError(average_by='image', string_format='%6.3lf')),
        ('1PE', NPixelError(n=1, average_by='image', string_format='%6.3lf')),
    ])

    for  iter, batch_data in enumerate(tqdm(data_loader)):

        batch_data = batch_to_cuda(batch_data, device)

        for domain in ['event', 'disparity','depth']:
            for location in ['left']:
                # pad to width and hight to 16 times
                if batch_data[domain][location].shape[2] % 4 != 0:
                    times = batch_data[domain][location].shape[2] // 4
                    top_pad = (times + 1) * 4 - batch_data[domain][location].shape[2]
                else:
                    top_pad = 0

                if batch_data[domain][location].shape[3] % 4 != 0:
                    times = batch_data[domain][location].shape[3] // 4
                    right_pad = (times + 1) * 4 - batch_data[domain][location].shape[3]
                else:
                    right_pad = 0

                batch_data[domain][location] = F.pad(batch_data[domain][location], (0, right_pad, top_pad, 0))

        bs, c, height, width = batch_data['event']['left'].shape
        params = batch_data['params']

        with torch.no_grad():

            pred, mask_val, warp_img = model(batch_data['event']['left'],
                           batch_data['depth']['left'], params)

        if top_pad != 0 or right_pad != 0:
            pred = pred[:,:, top_pad:, :-right_pad]
            batch_data['depth']['left'] = batch_data['depth']['left'][:,:, top_pad:, :-right_pad]
            batch_data['disparity']['left'] = batch_data['disparity']['left'][:,:, top_pad:, :-right_pad]

        disp_max = params['max_disp'][0].item()

        pred_depth = params['f_x_baseline'][0] / (pred + 1e-7)
        pred_depth[pred_depth>100] = 0
        pred_depth = pred_depth.squeeze(1).cpu()
        gt_depth = batch_data['depth']['left'].squeeze(1)
        invalid_inf = torch.isinf(gt_depth)
        invalid_nan = torch.isnan(gt_depth)
        mask_zero = gt_depth == 0
        mask = ~invalid_inf & ~invalid_nan & ~mask_zero

        pred = pred.squeeze(1).cpu()
        gt_disparity = batch_data['disparity']['left'].squeeze(1).cpu()
        mask_zero = gt_disparity == 0
        disp_mask = ~mask_zero  & ~torch.isinf(gt_disparity) & ~torch.isnan(gt_disparity)

        log_dict['absrel'].update(pred_depth, gt_depth, mask)
        log_dict['sqrel'].update(pred_depth, gt_depth, mask)
        log_dict['RMSE'].update(pred_depth, gt_depth, mask)
        log_dict['RMSElog'].update(pred_depth, gt_depth, mask)
        log_dict['Acc1.25'].update(pred_depth, gt_depth, mask)
        log_dict['Acc1.25^2'].update(pred_depth, gt_depth, mask)
        log_dict['Acc1.25^3'].update(pred_depth, gt_depth, mask)
        log_dict['EPE'].update(pred_depth, gt_depth, mask)
        log_dict['1PE'].update(pred, gt_disparity, disp_mask)

        mask_m10 = gt_depth <= 10
        log_dict['AAD10'].update(pred_depth, gt_depth, mask & mask_m10)
        mask_m20 = gt_depth <= 20
        log_dict['AAD20'].update(pred_depth, gt_depth, mask & mask_m20)
        mask_m30 = gt_depth <= 30
        log_dict['AAD30'].update(pred_depth, gt_depth, mask & mask_m30)

        cur_pred = pred.cpu()
        warp_img = (warp_img - warp_img.min()) / ( warp_img.max() - warp_img.min()) * 255

        error_map = torch.abs(pred - gt_disparity)
        error_map[error_map <= 1] = 0
        error_map = error_map * disp_mask

        pred_list = {
            'pred': visualizer.tensor_to_disparity_image(cur_pred),
            'pred_magma': visualizer.tensor_to_disparity_magma_image(cur_pred, vmax=disp_max),
            'groundtruth': visualizer.tensor_to_disparity_magma_image(batch_data['disparity']['left'][0].cpu(), vmax=disp_max),
            'warp_img': visualizer.tensor_to_disparity_image(warp_img.cpu()),
            'error_map': visualizer.tensor_to_disparity_magma_image(error_map.squeeze(0).cpu(), vmax=disp_max),
        }

        for i in range(len(pred_list)):
            for key in pred_list:
                logger.save_visualize(image=pred_list[key],
                                       visual_type=key,
                                       sequence_name='{}'.format(sequence_name),
                                       image_name='{}'.format(iter).zfill(6) + '.png')


    return log_dict


def batch_to_cuda(batch_data, device):
    def _batch_to_cuda(batch_data):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key])
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.to(device)
        else:
            raise NotImplementedError

        return batch_data

    for domain in ['event', 'disparity']:
        if domain not in batch_data:
            batch_data[domain] = {}
        for location in ['left']:
            if location in batch_data[domain]:
                batch_data[domain][location] = _batch_to_cuda(batch_data[domain][location])
            else:
                batch_data[domain][location] = None

    return batch_data

