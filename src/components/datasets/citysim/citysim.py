import os
import cv2
import h5py
import numpy as np
import torch
import yaml

from . import transforms
from src.components.datasets.mvsec.utils.grid import ev2grid
import src.components.datasets.citysim.constants as CONSTANTS


class CITYSIMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, subset, folder,  mode, crop_height, crop_width, max_disparity_pred, use_pre_data, events_num = 50000, max_interval = 20):

        self.debug = False
        self.use_half = True
        self.max_num_event = events_num
        self.max_interval = max_interval
        self.interval_s = 0.2
        self.max_depth = 80
        self.max_disparity_pred = max_disparity_pred
        self.disparity_limit = CONSTANTS._DISPARITY_LIMIT_
        self.essen = self.disparity_limit / self.max_disparity_pred

        self.safe_border = 50

        self.crop_height = crop_height
        self.crop_width = crop_width

        self.baseline = CONSTANTS._BASELINE_
        self._LOCATION_ = CONSTANTS._LOCATIONS_

        self.ymal_file = os.path.join(root_dir, folder, subset, 'calibration.yaml')

        if not os.path.isfile(os.path.join(root_dir, folder, subset, 'processed.h5')):
            depth_path = os.path.join(root_dir, folder, subset, 'depth_data.npy')
            depth_ts_path = os.path.join(root_dir,folder, subset, 'depth_time.npy')
            imu_path = os.path.join(root_dir, folder, subset, 'imu_data.npy')
            dvs_path = os.path.join(root_dir, folder, subset, 'dvs_data.npy')
            preprocessing.convert_data(depth_path= depth_path, depth_time_path= depth_ts_path,
                                       dvs_data_path = dvs_path, imu_path = imu_path)


        self.data =  h5py.File(os.path.join(root_dir, folder, subset, 'processed.h5'), 'r')

    #     generate calibration matrix
        with open(self.ymal_file) as yaml_file:
            intrinsic_extrinsic = yaml.load(yaml_file, Loader=yaml.SafeLoader)
        self.Pfx = intrinsic_extrinsic['cam0']['projection_matrix'][0][0]
        self.Ppx = intrinsic_extrinsic['cam0']['projection_matrix'][0][2]
        self.Pfy = intrinsic_extrinsic['cam0']['projection_matrix'][1][1]
        self.Ppy = intrinsic_extrinsic['cam0']['projection_matrix'][1][2]

        self.P = np.array([[self.Pfx, 0., self.Ppx],
                           [0., self.Pfy, self.Ppy],
                           [0., 0., 1.]])

        self.resolution = intrinsic_extrinsic['cam0']['resolution']

        self.events = np.array(self.data['left']['events'])
        self.ind_dvs_to_depth = np.array(self.data['left']['ind_dvs_to_depth'])
        self.gyro = np.array(self.data['left']['gyro'])
        self.linear_velocity = np.array(self.data['left']['linear_vel'])
        self.angular_velocity = np.array(self.data['left']['angular_vel'])
        self.imu_ts = np.array(self.data['left']['imu_ts'])
        self.depth_image_rect = np.array(self.data['left']['depth_image_rect'])
        self.depth_image_rect_ts = np.array(self.data['left']['depth_image_rect_ts'])
        # self.image = np.array(self.data['left']['image_raw'])
        # self.image_ts = np.array(self.data['left']['image_raw_ts'])

        # Gravitry Alignment
        # self.linear_velocity = np.zeros_like(self.linear_accerat.ion)
        # for i in range(self.linear_acceration.shape[0]):
        #     self.linear_velocity[i] = gravityAlignment(self.gyro[i], self.linear_acceration[i])

        filter_start, filter_end = CONSTANTS.FRAMES_FILTER[subset]
        self.dataset_offset = filter_start

        # the start point should be larger than the max_interval, such that enough pose can be used to generate the flow
        assert filter_start > self.max_interval

        self.velocity_set = {}
        self.velocity_set['linear_vel'] = [self.linear_velocity[i - max_interval:i] for i in range(filter_start, filter_end)]
        self.velocity_set['augular_vel'] = [self.angular_velocity[i - max_interval:i] for i in range(filter_start, filter_end)]
        self.velocity_set['imu_ts'] = [self.imu_ts[i - max_interval:i] for i in range(filter_start, filter_end)]

        self.dt_1 = {}
        self.dt_1['linear_vel'] = [self.linear_velocity[i:i+1] for i in range(filter_start, filter_end-1)]
        self.dt_1['augular_vel'] = [self.angular_velocity[i:i+1] for i in range(filter_start, filter_end-1)]
        self.dt_1['velocity_ts'] = [self.imu_ts[i:i+2] for i in range(filter_start, filter_end-1)]

        self.depth_set = {}
        self.depth_set['depth'] = torch.from_numpy(self.depth_image_rect[filter_start-1:filter_end-1])
        self.depth_set['depth_ts'] = torch.from_numpy(self.depth_image_rect_ts[filter_start-1:filter_end-1])
        self.depth_set['ind_dvs_to_depth'] = torch.from_numpy(self.ind_dvs_to_depth[filter_start-1:filter_end-1])

        # print('imu ts:{}, depth_ts:{}, dvs_ts:{}'.format(self.velocity_set['imu_ts'][0],
        #                                                  self.depth_set['depth_ts'][0],
        #                                                  self.events[self.depth_set['ind_dvs_to_depth'][0].int()]))

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
        disparity = MULTIPY_FACTOR * self.Pfx * self.baseline / (depth + 1e-7)
        invalid_nan = torch.isnan(disparity)
        invalid_inf = (disparity == float('inf'))
        invalid = invalid_nan | invalid_inf
        invalid_range = (disparity < 0) | (disparity > 255.0)
        invalid = invalid | invalid_range
        disparity[invalid] = 0
        return disparity


    def __len__(self):
        # print(len(self.depth_set['depth_ts']))
        return len(self.depth_set['depth_ts']) - 1


    def __getitem__(self, idx):

        ts = self.depth_set['depth_ts'][idx]

        # get events array
        ind = self.depth_set['ind_dvs_to_depth'][idx].int()
        assert ind > 0 and ind < self.events.shape[0]
        events_idx = self.clip_events(self.events, ts, self.interval_s, self.max_num_event, ind)
        if self.use_half:
            events_idx = events_idx[events_idx[:,3] == 1 ]

        # get disparity map
        depth_idx = self.depth_set['depth'][idx]
        depth_idx[depth_idx > self.max_depth] = 0
        disparity_idx = self.__depth_to_disparity(depth_idx)

        output = {}
        #  save relevant info

        for location in self._LOCATION_:

            # store the params
            output['params'] = {}
            output['params']['file_name'] = 'SimEvent' + '.png'
            output['params']['f_x_baseline'] = self.Pfx * self.baseline
            output['params']['essen'] = torch.tensor(self.essen).float()
            output['params']['max_disparity_pred'] = torch.tensor(self.max_disparity_pred).float()
            output['params']['max_disp'] = torch.tensor(self.essen * self.max_disparity_pred).float()
            # save dt1 velocity, and timestamp for the next time interval optical flow
            output['params']['dt1_linear_vel'] = self.dt_1['linear_vel'][idx].astype(np.float32).transpose(1,0)
            output['params']['dt1_augular_vel'] = self.dt_1['augular_vel'][idx].astype(np.float32).transpose(1,0)
            output['params']['dt1_velocity_ts'] = self.dt_1['velocity_ts'][idx][-1] - self.dt_1['velocity_ts'][idx][0]
            output['params']['p'] = self.P

            output['event'] = {}
            output['disparity'] = {}
            output['depth'] = {}
            output['flow'] = {}

            # apply the disparity limit
            disparity_idx[disparity_idx > self.disparity_limit] = 0
            disparity_idx = disparity_idx.unsqueeze(0).float().numpy()
            output['disparity'][location] = disparity_idx
            # print(disparity_idx.max())
            # print(disparity_idx.min())

            # apply the depth limit
            output['depth'][location] = self.depth_set['depth'][idx].unsqueeze(0).float()
            output['depth'][location][torch.isnan(output['depth'][location])] = 0
            output['depth'][location][torch.isinf(output['depth'][location])] = 0
            output['depth'][location] = output['depth'][location].numpy()

            # calc the gt flow
            # gt_x_flow, gt_y_flow = compute_flow_single_frame(output['params']['dt1_linear_vel']
            #                                                  , output['params']['dt1_augular_vel']
            #                                                  , torch.from_numpy(self.flow.P)
            #                                                  , torch.from_numpy(output['depth'][location])
            #                                                  , output['params']['dt1_velocity_ts'])
            # gt_x_flow = gt_x_flow.reshape(1, self.resolution[1], self.resolution[0])
            # gt_y_flow = gt_y_flow.reshape(1, self.resolution[1], self.resolution[0])
            # gt_flow = torch.cat((gt_x_flow, gt_y_flow), dim=0)
            # output['flow'][location] = gt_flow.numpy()

            # apply the dynamic motion model
            events_flow = np.tile(events_idx[:, :, np.newaxis], (1, 1, self.max_disparity_pred))
            # if events generated before the earliest pose of timestamp, abandon them
            if events_flow[0][2][0] < self.velocity_set['imu_ts'][idx][0]:
                for iter in range(events_flow.shape[0]):
                    if events_flow[iter][2][0] > self.velocity_set['imu_ts'][idx][0]:
                        events_flow = events_flow[iter:]
                        break

            for i_velocity_set in range(self.velocity_set['imu_ts'][idx].shape[0]):

                # abandon the first pose, since it is the start of the interval
                if i_velocity_set == 0:
                    continue

                velocity_set_ts = self.velocity_set['imu_ts'][idx][i_velocity_set].item()

                # no event fall into this interval
                if events_flow[0][2][0] > velocity_set_ts:
                    # print('skip interval:{}'.format( i_velocity_set))
                    continue

                index_event_end = torch.searchsorted(torch.from_numpy(np.ascontiguousarray(events_flow[:,2,0])), velocity_set_ts, right=True)

                events_subset = events_flow[:index_event_end]

                V = self.velocity_set['linear_vel'][idx][i_velocity_set]
                Omega = self.velocity_set['augular_vel'][idx][i_velocity_set]

                for ind_disp in range(self.max_disparity_pred):
                    flat_x_flow, flat_y_flow = self.compute_flow_single_frame(events_subset, V, Omega,
                                                                                   ind_disp, self.essen,
                                                                                   velocity_set_ts)

                    events_flow[:index_event_end, 0, ind_disp] += flat_x_flow
                    events_flow[:index_event_end, 1, ind_disp] += flat_y_flow
                    events_flow[:index_event_end, 2, ind_disp] = velocity_set_ts

            grid_image_disp = torch.zeros(self.max_disparity_pred, self.resolution[1], self.resolution[0])

            for i in range(self.max_disparity_pred):
                p = events_flow[:, 3, i].astype('float32')
                t = events_flow[:, 2, i].astype('float32')
                x = events_flow[:, 0, i].astype('float32')
                y = events_flow[:, 1, i].astype('float32')
                event_stack = np.stack((t, x, y, p), axis=1)
                # img = events_to_timestamp_image(event_stack, 1, self.resolution[0], self.resolution[1])
                img = ev2grid(event_stack, 1, self.resolution[0], self.resolution[1])
                # img = time_occupied_grid(event_stack, 1, self.resolution[0], self.resolution[1])
                grid_image_disp[i] = img

            grid_image_disp = (grid_image_disp - grid_image_disp.min()) / (
                        grid_image_disp.max() - grid_image_disp.min())
            output['event'][location] = grid_image_disp.numpy()

            rows, cols = 4, 5
            images_num = 16
            width, height = self.resolution
            spacing = 10  # gap between images
            big_width = cols * width + (cols - 1) * spacing
            big_height = rows * height + (rows - 1) * spacing
            big_image = np.zeros((big_height, big_width), dtype=np.uint8)

            for i in range(rows):
                for j in range(cols):
                    index = i * cols + j
                    if index < images_num:
                        y_offset = i * (height + spacing)
                        x_offset = j * (width + spacing)
                        big_image[y_offset:y_offset + height, x_offset:x_offset + width] = grid_image_disp[
                                                                                               index * 2] * 255
                    else:
                        #             put gt
                        y_offset = i * (height + spacing)
                        x_offset = (j + 1) * (width + spacing)
                        gt = output['disparity']['left']
                        gt = gt / gt.max() * 255
                        big_image[y_offset:y_offset + height, x_offset:x_offset + width] = gt[0].astype('uint8') * 255
                        #             put raw events
                        y_offset = i * (height + spacing)
                        x_offset = (j + 2) * (width + spacing)
                        p = events_idx[:, 3]
                        t = events_idx[:, 2]
                        x = events_idx[:, 0]
                        y = events_idx[:, 1]
                        event_stack = np.stack((t, x, y, p), axis=1)
                        img_new = ev2grid(event_stack, 1, self.resolution[0], self.resolution[1])
                        img_new = (img_new - img_new.min()) / (img_new.max() - img_new.min()) * 255
                        big_image[y_offset:y_offset + height, x_offset:x_offset + width] = img_new[0].numpy().astype(
                            'uint8')
                        #             put gt and raw events combined image
                        y_offset = i * (height + spacing)
                        x_offset = (j + 3) * (width + spacing)
                        combined = cv2.addWeighted(img_new[0].numpy().astype('uint8'), 0.7, gt[0].astype('uint8') * 255,
                                                   0.3, 0)
                        big_image[y_offset:y_offset + height, x_offset:x_offset + width] = combined
                        # cv2.imshow('big', combined)
                        break


            output['optical_input'] = torch.from_numpy(big_image).unsqueeze(0).float()

            if self.debug:

                #  depth
                depth_idx = output['depth']['left']
                depth_idx[depth_idx >255]  = 0
                cv2.imshow('depth', depth_idx[0].astype(np.uint8))


                # dispaly disparity
                disp = output['disparity']['left']
                disp_norm = (disp - disp.min()) / (disp.max() - disp.min()) * 255
                print(disp.max())
                disp = disp * 4
                cv2.imshow('disparity_ori', disp[0].astype(np.uint8))
                cv2.imshow('disparity_norm', disp_norm[0].astype(np.uint8))

                # display optical input
                # cv2.imshow('optical_input', output['optical_input'][0].numpy().astype('uint8'))

                num_windows = 15
                window_width = 346
                window_height = 260
                rows = 3
                cols = 5
                spacing_x = 100
                spacing_y = 150

                p = events_idx[:, 3]
                t = events_idx[:, 2]
                x = events_idx[:, 0]
                y = events_idx[:, 1]
                event_stack = np.stack((t, x, y, p), axis=1)
                img_new = ev2grid(event_stack, 1, self.resolution[0], self.resolution[1])
                # img_new_np = img_new.numpy()
                # cv2.imshow('events_ori', img_new[0].numpy().astype(np.uint8))
                img_new = (img_new - img_new.min()) / (img_new.max() - img_new.min()) * 255
                # cv2.imshow('events_norm', img_new[0].numpy().astype(np.uint8))
                cv2.imshow('raw_events', img_new[0].numpy().astype(np.float32))

                # combined = cv2.addWeighted(img_new[0].numpy().astype(np.uint8), 0.7, disp_norm[0].astype(np.uint8),
                #                            0.3, 0)
                # cv2.imshow('combined', combined)

                for i in range(0, self.max_disparity_pred, 3):
                    img_32 = torch.from_numpy(output['event'][location][i:i + 1])
                    img_32 = img_32 * 255

                    iter = int(i / 3)
                    row = iter // cols
                    col = iter % cols
                    x_c = col * (window_width + spacing_x)
                    y_c = row * (window_height + spacing_y)
                    window_name = "disp {}".format(i*self.essen)
                    # window_name_var = "Window {} var".format(i)
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    # cv2.namedWindow(window_name_var, cv2.WINDOW_NORMAL)
                    cv2.moveWindow(window_name, x_c, y_c)
                    # cv2.moveWindow(window_name_var, x_c, y_c + 100)
                    cv2.imshow(window_name, img_32[0].numpy().astype(np.float32))
                    # cv2.imshow(window_name_var, img_var_32[0].numpy().astype('uint8'))

                cv2.waitKey(0)

        output = self.transforms(output)
        return output
        # return data


    def clip_events(self, events, ts, interval_s, max_num_event, event_index = None):
        """
        params:
            events: [n, 4], timestamp, x, y, polarity
            ts: float, timestamp of ground truth depth
        """
        if event_index == None:
            raise NotImplementedError
        else:
            end_index = event_index.item()
            ts = ts.item()
            for i in range(event_index, 0, -1):
                if not ts - events[i, 2] <= interval_s:
                    start_index = i
                    break

            events = events[start_index:end_index+self.safe_border]

            # delete event to prevent too many event
            if events.shape[0] > max_num_event :
                clip_step = events.shape[0] / max_num_event
                events = events[::int(clip_step)]

            return events

    def compute_flow_single_frame(self, rectified_event_stack, V, Omega, disp_ind, essen, ts_curr):
        """
        params:
            V : [3,1]
            Omega : [3,1]
            disparity_image : [m,n]
        """

        x_inds, y_inds = rectified_event_stack[:, 0, disp_ind], rectified_event_stack[:, 1, disp_ind]
        x_inds_ori, y_inds_ori = rectified_event_stack[:, 0, disp_ind], rectified_event_stack[:, 1, disp_ind]
        x_inds = x_inds.astype(np.float32)
        y_inds = y_inds.astype(np.float32)

        x_inds -= self.P[0, 2]
        x_inds *= (1. / self.P[0, 0])

        y_inds -= self.P[1, 2]
        y_inds *= (1. / self.P[1, 1])

        flat_x_map = x_inds.reshape((-1))
        flat_y_map = y_inds.reshape((-1))

        N = flat_x_map.shape[0]

        omega_mat = np.zeros((N, 2, 3))

        omega_mat[:, 0, 0] = flat_x_map * flat_y_map
        omega_mat[:, 1, 0] = 1 + np.square(flat_y_map)

        omega_mat[:, 0, 1] = -(1 + np.square(flat_x_map))
        omega_mat[:, 1, 1] = -(flat_x_map * flat_y_map)

        omega_mat[:, 0, 2] = flat_y_map
        omega_mat[:, 1, 2] = -flat_x_map

        disp_candidate = disp_ind * essen

        disparity_mat = np.full((flat_x_map.shape[0]), disp_candidate)
        disparity_mat = np.abs(self.P[0, 0] * self.baseline) / (disparity_mat + 1e-15)

        dt = ts_curr - rectified_event_stack[:, 2, disp_ind]

        fdm = 1. / disparity_mat
        fxm = flat_x_map
        fym = flat_y_map
        omm = omega_mat

        flat_x_flow_out = np.zeros(flat_x_map.shape[0])
        flat_x_flow_out = fdm * (fxm * V[2] - V[0])
        flat_x_flow_out += np.squeeze(np.dot(omm[:, 0, :], Omega))

        flat_y_flow_out = np.zeros(flat_y_map.shape[0])
        flat_y_flow_out = fdm * (fym * V[2] - V[1])
        flat_y_flow_out += np.squeeze(np.dot(omm[:, 1, :], Omega))

        flat_x_flow_out = flat_x_flow_out * self.P[0, 0] * dt
        flat_y_flow_out = flat_y_flow_out * self.P[1, 1] * dt

        # x_flow_update = x_inds_ori + flat_x_flow_out
        # y_flow_update = y_inds_ori + flat_y_flow_out

        return flat_x_flow_out, flat_y_flow_out

def get_dataloader(args, dataset_cfg, dataloader_cfg, is_distributed=False):

    dataset_sequences = list()
    subfolder_list = os.listdir(os.path.join(args.data_root, dataset_cfg.FOLDER ))
    subfolder_list.sort()
    for subfolder in subfolder_list:
        if dataset_cfg.PARAMS.mode == 'train':
            if subfolder not in CONSTANTS.Training_set:
                continue
        elif dataset_cfg.PARAMS.mode == 'test':
            if subfolder not in CONSTANTS.Testing_set:
                continue
        print('loading {}'.format(subfolder))
        dataset = CITYSIMDataset(root_dir=args.data_root, folder=dataset_cfg.FOLDER, subset = subfolder,  **dataset_cfg.PARAMS)
        dataset_sequences.append(dataset)

    loder =  torch.utils.data.ConcatDataset(dataset_sequences)
    dataloader = globals()[dataloader_cfg.NAME](dataset=loder,
                                          dataloader_cfg=dataloader_cfg,
                                          num_workers=args.num_workers,
                                          is_distributed=is_distributed,
                                          world_size=args.world_size if is_distributed else None, )

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

#
# def gravityAlignment(orientation, linear_acc):
#     # based on Z-up left-handed system used in carla
#     linear_acc_x, linear_acc_y, linear_acc_z = linear_acc
#     yaw, pitch, roll = orientation
#     accX = linear_acc_y - math.sin(math.radians(roll)) * math.cos(math.radians(pitch)) * 9.81
#     # print('linear_acc_y: {}, roll:{}, pitch:{}, partial:{}, accX:{}'.format(linear_acc_y, roll, pitch, math.sin(math.radians(roll)) * math.cos(math.radians(pitch)) * 9.81, accX))
#     accY = linear_acc_z - math.cos(math.radians(roll)) * math.cos(math.radians(pitch)) * 9.81
#     # print('linear_acc_z: {}, roll:{}, pitch:{}, partial:{}, accY:{}'.format(linear_acc_z, roll,pitch, math.cos(math.radians(roll)) * math.cos(math.radians(pitch)) * 9.81,accY))
#     accZ = linear_acc_x + math.sin(math.radians(pitch)) * 9.81
#     # print('linear_acc_x: {}, pitch:{},partial:{}, accZ:{}'.format(linear_acc_x, pitch, math.sin(math.radians(pitch)) * 9.81,accZ))
#
#     return [accX, accY, accZ]
