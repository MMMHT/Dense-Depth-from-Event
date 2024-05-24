import numpy as np
import torch

class Flow:
    """
    - parameters
        - calibration :: a Calibration object from calibration.py
    """

    def __init__(self, calibration):

        self.cal = calibration

        self.left_map = calibration.left_map
        self.right_map = calibration.right_map

        self.baseline = 0.1
        self.Pfx = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][0][0]
        self.Ppx = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][0][2]
        self.Pfy = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][1][1]
        self.Ppy = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][1][2]

        intrinsics = self.cal.intrinsic_extrinsic['cam0']['intrinsics']
        self.P = np.array([[self.Pfx, 0., self.Ppx],
                           [0., self.Pfy, self.Ppy],
                           [0., 0., 1.]])

        self.K = np.array([[intrinsics[0], 0., intrinsics[2]],
                           [0., intrinsics[1], intrinsics[3]],
                           [0., 0., 1.]])

        self.distortion_coeffs = np.array(self.cal.intrinsic_extrinsic['cam0']['distortion_coeffs'])
        self.resolution = self.cal.intrinsic_extrinsic['cam0']['resolution']

    # def compute_flow_single_frame(self, rectified_event_stack, V, Omega, disp_ind, essen, ts_curr):
    #     """
    #     params:
    #         V : [3,1]
    #         Omega : [3,1]
    #         disparity_image : [m,n]
    #     """
    #
    #     x_inds, y_inds = rectified_event_stack[:, 0, disp_ind], rectified_event_stack[:, 1, disp_ind]
    #     x_inds_ori, y_inds_ori = rectified_event_stack[:, 0, disp_ind], rectified_event_stack[:, 1, disp_ind]
    #     x_inds = x_inds.astype(np.float32)
    #     y_inds = y_inds.astype(np.float32)
    #
    #     x_inds -= self.P[0, 2]
    #     y_inds -= self.P[1, 2]
    #
    #     N = x_inds.shape[0]
    #
    #     focal_length = self.P[0, 0]
    #
    #     omega_mat = np.zeros((N, 2, 3))
    #
    #     omega_mat[:, 0, 0] = x_inds * y_inds / focal_length
    #     omega_mat[:, 1, 0] = focal_length + np.square(y_inds) / focal_length
    #
    #     omega_mat[:, 0, 1] = -focal_length  - np.square(x_inds) / focal_length
    #     omega_mat[:, 1, 1] = -(x_inds * y_inds) / focal_length
    #
    #     omega_mat[:, 0, 2] = y_inds / focal_length
    #     omega_mat[:, 1, 2] = -x_inds
    #
    #     disp_candidate = disp_ind * essen
    #
    #     disparity_mat = np.full((x_inds.shape[0]), disp_candidate)
    #     depth_mat = np.abs(self.P[0, 0] * self.baseline) / (disparity_mat + 1e-15)
    #
    #     dt = ts_curr - rectified_event_stack[:, 2, disp_ind]
    #
    #     fdm = 1. / depth_mat
    #     fxm = x_inds
    #     fym = y_inds
    #     omm = omega_mat
    #
    #     flat_x_flow_out = np.zeros(x_inds.shape[0])
    #     flat_x_flow_out = fdm * (fxm * V[2] - V[0] * focal_length)
    #     flat_x_flow_out += np.squeeze(np.dot(omm[:, 0, :], Omega))
    #
    #     flat_y_flow_out = np.zeros(y_inds.shape[0])
    #     flat_y_flow_out = fdm * (fym * V[2] - V[1] * focal_length)
    #     flat_y_flow_out += np.squeeze(np.dot(omm[:, 1, :], Omega))
    #
    #     flat_x_flow_out = flat_x_flow_out * dt
    #     flat_y_flow_out = flat_y_flow_out * dt
    #
    #     return flat_x_flow_out, flat_y_flow_out



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


    def _rectification_map(self):
        """Produces tables that map rectified coordinates to distorted coordinates.

           x_distorted = rectified_to_distorted_x[y_rectified, x_rectified]
           y_distorted = rectified_to_distorted_y[y_rectified, x_rectified]
        """
        dist_coeffs = self.cal.intrinsic_extrinsic['cam0']['distortion_coeffs']
        D = np.array(dist_coeffs)

        intrinsics = self.cal.intrinsic_extrinsic['cam0']['intrinsics']
        K = np.array([[intrinsics[0], 0., intrinsics[2]],
                      [0., intrinsics[1], intrinsics[3]], [0., 0., 1.]])
        K_new = np.array(self.cal.intrinsic_extrinsic['cam0']['projection_matrix'])[0:3, 0:3]

        R = np.array(self.cal.intrinsic_extrinsic['cam0']['rectification_matrix'])

        size = (self.cal.intrinsic_extrinsic['cam0']['resolution'][0],
                self.cal.intrinsic_extrinsic['cam0']['resolution'][1])

        rectified_to_distorted_x, rectified_to_distorted_y = cv2.fisheye.initUndistortRectifyMap(
            K, D, R, K_new, size, cv2.CV_32FC1)

        return rectified_to_distorted_x, rectified_to_distorted_y

    def _rectify_events(self, events, distorted_to_rectified, image_size):

        # rectified_events = []
        # width, height = image_size
        # for event in events:
        #     x, y, timestamp, polarity = event
        #     x_rectified = round(distorted_to_rectified[int(y), int(x)][0])
        #     y_rectified = round(distorted_to_rectified[int(y), int(x)][1])
        #     if (0 <= x_rectified < width) and (0 <= y_rectified < height):
        #         rectified_events.append(
        #             [ x_rectified, y_rectified, timestamp, polarity])
        # rectified_events = np.array(rectified_events)

        ################
        # parallel version
        width, height = image_size

        x_ori, y_ori, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

        x_coords = np.round(distorted_to_rectified[:, :, 0][events[:, 1].astype(int), events[:, 0].astype(int)])
        y_coords = np.round(distorted_to_rectified[:, :, 1][events[:, 1].astype(int), events[:, 0].astype(int)])

        valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)

        valid_indices = torch.from_numpy(valid_indices)

        x_rect = torch.masked_select(torch.from_numpy(x_coords), valid_indices)
        y_rect = torch.masked_select(torch.from_numpy(y_coords), valid_indices)
        t = torch.masked_select(torch.from_numpy(t), valid_indices)
        p = torch.masked_select(torch.from_numpy(p), valid_indices)

        events_rectified = torch.stack((x_rect, y_rect, t, p), dim=1).numpy()

        return events_rectified