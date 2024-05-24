import math

import numpy as np
import h5py
from tqdm import tqdm

depth_path = ''
depth_time_path = ''
dvs_data_path = ''
imu_path = ''

imu = np.load(imu_path, allow_pickle=True)
depth = np.load(depth_path, allow_pickle=True)
depth_time = np.load(depth_time_path, allow_pickle=True)
dvs_data = np.load(dvs_data_path, allow_pickle=True)

# depth data processing
depth_data_set = []
depth_ts = np.zeros((depth_time.shape[0]))
print('processing depth data')
for i in tqdm(range(depth_time.shape[0])):
    depth_ts[i] = depth_time[i]
    depth_0 = depth[i]

    depth_0_reshape = np.zeros((1, depth_0.shape[0], depth_0.shape[1]))
    depth_0_reshape[:] = depth_0
    depth_data_set.append(depth_0_reshape)

depth_data_np = np.concatenate(depth_data_set, axis=0)

#  imu data processing
imu_t = imu[:, 0]
imu = imu[:, 1]
imu_ts = np.zeros((imu.shape[0]))
gyro = np.zeros((imu.shape[0], 3))
linear_vel = np.zeros((imu.shape[0], 3))
angular_vel = np.zeros((imu.shape[0], 3))
print('processing imu data')
for i in tqdm(range(imu.shape[0] - 1)):
    imu_ts[i] = imu_t[i]

    # gyro
    gyro[i] = imu[i][2]

    # linear velocity
    linear_x, linear_y, linear_z = imu[i][3]
    linear_vel[i][0] = linear_y
    linear_vel[i][1] = - linear_z
    linear_vel[i][2] = linear_x

    # angular velocity
    angular_x, angular_y, angular_z = imu[i][1]
    angular_vel[i][0] = angular_x  * math.pi / 180
    angular_vel[i][1] = angular_z  * math.pi / 180
    angular_vel[i][2] = angular_y  * math.pi / 180

#  dvs data processing
print('processing dvs data')
dvx_data_set = []
pre_dvs_ts = 0

for i in tqdm(range(dvs_data.shape[0])):
    dvs_0 = dvs_data[i][1]
    dvs_0_reshape = np.zeros((dvs_0.shape[0], 4))
    for i in range(dvs_0.shape[0]):
        x, y, t, p = dvs_0[i]
        dvs_0_reshape[i, :] = [x, y, t.astype(np.float32), p.astype(np.float32)]
    assert pre_dvs_ts <= dvs_0_reshape[-1][2]
    pre_dvs_ts = dvs_0_reshape[-1][2]
    dvx_data_set.append(dvs_0_reshape)
dvs_data_np = np.concatenate(dvx_data_set, axis=0)

#  generate index for dvs to depth
ind = 0
ind_dvs_to_depth = np.zeros_like(depth_ts)
for i in tqdm(range(depth_ts.shape[0])):
    for j in range(ind, dvs_data_np.shape[0]):
        if dvs_data_np[j][2] > depth_ts[i]:
            ind_dvs_to_depth[i] = j
            ind = j
            break

h5_path = depth_path.replace('depth_data.npy', 'processed.h5' )

with h5py.File(h5_path, 'w') as f:
    left_group = f.create_group('left')
    left_group.create_dataset('events', data=dvs_data_np)

    left_group.create_dataset('gyro', data=gyro)
    left_group.create_dataset('linear_vel', data=linear_vel)
    left_group.create_dataset('angular_vel', data=angular_vel)
    left_group.create_dataset('imu_ts', data=imu_ts)

    left_group.create_dataset('depth_image_rect', data=depth_data_np)
    left_group.create_dataset('depth_image_rect_ts', data=depth_ts)
    left_group.create_dataset('ind_dvs_to_depth', data=ind_dvs_to_depth)

    f.close()
