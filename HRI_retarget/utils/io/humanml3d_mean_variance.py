import numpy as np
import sys
import os
from os.path import join as pjoin


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir, joints_num, links_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(links_num - 1) * 3] = Std[4: 4+(links_num - 1) * 3].mean() / 1.0
    Std[4+(links_num - 1) * 3: 4+(links_num - 1) * 3 + joints_num] = Std[4+(links_num - 1) * 3: 4+(links_num - 1) * 3 + joints_num].mean() / 1.0
    Std[4+(links_num - 1) * 3 + joints_num: 4+(links_num - 1) * 6 + joints_num + 3] = Std[4+(links_num - 1) * 3 + joints_num: 4+(links_num - 1) * 6 + joints_num + 3].mean() / 1.0
    Std[4+(links_num - 1) * 6 + joints_num: ] = Std[4+(links_num - 1) * 6 + joints_num: ].mean() / 1.0

    assert 8 + (links_num - 1) * 6 + joints_num + 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std

if __name__ == '__main__':
    data_dir = '${HOME}/HRI_retarget/data/G1ML3D/new_joint_vecs'
    save_dir = '${HOME}/HRI_retarget/data/G1ML3D'
    mean, std = mean_variance(data_dir, save_dir, 29, 41)