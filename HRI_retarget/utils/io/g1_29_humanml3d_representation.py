### converting 29 dof g1 to humanml3d vec representation 
### for g1_29, dof=29, num_links=41
### shape (num_frames - 1, 277)
###     root_rot_velocity 1
###     root_linear_velocity 2
###     root_y 1
###     link_local_position (41 - 1) * 3
###     dof_angle 29
###     link_local_velocity (41)* 3  ## TODO: this is an error, should be (41 - 1) * 3
###     foot_contact 4

### reference: https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb

### note that  "global_rotation" and "global_translation" may be changed for normalization

### usage: python src/utils/io/g1_29_humanml3d_representation.py <pickle file>
### example: python src/utils/io/g1_29_humanml3d_representation.py data/motion/g1/HumanML3D/000000.pickle


import sys
import os
from HRI_retarget.model.g1_29 import G1_29_Motion_Model
from HRI_retarget.config.joint_mapping import G1_LINKS
from HRI_retarget.utils.motion_lib.quaternion import qbetween_np, qinv_np, qmul_np, qrot_np
from HRI_retarget.utils.torch_utils.diff_quat import vec6d_to_matrix, vec6d_to_quat

import torch
from tqdm import tqdm
import pickle

import ipdb

import numpy as np

rot = torch.tensor([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=torch.float).to("cuda:0")


def data_pkl_to_vec(data_dict):
    """
    Convert the data_dict to vec representation
    :param data_dict: dict, contains the data
    :return: vec, the vec representation
    """
    num_frames = data_dict["angles"].shape[0]

    match data_dict["robot_name"]:
        case "g1_29":
            model = G1_29_Motion_Model(num_frames)
            robot_link = G1_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()

    model.set_angles(torch.tensor(data_dict["angles"]))
  

    # # # data_dict["global_rotation"] = (vec6d_to_matrix(torch.tensor(data_dict["global_rotation"]).to("cuda:0")) @ rot)[:, :, :2].detach().cpu().numpy()
    # # # data_dict["global_translation"] = (torch.tensor(data_dict["global_translation"]).to("cuda:0")[:, :, 0] @ rot).detach().cpu().numpy()[:, :, None]
    # # # ipdb.set_trace()


    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3] @ rot
    local_positions = link_to_root_pos.detach().cpu().numpy()

    model.set_global_matrix(data_dict)
    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3] @ rot
    positions = link_to_root_pos.detach().cpu().numpy()

    
    
    

    ### data normalization from humanml3d motion_representation.ipynb
    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    local_floor_height = local_positions.min(axis=0).min(axis=0)[1]
    local_positions[:, :, 1] -= local_floor_height


    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    local_root_pos_init = local_positions[0]
    local_root_pose_init_xz = local_root_pos_init[0] * np.array([1, 0, 1])
    local_positions = local_positions - local_root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = [G1_LINKS.index(link) for link in ["right_hip_pitch_link", "left_hip_pitch_link", "right_shoulder_pitch_link", "left_shoulder_pitch_link"]]
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    ### local_
    local_across1 = local_root_pos_init[r_hip] - local_root_pos_init[l_hip]
    local_across2 = local_root_pos_init[sdr_r] - local_root_pos_init[sdr_l]
    local_across = local_across1 + local_across2
    local_across = local_across / np.sqrt((local_across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    local_forward_init = np.cross(np.array([[0, 1, 0]]), local_across, axis=-1)
    # forward (3,)
    local_forward_init = local_forward_init / np.sqrt((local_forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    local_target = np.array([[0, 0, 1]])
    local_root_quat_init = qbetween_np(local_forward_init, local_target)
    local_root_quat_init = np.ones(local_positions.shape[:-1] + (4,)) * local_root_quat_init


    local_positions = qrot_np(local_root_quat_init, local_positions)







    '''New ground truth positions'''

    fid_l, fid_r =  [G1_LINKS.index(link) for link in ["left_ankle_pitch_link", "left_ankle_roll_link"]], [G1_LINKS.index(link) for link in ["right_ankle_pitch_link", "right_ankle_roll_link"]]
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, thres=0.02)

    


    '''Root height'''
    root_y = data_dict["global_translation"][:, 1:2, 0]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    # todo : discrepancy here

    r_rot =  vec6d_to_quat(torch.from_numpy(data_dict["global_rotation"])).numpy() 
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    velocity = data_dict["global_translation"][1:, :, 0] - data_dict["global_translation"][:-1, :, 0]
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)


    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, dof) dof for skeleton joints
    rot_data = data_dict["angles"]

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (link-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, (link-1)*3)
    local_vel = local_positions[1:] - local_positions[:-1]
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data

def vec_to_data_pkl(vec):
    """
    Convert the vec representation to data_dict
    :param vec: vec, the vec representation
    :return: data_dict, the data_dict
    """
    ### TODO
    data_dict = {}
    return data_dict


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Call the function with the Pickle file')
        quit()
    
    filename = sys.argv[1]
    with open(filename, "rb") as file:
        data_dict = pickle.load(file)

    vec = data_pkl_to_vec(data_dict)
    




