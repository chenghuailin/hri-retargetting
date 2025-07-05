# 2025.04.28 HIT-xiaowangzi
# 可视化hand_retargeting的参数，单独显示手的动作

import sys
import os
from HRI_retarget import DATA_ROOT, ROOT

import pickle
from HRI_retarget.utils.vis.bvh_vis import Get_bvh_joint_local_coord_parallel, calc_relative_transform
from HRI_retarget.utils.vis.bvh_vis import Rx, Ry, Rz
from HRI_retarget.config.joint_mapping import BBDB_LINKS, BBDB_G1_INSPIREHANDS_CORRESPONDENCE, \
    BBDB_LEFT_HAND_LINK, BBDB_RIGHT_HAND_LINK
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch

left_hand_to_inspire = np.array([[1,0,0],[0,0,1],[0,-1,0]])
right_hand_to_inspire = np.array([[-1,0,0],[0,0,1],[0,1,0]])

config_file_path = os.path.join(DATA_ROOT, "resources/robots/g1_inspirehands/inspire_hand.yml")
RetargetingConfig.set_default_urdf_dir(os.path.join(DATA_ROOT,"resources/robots/g1_inspirehands"))
with Path(config_file_path).open('r') as f:
    cfg = yaml.safe_load(f)
left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
left_retargeting = left_retargeting_config.build()
right_retargeting = right_retargeting_config.build()

print("left joint names: ", left_retargeting.joint_names)
print("right_joint_names: ", right_retargeting.joint_names)

def fingerpos_clip(qpos):
    limits = np.array([
        [-0.1, 1.3], ## L_thumb_proximal_yaw_joint
        [-0.1, 0.6], ##L_thumb_proximal_pitch_joint
        [0, 0.8], ##L_thumb_intermediate_joint
        [0, 1.2], ##L_thumb_distal_joint
        [0, 1.7], ##L_index_proximal_joint
        [0, 1.7], ##L_index_intermediate_joint
        [0, 1.7], ##L_middle_proximal_joint
        [0, 1.7], ##L_middle_intermediate_joint
        [0, 1.7], ##L_ring_proximal_joint
        [0, 1.7], ##L_ring_intermediate_joint
        [0, 1.7], ##L_pinky_proximal_joint
        [0, 1.7], ##L_pinky_intermediate_joint
    ])
    qpos[qpos < limits[:,0]] = limits[:,0][qpos < limits[:,0]]
    qpos[qpos > limits[:,1]] = limits[:,1][qpos > limits[:,1]]
    return qpos


if __name__ == "__main__":
    
    # 1. extract the transformations from the bvh file
    filename = os.path.join(DATA_ROOT, "motion/human/misc/suisei_vivideba_motion_.bvh")
    bvh_joint_local_coord, bvh_joint_local_rot = Get_bvh_joint_local_coord_parallel(filename, link_list = BBDB_LINKS)
    num_frames = len(bvh_joint_local_coord)
    print("Num of frames: ", num_frames)
    
    # 2. get the vector from the transformations
    left_rel_pos = torch.zeros([num_frames,5,3])
    right_rel_pos = torch.zeros([num_frames, 5,3])
    left_root_id = BBDB_LEFT_HAND_LINK["base_link"]
    left_tip_id_list = BBDB_LEFT_HAND_LINK["tip_link"]
    right_root_id = BBDB_RIGHT_HAND_LINK["base_link"]
    right_tip_id_list = BBDB_RIGHT_HAND_LINK["tip_link"]
    left_rot_batch = torch.from_numpy(left_hand_to_inspire).view(1,3,3).repeat(num_frames,1,1).type(torch.float)
    right_rot_batch = torch.from_numpy(right_hand_to_inspire).view(1,3,3).repeat(num_frames,1,1).type(torch.float)
    
    for i,tip_id in enumerate(left_tip_id_list):
        pos, rot = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, left_root_id, tip_id)
        pos[:,0,:] = -pos[:,0,:]
        print("left_original_pos: ", pos[0])
        # pos[:,0] = pos[:,0] # add negative because the hand_link in bvh and in inspire_hands are different
        # pos[:,1] = pos[:,1]
        left_rel_pos[:,i,:] = torch.bmm(left_rot_batch,pos).squeeze(2)
        # left_rel_pos [:,i,:] = pos.squeeze(2)
    print("left_final_pos: ", left_rel_pos[0])
    for i,tip_id in enumerate(right_tip_id_list):
        pos, rot = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, right_root_id, tip_id)
        # pos[:,0] = pos[:,0]
        # pos[:,1] = pos[:,1]
        pos[:,0,:] = -pos[:,0,:]
        right_rel_pos[:,i,:] = torch.bmm(right_rot_batch,pos).squeeze(2)
        # right_rel_pos[:,i,:] = pos.squeeze(2)
    # rot_left = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    # rot_right = np.array([[-1,0,0],[0,0,-1],[0,-1,0]])
    # left_path = os.path.join(ROOT,"log","left_finger_pos_tf.csv")
    # right_path = os.path.join(ROOT,"log","right_finger_pos_tf.csv")
    # left_data_df = pd.read_csv(left_path)
    # right_data_df = pd.read_csv(right_path)
    # left_data = left_data_df.values[:,1:]
    # right_data = right_data_df.values[:,1:]
    # print("left_data: ", left_data.shape)
    # print("right_data: ", right_data.shape)
    # print("left_data first row: ", left_data[0])
    # print("right_data first row: ", right_data[0])
    
    
    left_qpos_list = np.zeros((num_frames,12))
    right_qpos_list = np.zeros((num_frames,12))
    left_last_qpos = np.zeros((12,1))
    right_last_qpos = np.zeros((12,1))
    for i in range(num_frames):
        left_ref = left_rel_pos[i,:,:].numpy()
        right_ref = right_rel_pos[i,:,:].numpy()
        # left_tippos = left_data[i,:].reshape(5,3)
        # right_tippos = right_data[i,:].reshape(5,3)
        # left_ref = np.zeros((5,3))
        # right_ref = np.zeros((5,3))
        # for j in range(5):
        #     left_temp = rot_left @ (left_tippos[j,:].reshape(3,1)) 
        #     left_ref[j,:] = left_temp.reshape(3,)
        #     right_temp = rot_right @ (right_tippos[j,:].reshape(3,1))
        #     right_ref[j,:] = right_temp.reshape(3,)  
                           
        left_qpos = left_retargeting.retarget(left_ref)
        right_qpos = right_retargeting.retarget(right_ref)
        # left_qpos_list[i,:] = fingerpos_clip(left_qpos[[8,9,10,11,0,1,2,3,6,7,4,5]])
        # right_qpos_list[i,:] = fingerpos_clip(right_qpos[[8,9,10,11,0,1,2,3,6,7,4,5]])
        left_qpos_list[i,:] = fingerpos_clip(left_qpos[[9,10,11,12,1,2,3,4,7,8,5,6]])
        right_qpos_list[i,:] = fingerpos_clip(right_qpos[[9,10,11,12,1,2,3,4,7,8,5,6]])
        
    data_dict = {
        "left_ref": left_ref,
        "right_ref": right_ref,
        "left_qpos": left_qpos_list,
        "right_qpos": right_qpos_list
    }
    
    with open(os.path.join(DATA_ROOT,"motion/g1/BBDB", "hand_vis.pickle"), "wb") as file:
        pickle.dump(data_dict, file)
        
    left_df = pd.DataFrame(left_qpos_list)
    right_df = pd.DataFrame(right_qpos_list)
    left_ref_df = pd.DataFrame(left_rel_pos.reshape(-1,15))
    right_ref_df = pd.DataFrame(right_rel_pos.reshape(-1,15))
    # left_df.columns = ['L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint',
    #                    'L_thumb_intermediate_joint', 'L_thumb_distal_joint',
    #                    'L_index_proximal_joint', 'L_index_intermediate_joint',
    #                    'L_middle_proximal_joint', 'L_middle_intermediate_joint',
    #                    'L_ring_proximal_joint', 'L_ring_intermediate_joint',
    #                    'L_pinky_proximal_joint', 'L_pinky_intermediate_joint',
    #                    ]
    # right_df.columns = ['R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint',
    #                     'R_thumb_intermediate_joint', 'R_thumb_distal_joint',
    #                     'R_index_proximal_joint', 'R_index_intermediate_joint',
    #                     'R_middle_proximal_joint', 'R_middle_intermediate_joint',
    #                     'R_ring_proximal_joint', 'R_ring_intermediate_joint',
    #                     'R_pinky_proximal_joint', 'R_pinky_intermediate_joint',
    #                     ]
    # left_ref_df.columns = ["thumb_x","thumb_y","thumb_z",
    #                        "index_x","index_y","index_z",
    #                        "middle_x","middle_y","middle_z",
    #                        "ring_x","ring_y","ring_z",
    #                        "pinky_x","pinky_y","pinky_z"]
    # right_ref_df.columns = ["thumb_x","thumb_y","thumb_z",
    #                         "index_x","index_y","index_z",
    #                         "middle_x","middle_y","middle_z",
    #                         "ring_x","ring_y","ring_z",
    #                         "pinky_x","pinky_y","pinky_z"]
    # left_df.to_excel(os.path.join(ROOT,"log","left_finger_joints.xlsx"))
    # right_df.to_excel(os.path.join(ROOT,"log","right_finger_joints.xlsx"))
    # left_ref_df.to_excel(os.path.join(ROOT,"log","left_fingertip_pos.xlsx"))
    # right_ref_df.to_excel(os.path.join(ROOT,"log","right_fingertip_pos.xlsx"))
    left_df.to_csv(os.path.join(ROOT,"log","left_finger_joints.csv"))
    right_df.to_csv(os.path.join(ROOT,"log","right_finger_joints.csv"))
    left_ref_df.to_csv(os.path.join(ROOT,"log","left_fingertip_pos.csv"))
    right_ref_df.to_csv(os.path.join(ROOT,"log","right_fingertip_pos.csv"))
        
    
