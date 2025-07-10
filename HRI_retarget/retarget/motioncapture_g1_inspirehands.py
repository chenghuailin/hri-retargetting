### usage:
### python motioncapture_g1_inspirehands.py {path_to_npy_file}
### python HRI_retarget/retarget/motioncapture_g1_inspirehands.py data/motion/human/motion_capture/defense_Skeleton.bvh
# 2025.05.03
# retarget body motion + hand motion

import sys
import os
from HRI_retarget import DATA_ROOT


import torch
from tqdm import tqdm
import pickle

import numpy as np

from HRI_retarget import ROOT
from HRI_retarget.utils.vis.bvh_vis import Get_bvh_joint_global_pos, Get_bvh_joint_pos_and_Rot, calc_relative_transform
from HRI_retarget.utils.vis.kinematic_vis import vis_kinematic_result
from HRI_retarget.model.g1_inspirehands import G1_Inspirehands_Motion_Model
from HRI_retarget.config.joint_mapping import MOTION_CAPTURE_LINKS, MOTION_CAPTURE_G1_INSPIREHANDS_CORRESPONDENCE, \
    MOTION_CAPTURE_LEFT_HAND_LINK, MOTION_CAPTURE_RIGHT_HAND_LINK
from HRI_retarget.utils.vis.bvh_vis import Rx, Ry, Rz

import matplotlib.pyplot as plt
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
import pandas as pd

# rot = np.eye(3)
rot = np.array([[0,0,1],
                [1,0,0],
                [0,1,0]])
left_hand_to_inspire = np.array([[1,0,0],[0,0,1],[0,-1,0]])
right_hand_to_inspire = np.array([[-1,0,0],[0,0,1],[0,1,0]])
config_file_path = os.path.join(DATA_ROOT, "resources/robots/g1_inspirehands/inspire_hand.yml")
default_urdf_dir = os.path.join(DATA_ROOT, "resources/robots/g1_inspirehands")

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('Call the function with the BVH file and the corresponding suffix')
        quit()

    filename = sys.argv[1]
    # bvh_joint_local_coord_pos, bvh_joint_local_coord_rot = Get_bvh_joint_pos_and_Rot(filename, link_list = MOTION_CAPTURE_LINKS)
    bvh_joint_local_coord_pos, _, bvh_root_pos, bvh_root_quat = Get_bvh_joint_pos_and_Rot(filename, link_list = MOTION_CAPTURE_LINKS)
    # bvh_joint_global_coord_pos = Get_bvh_joint_global_pos(filename, link_list = MOTION_CAPTURE_LINKS)


    num_frames = len(bvh_joint_local_coord_pos)
    # num_frames = len(bvh_joint_global_coord_pos)

    print("Num of frames: ", num_frames)

    rot_batch = torch.from_numpy(rot).view(1,3,3).repeat(num_frames,1,1).type(torch.float)
    bvh_root_pos = torch.from_numpy(bvh_root_pos).view( num_frames, 1, 3)
    bvh_root_pos = torch.bmm( bvh_root_pos,  rot_batch.transpose(1,2)) 
    
    model = G1_Inspirehands_Motion_Model(batch_size=num_frames, global_trans=bvh_root_pos, global_rot=bvh_root_quat, joint_correspondence=MOTION_CAPTURE_G1_INSPIREHANDS_CORRESPONDENCE)

    model.set_gt_joint_positions(torch.bmm(bvh_joint_local_coord_pos, rot_batch.transpose(1,2)))
    # model.set_gt_joint_positions(torch.bmm(bvh_joint_global_coord_pos,rot_batch.transpose(1,2)))
    # model.set_gt_joint_positions(bvh_joint_local_coord @ rot.T)

    print("Links of robot: ", model.chain.get_link_names())
    # print(model.global_trans)
    # import ipdb;ipdb.set_trace()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    model.train()

    history_losses = []
    # best_loss = float('inf')
    # best_model_state = None
    
    pbar = tqdm(range(3000))
    for epoch in pbar:
        
        ### normalize
        with torch.no_grad():
            # model.refine_wrist_angle()reshape(3, 1)
            model.normalize()
    
        joint_local_velocity_loss, joint_local_accel_loss = model.joint_local_velocity_loss()
        joint_global_position_loss = model.retarget_joint_loss()
        dof_limit_loss = model.dof_limit_loss()
        collision_loss = model.collision_loss()
        constraint_loss = model.constraint_loss_Jappelio_rays()
        
        # orientation_loss = model.orientation_loss()
        # init_angle_loss = model.init_angle_loss()
        # elbow_loss = model.elbow_loss()
        
        loss_dict = {
            "joint_global_position_loss": [2.0, joint_global_position_loss],
            "joint_local_velocity_loss": [1.0, joint_local_velocity_loss],
            "joint_local_accel_loss": [0.5, joint_local_accel_loss],
            "dof_limit_loss": [1.0, dof_limit_loss],
            # "orientation_loss": [0.3, orientation_loss],
            "collision_loss": [1.5, collision_loss],
            "constraint_loss": [1.5, constraint_loss],
        }

        loss = 0
        log_str = "#" * 50 + "\n"
        for loss_name in loss_dict.keys():
            loss += loss_dict[loss_name][0] * loss_dict[loss_name][1]
            log_str += f"{loss_name}: {loss_dict[loss_name][0] * loss_dict[loss_name][1].item()}" + "\n"
            print(f"{loss_name}: {loss_dict[loss_name][0] * loss_dict[loss_name][1].item()}")
        # pbar.set_description(log_str) 
        # print("dof_limit_loss", dof_limit_loss.item())
        # print("collision_loss", collision_loss.item())
        # print("hand_orientation_loss: ", hand_orientation_loss.item())

        pbar.set_description(f"loss: {loss.item()}")
        history_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #     if loss.item() < best_loss:
    #         best_loss = loss.item()
    #         best_model_state = copy.deepcopy(model.state_dict())
        

    # # After training, load the best state
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    #     print(f"Best loss: {best_loss}")

  
    with torch.no_grad():
        pred_joint_angles = model.joint_angles.detach().cpu().numpy()
        # global_rotation = model.global_rot.detach().cpu().numpy()
        global_translation = model.global_trans.detach().cpu().numpy()
        scale = model.scale.detach().cpu().numpy()
        

    data_dict = {
        "fps": 120,
        "reference_motion_pth": filename,
        "robot_name": "g1_inspirehands",
        "angles": pred_joint_angles,
        "global_rotation": bvh_root_quat,
        "global_translation": bvh_root_pos,
        "scale": scale,
    }

    # import ipdb;ipdb.set_trace()
    
    path_g1_motion_capture = os.path.join(DATA_ROOT, "motion/g1/motion_capture")
    Path(path_g1_motion_capture).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(path_g1_motion_capture, filename.split("/")[-1][:-4] + sys.argv[2] + ".pickle"), "wb") as file:
        pickle.dump(data_dict, file)
    

    ### visualize results.

    ## draw loss curve
    plt.plot(history_losses[len(history_losses) // 10:], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ### vis motion
    # ### press esc to quit plt visualization

    # vis_kinematic_result(os.path.join(DATA_ROOT,"motion/g1/motion_capture", filename.split("/")[-1][:-4] + ".pickle"), dataset="motion_capture", robot="g1_inspirehands", correspondence=MOTION_CAPTURE_G1_INSPIREHANDS_CORRESPONDENCE)
