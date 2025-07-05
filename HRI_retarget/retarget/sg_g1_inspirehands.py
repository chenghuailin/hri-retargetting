### usage:
### python beat_g1_inspirehands.py {path_to_npy_file}
### python src/retarget/hand_retarget.py data/motion/human/BEAT_ZIP/beat_english_v0.2.1/1/1_wayne_0_1_1.bvh
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
from HRI_retarget.utils.vis.bvh_vis import Get_bvh_joint_pos_and_Rot, calc_relative_transform, Get_bvh_joint_angles
from HRI_retarget.utils.vis.kinematic_vis import vis_kinematic_result
from HRI_retarget.model.g1_inspirehands import G1_Inspirehands_Motion_Model
from HRI_retarget.config.joint_mapping import SG_LINKS, SG_G1_INSPIREHANDS_CORRESPONDENCE, \
    SG_LEFT_HAND_LINK, SG_RIGHT_HAND_LINK
from HRI_retarget.utils.vis.bvh_vis import Rx, Ry, Rz

import matplotlib.pyplot as plt
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
import pandas as pd

# rot = torch.eye(3)
rot = np.array([[0,0,1],[1,0,0],[0,1,0]])
left_hand_to_inspire = np.array([[0,1,0],[-1,0,0],[0,0,1]])
right_hand_to_inspire = np.array([[0,1,0],[1,0,0],[0,0,-1]])
config_file_path = os.path.join(DATA_ROOT, "resources/robots/g1_inspirehands/inspire_hand.yml")
default_urdf_dir = os.path.join(DATA_ROOT,"resources/robots/g1_inspirehands")

RIGHT_HAND_JOINTS = {
   "index":[SG_LINKS.index('RightInHandIndex'),
            SG_LINKS.index('RightHandIndex1'),
            SG_LINKS.index('RightHandIndex2'), 
            SG_LINKS.index('RightHandIndex3')],
   "middle":[SG_LINKS.index('RightInHandMiddle'), 
             SG_LINKS.index('RightHandMiddle1'),
             SG_LINKS.index('RightHandMiddle2'),
             SG_LINKS.index('RightHandMiddle3')],
   "pinky":[SG_LINKS.index('RightInHandPinky'),
            SG_LINKS.index('RightHandPinky1'),
            SG_LINKS.index('RightHandPinky2'),
            SG_LINKS.index('RightHandPinky3')],
   "ring":[SG_LINKS.index('RightInHandRing'),
            SG_LINKS.index('RightHandRing1'),
            SG_LINKS.index('RightHandRing2'),
            SG_LINKS.index('RightHandRing3')],
   "thumb":[SG_LINKS.index('RightHandThumb1'), 
            SG_LINKS.index('RightHandThumb2'), 
            SG_LINKS.index('RightHandThumb3')]
}

LEFT_HAND_JOINTS = {
   "index":[SG_LINKS.index('LeftInHandIndex'),
            SG_LINKS.index('LeftHandIndex1'),
            SG_LINKS.index('LeftHandIndex2'), 
            SG_LINKS.index('LeftHandIndex3')],
   "middle":[SG_LINKS.index('LeftInHandMiddle'), 
             SG_LINKS.index('LeftHandMiddle1'),
             SG_LINKS.index('LeftHandMiddle2'),
             SG_LINKS.index('LeftHandMiddle3')],
   "pinky":[SG_LINKS.index('LeftInHandPinky'),
            SG_LINKS.index('LeftHandPinky1'),
            SG_LINKS.index('LeftHandPinky2'),
            SG_LINKS.index('LeftHandPinky3')],
   "ring":[SG_LINKS.index('LeftInHandRing'),
            SG_LINKS.index('LeftHandRing1'),
            SG_LINKS.index('LeftHandRing2'),
            SG_LINKS.index('LeftHandRing3')],
   "thumb":[SG_LINKS.index('LeftHandThumb1'), 
            SG_LINKS.index('LeftHandThumb2'), 
            SG_LINKS.index('LeftHandThumb3')]
}

if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #     print('Call the function with the BVH file')
    #     quit()

    # filename = sys.argv[1]
    filename = os.path.join(DATA_ROOT, "motion/human/SG/volcengine_394813b9b5eb7174d442b63a7b809463_original_motion.bvh")
    bvh_joint_local_coord, bvh_joint_local_rot = Get_bvh_joint_pos_and_Rot(filename, link_list = SG_LINKS)
    
    # bvh_joint_local_coord = bvh_joint_local_coord[0:100,...]
    # bvh_joint_local_rot = bvh_joint_local_rot[0:100,...]
    


    num_frames = len(bvh_joint_local_coord)
    print("Num of frames: ", num_frames)
    
    joint_names, joint_angles = Get_bvh_joint_angles(filename, link_list = SG_LINKS)
    # joint_angles = joint_angles[0:100,...]
    # print(joint_angles[-10:-1,RIGHT_HAND_JOINTS["thumb"],:])
    left_hand_joints = {}
    
    for key, value in LEFT_HAND_JOINTS.items():
        left_hand_joints[key] = torch.norm(joint_angles[:,value,:],dim =2) * torch.pi / 180.0
    right_hand_joints = {}
    for key, value in RIGHT_HAND_JOINTS.items():
        right_hand_joints[key] = torch.norm(joint_angles[:,value,:],dim =2) * torch.pi / 180.0
    
    model = G1_Inspirehands_Motion_Model(batch_size=num_frames, joint_correspondence=SG_G1_INSPIREHANDS_CORRESPONDENCE)
    model.copy_qpos_from_bvh(left_hand_joints, right_hand_joints)

    rot_batch = torch.from_numpy(rot).view(1,3,3).repeat(num_frames,1,1).type(torch.float)
    model.set_gt_joint_positions(torch.bmm(bvh_joint_local_coord,rot_batch.transpose(1,2)))
    # model.set_gt_joint_positions(bvh_joint_local_coord @ rot.T)
    print("Links of robot: ", model.chain.get_link_names())
    
    # Set the fingertip pos 
    left_rel_pos = torch.zeros([num_frames,5,3])
    right_rel_pos = torch.zeros([num_frames, 5,3])
    left_root_id = SG_LEFT_HAND_LINK["base_link"]
    left_tip_id_list = SG_LEFT_HAND_LINK["tip_link"]
    right_root_id = SG_RIGHT_HAND_LINK["base_link"]
    right_tip_id_list = SG_RIGHT_HAND_LINK["tip_link"]
    left_rot_batch = torch.from_numpy(left_hand_to_inspire).view(1,3,3).repeat(num_frames,1,1).type(torch.float)
    right_rot_batch = torch.from_numpy(right_hand_to_inspire).view(1,3,3).repeat(num_frames,1,1).type(torch.float)
        
    for i,tip_id in enumerate(left_tip_id_list):
        pos, rot = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, left_root_id, tip_id)
        left_rel_pos[:,i,:] = torch.bmm(left_rot_batch,pos).squeeze(2)
    for i,tip_id in enumerate(right_tip_id_list):
        pos, rot = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, right_root_id, tip_id)
        right_rel_pos[:,i,:] = torch.bmm(right_rot_batch,pos).squeeze(2)
        
        
    model.set_hand_optimizer(config_file_path=config_file_path, default_urdf_dir=default_urdf_dir)
    model.set_hand_tip_positions(left_rel_pos,right_rel_pos)
    
    # Set the hand rotations
    left_forearm = SG_LINKS.index("LeftForeArm")
    left_hand = SG_LINKS.index("LeftHand")
    right_forearm = SG_LINKS.index("RightForeArm")
    right_hand = SG_LINKS.index("RightHand")
    _ , left_hand_rotations = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot,left_forearm, left_hand)
    _ , right_hand_rotations = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot,right_forearm, right_hand)

    model.set_hand_rotations(left_hand_rotations, right_hand_rotations)
    
    # Calculate the desired hand orientations from inspire hand to pelvis
    hip = SG_LINKS.index("Hips")
    left_wrist = SG_LINKS.index("LeftHand")
    right_wrist = SG_LINKS.index("RightHand")
    _,left_hand_to_hip = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, hip, left_wrist)
    _,right_hand_to_hip = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, hip, right_wrist)
    left_inspire_to_pelvis = torch.bmm(torch.bmm(rot_batch,left_hand_to_hip),left_rot_batch.transpose(1,2))
    right_inspire_to_pelvis = torch.bmm(torch.bmm(rot_batch,right_hand_to_hip),right_rot_batch.transpose(1,2))
    model.set_hand_rotations_world(left_inspire_to_pelvis, right_inspire_to_pelvis)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    model.train()

    history_losses = []
    
    pbar = tqdm(range(100))
    for epoch in pbar:
        
        ### normalize
        with torch.no_grad():
            # model.refine_wrist_angle()
            model.normalize()
    
        joint_local_velocity_loss, joint_local_accel_loss = model.joint_local_velocity_loss()
        joint_global_position_loss = model.retarget_joint_loss()
        dof_limit_loss = model.dof_limit_loss()
        hand_orientation_loss = model.hand_orientation_loss()
        collision_loss = model.collision_loss()
        # init_angle_loss = model.init_angle_loss()
        # elbow_loss = model.elbow_loss()
        


        loss_dict = {
            "joint_global_position_loss": [1.0, joint_global_position_loss],
            "joint_local_velocity_loss": [1.0, joint_local_velocity_loss],
            "joint_local_accel_loss": [0.3, joint_local_accel_loss],
            "dof_limit_loss": [1.0, dof_limit_loss],
            "hand_orientation_loss": [0.3, hand_orientation_loss],
            "collision_loss": [0.1, collision_loss],
        }

        loss = 0
        log_str = "#" * 50 + "\n"
        for loss_name in loss_dict.keys():
            loss += loss_dict[loss_name][0] * loss_dict[loss_name][1]
            log_str += f"{loss_name}: {loss_dict[loss_name][0] * loss_dict[loss_name][1].item()}" + "\n"
        # pbar.set_description(log_str)  
        # print("dof_limit_loss", dof_limit_loss.item())
        print("collision_loss", collision_loss.item())
        # print("hand_orientation_loss: ", hand_orientation_loss.item())

        pbar.set_description(f"loss:, {loss.item()}")
        history_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 最后统一计算灵巧手的角度  
    
    # model.refine_hand_angle() 
    
    left_tip_pos, _ = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, left_root_id, left_tip_id_list[0])
    right_tip_pos, _ = calc_relative_transform(bvh_joint_local_coord, bvh_joint_local_rot, right_root_id, right_tip_id_list[0])
    left_thumb_tip = left_tip_pos.squeeze()
    left_thumb_jnt1 = torch.acos(left_thumb_tip[:,2]/torch.norm(left_thumb_tip[:,1:],dim = 1))
    right_thumb_tip = right_tip_pos.squeeze()
    right_thumb_jnt1 = torch.acos(right_thumb_tip[:,2]/torch.norm(right_thumb_tip[:,1:],dim = 1))
    
    
    with torch.no_grad():
        pred_joint_angles = model.joint_angles.detach().cpu().numpy()
        # pred_joint_angles[:,22:34] = model.left_qpos_list
        # pred_joint_angles[:,41:53] = model.right_qpos_list
        pred_joint_angles[:,22:34] = model.left_hand_qpos
        pred_joint_angles[:,41:53] = model.right_hand_qpos
        # refine thumb
        pred_joint_angles[:,30] = left_thumb_jnt1.numpy()
        pred_joint_angles[:,49] = right_thumb_jnt1.numpy()
        # pred_joint_angles[:,30:34] = model.left_qpos_list[:,-4:]
        # pred_joint_angles[:,49:53] = model.right_qpos_list[:,-4:]
        global_rotation = model.global_rot.detach().cpu().numpy()
        global_translation = model.global_trans.detach().cpu().numpy()
        scale = model.scale.detach().cpu().numpy()
        

    data_dict = {
        "fps": 60,
        "reference_motion_pth": filename,
        "robot_name": "g1_inspirehands",
        "angles": pred_joint_angles,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
        "scale": scale,
    }
    

    with open(os.path.join(DATA_ROOT,"motion/g1/SG_with_hand", filename.split("/")[-1][:-4] + ".pickle"), "wb") as file:
        pickle.dump(data_dict, file)
        
    joints_df = pd.DataFrame(pred_joint_angles)
    joints_df.to_csv(os.path.join(ROOT,"..","log","sg_g1_inspirehands_joints.csv"))
    
    left_hand_df = pd.DataFrame(left_rel_pos.reshape(-1,15))
    right_hand_df = pd.DataFrame(right_rel_pos.reshape(-1,15))
    left_hand_df.to_csv(os.path.join(ROOT,"..","log","sg_left_tip.csv"))
    right_hand_df.to_csv(os.path.join(ROOT,"..","log","sg_right_tip.csv"))

    

    # ### visualize results.

    # ### draw loss curve
    # plt.plot(history_losses[len(history_losses) // 10:], label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curve')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # ### vis motion
    # ### press esc to quit plt visualization

    # vis_kinematic_result(os.path.join(DATA_ROOT,"motion/g1/BBDB", filename.split("/")[-1][:-4] + ".pickle"), dataset="BBDB", robot="g1_inspirehands", correspondence=BBDB_G1_INSPIREHANDS_CORRESPONDENCE)
