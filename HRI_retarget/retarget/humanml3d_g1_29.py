### usage:
### python humanml3d_g1_29.py {path_to_npy_file}
### python HRI_retarget/retarget/humanml3d_g1_29.py data/motion/human/HumanML3D/new_joints/000000.npy

import sys
import os
from HRI_retarget import DATA_ROOT


import torch
from tqdm import tqdm
import pickle

import numpy as np

from HRI_retarget.utils.vis.kinematic_vis import vis_kinematic_result
from HRI_retarget.model.g1_29 import G1_29_Motion_Model
from HRI_retarget.config.joint_mapping import SMPL_G1_FULLBODY_CORRESPONDENCE

import matplotlib.pyplot as plt

### magic numbers
### transition from sg to galbot
rot = torch.tensor([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=torch.float)



if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Call the function with the BVH file')
        quit()

    filename = sys.argv[1]
    bvh_joint_local_coord = torch.from_numpy(np.load(filename))


   
    num_frames = len(bvh_joint_local_coord)
    print("Num of frames: ", num_frames)
    
    model = G1_29_Motion_Model(batch_size=num_frames, joint_correspondence=SMPL_G1_FULLBODY_CORRESPONDENCE)



    print(bvh_joint_local_coord.shape)

    model.set_gt_joint_positions(bvh_joint_local_coord @ rot.T)
    print("Links of robot: ", model.chain.get_link_names())

    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    model.train()

    history_losses = []
    
    pbar = tqdm(range(2000))
    for epoch in pbar:
        
        ### normalize
        with torch.no_grad():
            model.normalize()
    
        joint_local_velocity_loss, joint_local_accel_loss = model.joint_local_velocity_loss()
        joint_global_position_loss = model.retarget_joint_loss()
        dof_limit_loss = model.dof_limit_loss()
        # collision_loss = model.collision_loss()
        # init_angle_loss = model.init_angle_loss()
        # elbow_loss = model.elbow_loss()


        loss_dict = {
            "joint_global_position_loss": [1.0, joint_global_position_loss],
            "joint_local_velocity_loss": [1.0, joint_local_velocity_loss],
            "joint_local_accel_loss": [0.0, joint_local_accel_loss],
            "dof_limit_loss": [1.0, dof_limit_loss],
            # "collision_loss": [1.0, collision_loss],
        }

        loss = 0
        log_str = "#" * 50 + "\n"
        for loss_name in loss_dict.keys():
            loss += loss_dict[loss_name][0] * loss_dict[loss_name][1]
            log_str += f"{loss_name}: {loss_dict[loss_name][0] * loss_dict[loss_name][1].item()}" + "\n"
        # pbar.set_description(log_str)  
        # print("dof_limit_loss", dof_limit_loss.item())
        # print("collision_loss", collision_loss.item())

        pbar.set_description(f"loss:, {loss.item()}")
        history_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

    with torch.no_grad():
        
        pred_joint_angles = model.joint_angles.detach().cpu().numpy()
        global_rotation = model.global_rot.detach().cpu().numpy()
        global_translation = model.global_trans.detach().cpu().numpy()
        scale = model.scale.detach().cpu().numpy()

    data_dict = {
        "fps": 20,
        "reference_motion_pth": filename,
        "robot_name": "g1_29",
        "angles": pred_joint_angles,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
        "scale": scale,
    }

    with open(os.path.join(DATA_ROOT,"motion/g1/HumanML3D", filename.split("/")[-1][:-4] + ".pickle"), "wb") as file:
        pickle.dump(data_dict, file)

    

    ### visualize results.

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

    vis_kinematic_result(os.path.join(DATA_ROOT,"motion/g1/HumanML3D", filename.split("/")[-1][:-4] + ".pickle"), dataset="HumanML3D", robot="g1_29", correspondence=SMPL_G1_FULLBODY_CORRESPONDENCE)
