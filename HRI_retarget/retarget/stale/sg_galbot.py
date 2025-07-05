### usage:
### python sg_galbot.py {path_to_bvh_file}
### python sg_galbot.py data/human/SG/output.bvh
### todo: 
###   tune magic numbers

import sys
import os
from HRI_retarget import DATA_ROOT


import torch
from tqdm import tqdm
import pickle

from HRI_retarget.utils.vis.bvh_vis import Get_bvh_joint_local_coord
from HRI_retarget.utils.vis.kinematic_vis import vis_kinematic_result
from HRI_retarget.model.galbot_charlie import Galbot_Charlie_Motion_Model
from HRI_retarget.config.joint_mapping import SG_LINKS, SG_GALBOT_CHARLIE_CORRESPONDENCE


### magic numbers
### transition from sg to galbot
rot = torch.tensor([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=torch.float)
# # pos = torch.tensor([0, 0, 0.8], dtype=torch.float)
# scale = 1.5

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Call the function with the BVH file')
        quit()

    filename = sys.argv[1]
    bvh_joint_local_coord = Get_bvh_joint_local_coord(filename, link_list=SG_LINKS)

   
    num_frames = len(bvh_joint_local_coord)
    print("Num of frames: ", num_frames)
    
    model = Galbot_Charlie_Motion_Model(batch_size=num_frames, joint_correspondence=SG_GALBOT_CHARLIE_CORRESPONDENCE)


    


    print(bvh_joint_local_coord.shape)
    # model.set_gt_joint_positions((bvh_joint_local_coord) @ rot.T @ scale + pos)
    model.set_gt_joint_positions(bvh_joint_local_coord @ rot.T)
    # print(model.chain.get_link_names())

    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    model.train()

    pbar = tqdm(range(2000))
    for epoch in pbar:
        
        joint_local_velocity_loss, joint_local_accel_loss = model.joint_local_velocity_loss()
        joint_global_position_loss = model.retarget_joint_loss()
        # init_angle_loss = model.init_angle_loss()
        # elbow_loss = model.elbow_loss()

        loss = 1.0 * joint_global_position_loss + 1.0 * joint_local_velocity_loss + 0.0 * joint_local_accel_loss #+ 0.0 * init_angle_loss + 0.0 * elbow_loss
        pbar.set_description(f"{loss.item()}, {joint_global_position_loss.item()}, {joint_local_velocity_loss.item()}")


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred_joint_angles = model.joint_angles.detach().cpu().numpy()
        global_rotation = model.global_rot.detach().cpu().numpy()
        global_translation = model.global_trans.detach().cpu().numpy()
        scale = model.scale.detach().cpu().numpy()

    data_dict = {
        "angles": pred_joint_angles,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
        "scale": scale,
    }

    with open(os.path.join(DATA_ROOT,"motion/galbot/SG", filename.split("/")[-1][:-4] + ".pickle"), "wb") as file:
        pickle.dump(data_dict, file)
    
    vis_kinematic_result(os.path.join(DATA_ROOT,"motion/galbot/SG", filename.split("/")[-1][:-4] + ".pickle"), dataset="SG", robot="galbot", correspondence=SG_GALBOT_CHARLIE_CORRESPONDENCE)
