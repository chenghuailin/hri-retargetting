### usage: visualize galbot and bvh at the same time
### specific to /data/SeG_dataset results
### python kinematic_vis.py data/SeG_dataset/galbot_motion/ARMS_SELF_EMBRACE-1.pickle

### press esc to quit the plt visualization

import torch
import time
import pickle 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_kinematics as pk
from HRI_retarget import DATA_ROOT
import textwrap

from HRI_retarget.config.joint_mapping import GALBOT_CHARLIE_LINKS, G1_LINKS, G1_INSPIREHANDS_LINKS, MOTION_CAPTURE_G1_INSPIREHANDS_CORRESPONDENCE, MOTION_CAPTURE_LINKS, SG_LINKS, SEG_LINKS, SMPL_LINKS, BEAT_LINKS, SG_G1_CORRESPONDENCE, SMPL_G1_CORRESPONDENCE
from HRI_retarget.utils.vis.bvh_vis import Draw_bvh_frame, ProcessBVH, Get_bvh_joint_local_coord
from HRI_retarget.model.galbot_charlie import Galbot_Charlie_Motion_Model
from HRI_retarget.model.g1_15 import G1_15_Motion_Model
from HRI_retarget.model.g1_29 import G1_29_Motion_Model
from HRI_retarget.model.g1_inspirehands import G1_Inspirehands_Motion_Model

from pynput import keyboard


def Draw_Motiongpt_bvh(urdf_link_pos, urdf_chain, robot_link=G1_LINKS, text_prompt=""):
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    figure_limit = 2 #used to set figure axis limits

    
    text_prompt = '\n'.join(textwrap.wrap(text_prompt, width=100))

    idx = 0

    # 检测键盘输入esc时退出
    def on_press(key):
        if key == keyboard.Key.esc:
            # 停止监听
            return False
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while listener.running:
        idx += 1
        i = idx % len(urdf_link_pos) 

        # import ipdb;ipdb.set_trace()


        ### visualizeing urdf 
        frames_to_draw = [robot_link[0]]
        urdf_pos = urdf_link_pos[i]

        while frames_to_draw != []:
            link = frames_to_draw[0]
            frames_to_draw.pop(0)
            for child_frame in urdf_chain.find_frame(link).children:
                child_link = child_frame.name 
                frames_to_draw.append(child_link)
                ### by default apply rotation on robot frame
                plt.plot(xs = [-urdf_pos[robot_link.index(link)][0], -urdf_pos[robot_link.index(child_link)][0]],
                        zs = [-urdf_pos[robot_link.index(link)][1], -urdf_pos[robot_link.index(child_link)][1]],
                        ys = [urdf_pos[robot_link.index(link)][2], urdf_pos[robot_link.index(child_link)][2]],c = 'red', lw = 2.5)

        #Depending on the file, the axis limits might be too small or too big. Change accordingly.
        ax.set_axis_off()
        ax.set_xlim(-0.6*figure_limit, 0.6*figure_limit)
        ax.set_ylim(-0.6*figure_limit, 0.6*figure_limit)
        ax.set_zlim(-0.2*figure_limit, 1.*figure_limit)
        plt.title(text_prompt + '\n' + 'frame: ' + str(i))
        plt.pause(0.01)
        ax.cla()
    pass



def Draw_bvh_urdf(bvh_link_pos, bvh_skeleton_data, urdf_link_pos, urdf_chain, reference_link=SG_LINKS, robot_link=G1_LINKS, correspondence=SG_G1_CORRESPONDENCE):

    bvh_joints = bvh_skeleton_data[0]
    bvh_joints_hierarchy = bvh_skeleton_data[2]
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
   

    """
    Number of frames skipped is controlled with this variable below. If you want all frames, set to 1.
    """
    frame_skips = 1

    figure_limit = 2 #used to set figure axis limits

    offset = 0.0 ### seperate two bvhs when offset > 0m


    
    idx = 0

    # 检测键盘输入esc时退出
    def on_press(key):
        if key == keyboard.Key.esc:
            # 停止监听
            return False
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while listener.running:
        idx += 1
        i = idx % len(bvh_link_pos) 
        
        if isinstance(bvh_link_pos[i], torch.Tensor):
            bvh_pos = bvh_link_pos[i].detach().cpu().numpy()
        else:
            bvh_pos = bvh_link_pos[i]
        #calculate the limits of the figure. Usually the last joint in the dictionary is one of the feet.
        if figure_limit == None:
            lim_min = np.abs(np.min(bvh_pos[-1]))
            lim_max = np.abs(np.max(bvh_pos[-1]))
            lim = lim_min if lim_min > lim_max else lim_max
            figure_limit = lim
        
        
        ### visualizeing bvh
        for joint in bvh_joints:
        
            if joint == bvh_joints[0]: continue #skip root joint
            parent_joint = bvh_joints_hierarchy[joint][0]
            plt.plot(xs = [bvh_pos[reference_link.index(parent_joint)][0], bvh_pos[reference_link.index(joint)][0]],
                     zs = [bvh_pos[reference_link.index(parent_joint)][1], bvh_pos[reference_link.index(joint)][1]],
                     ys = [bvh_pos[reference_link.index(parent_joint)][2], bvh_pos[reference_link.index(joint)][2]], c = 'blue', lw = 2.5)

        ### visualizing urdf 
        frames_to_draw = [robot_link[0]]
        urdf_pos = urdf_link_pos[i].detach().cpu().numpy()

        while frames_to_draw != []:
            link = frames_to_draw[0]
            frames_to_draw.pop(0)
            for child_frame in urdf_chain.find_frame(link).children:
                child_link = child_frame.name 
                frames_to_draw.append(child_link)
                ### by default apply rotation on robot frame
                plt.plot(xs = [urdf_pos[robot_link.index(link)][1] + offset, urdf_pos[robot_link.index(child_link)][1] + offset],
                         zs = [urdf_pos[robot_link.index(link)][2], urdf_pos[robot_link.index(child_link)][2]],
                         ys = [urdf_pos[robot_link.index(link)][0], urdf_pos[robot_link.index(child_link)][0]],c = 'red', lw = 2.5)


        ### visualizing corespondence
        for ii, jj, v in correspondence:
            plt.plot(xs = [urdf_pos[jj][1] + offset, bvh_pos[ii][0]],
                    zs = [urdf_pos[jj][2], bvh_pos[ii][1]],
                    ys = [urdf_pos[jj][0], bvh_pos[ii][2]],c = 'green', lw = 2.5)

            #uncomment here if you want to see the world coords. If nothing appears on screen, change the axis limits below!
            # plt.plot(xs = [world_pos[parent_joint][0], world_pos[joint][0]],
            #          zs = [world_pos[parent_joint][1], world_pos[joint][1]],
            #          ys = [world_pos[parent_joint][2], world_pos[joint][2]], c = 'red', lw = 2.5)

        #Depending on the file, the axis limits might be too small or too big. Change accordingly.
        ax.set_axis_off()
        ax.set_xlim(-0.6*figure_limit, 0.6*figure_limit)
        ax.set_ylim(-0.6*figure_limit, 0.6*figure_limit)
        ax.set_zlim(-0.2*figure_limit, 1.*figure_limit)
        plt.title('frame: ' + str(i))
        plt.pause(0.01)
        ax.cla()

    pass



def vis_kinematic_result(filename, dataset="SG", robot="g1_15", correspondence=SG_G1_CORRESPONDENCE):
    ### loading estimated joint angles
    with open(filename, "rb") as file:
        data_dict = pickle.load(file)
        joints_angle = data_dict["angles"]

    
    ### loading dataset links
    match dataset:
        case "SG":
            reference_link = SG_LINKS 
        case "SeG":
            reference_link = SEG_LINKS
        case "MDM":
            reference_link = SMPL_LINKS
        case "HumanML3D":
            reference_link = SMPL_LINKS
        case "BEAT":
            reference_link = BEAT_LINKS
        case "motion_capture":
            reference_link = MOTION_CAPTURE_LINKS
        


    ### loading bvh data

    ### bvh data
    if dataset in ["SG", "SeG", "BEAT", "motion_capture"]:
        # match dataset:
        #     case "SG":
        #         bvh_path = os.path.join(DATA_ROOT, f"motion/human/{dataset}", filename.split("/")[-1][:-7] + ".bvh")
        #     case "Seg":
        #         bvh_path = os.path.join(DATA_ROOT, f"motion/human/{dataset}", filename.split("/")[-1][:-7] + ".bvh")
        #     case "BEAT":
        #         ### todo update beat motion pth
        #         bvh_path = os.path.join(DATA_ROOT, f"motion/human/BEAT_ZIP/beat_english_v0.2.1/1", filename.split("/")[-1][:-7] + ".bvh")
        bvh_path = data_dict["reference_motion_pth"]
        
        skeleton_data = ProcessBVH(bvh_path)
        bvh_joint_local_coord = Get_bvh_joint_local_coord(bvh_path, link_list=reference_link)


        num_frames = len(bvh_joint_local_coord)
        print("Num of frames: ", num_frames)


    ### npy data
    elif dataset in ["MDM", "HumanML3D"]:
        ### creating pseudo skeleton for smpl-like joints
        # match dataset:
        #     case "MDM":
        #         npy_path = os.path.join(DATA_ROOT, f"motion/human/{dataset}", filename.split("/")[-1][:-7] + ".npy")
        #     case "HumanML3D":
        #         npy_path = os.path.join(DATA_ROOT, f"motion/human/HumanML3D/new_joints", filename.split("/")[-1][:-7] + ".npy")

        npy_path = data_dict["reference_motion_pth"]
        joint_data = np.load(npy_path)
        skeleton_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        skeleton = {}
        for chain in skeleton_chain:
            for idx, link_idx in enumerate(chain):
                if idx == 0:
                    continue
                skeleton[SMPL_LINKS[chain[idx]]] = [SMPL_LINKS[chain[idx-1]]]
        skeleton_data = [SMPL_LINKS, None, skeleton]
        num_frames = len(joint_data)
        bvh_joint_local_coord = joint_data
    


    ### loading robot model 
    match robot:
        case "galbot":
            model = Galbot_Charlie_Motion_Model(num_frames)
            robot_link = GALBOT_CHARLIE_LINKS
        case "g1_15":
            model = G1_15_Motion_Model(num_frames)
            robot_link = G1_LINKS
        case "g1_29":
            model = G1_29_Motion_Model(num_frames)
            robot_link = G1_LINKS
        case "g1_inspirehands":
            model = G1_Inspirehands_Motion_Model(num_frames)
            robot_link = G1_INSPIREHANDS_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()
   
    
    model.set_angles(torch.tensor(joints_angle))
    model.set_global_matrix(data_dict)

    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3]


    Draw_bvh_urdf(bvh_joint_local_coord, skeleton_data, link_to_root_pos, model.chain, reference_link=reference_link, robot_link=robot_link, correspondence=correspondence)


def vis_solami_result(filename, robot='g1_29'):
    data_dict = np.load(filename, allow_pickle=True)['t2m'].item()
    link_to_root_pos = data_dict['pred']
    print("link_to_root_pos shape: ", link_to_root_pos.shape)
    num_frames = link_to_root_pos.shape[0]
    ### loading robot model 
    match robot:
        case "g1_29":
            model = G1_29_Motion_Model(num_frames)
            robot_link = G1_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()

    text_prompt = data_dict['text']
        

    Draw_Motiongpt_bvh(link_to_root_pos, model.chain, robot_link=robot_link, text_prompt=text_prompt)


def vis_motiongpt_result(filename, robot="g1_29"):
    
    link_to_root_pos = np.load(filename)[0]
    print("link_to_root_pos shape: ", link_to_root_pos.shape)
    num_frames = link_to_root_pos.shape[0]
    ### loading robot model 
    match robot:
        case "g1_29":
            model = G1_29_Motion_Model(num_frames)
            robot_link = G1_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()

    text_prompt_path = filename[:-7] + "in.txt"
    with open(text_prompt_path, "r") as file:
        text_prompt = file.read()
        print("text_prompt: ", text_prompt)
        

    Draw_Motiongpt_bvh(link_to_root_pos, model.chain, robot_link=robot_link, text_prompt=text_prompt)
    


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Call the function with the motion file')
        filename = os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle")

    else:
        filename = sys.argv[1]
    vis_kinematic_result(filename, dataset="motion_capture", robot="g1_inspirehands", correspondence=MOTION_CAPTURE_G1_INSPIREHANDS_CORRESPONDENCE)


    # vis_motiongpt_result(filename)
    # vis_solami_result(filename)






