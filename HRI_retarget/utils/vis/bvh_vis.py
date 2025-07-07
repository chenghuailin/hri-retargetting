### https://github.com/TemugeB/Python_BVH_viewer/blob/main/view_bvh.py
### usage: 
### python bvh_vis.py {path_to_bvh_file}
from datetime import timedelta
import torch

import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from HRI_retarget import DATA_ROOT


from HRI_retarget.utils.io.bvh_io import ProcessBVH
from HRI_retarget.config.joint_mapping import MOTION_CAPTURE_LINKS, SEG_LINKS
from HRI_retarget.config.joint_mapping import BEAT_LINKS, BEAT_G1_INSPIREHANDS_CORRESPONDENCE, SG_LINKS, \
BBDB_G1_INSPIREHANDS_CORRESPONDENCE, BBDB_LINKS

from tqdm import tqdm
from joblib import Parallel, delayed

import os

#rotation matrices
def Rx(ang, in_radians = False):
    if in_radians == False:
        ang = np.radians(ang)

    Rot_Mat = np.array([
        [1, 0, 0],
        [0, np.cos(ang), -1*np.sin(ang)],
        [0, np.sin(ang),    np.cos(ang)]
    ])
    return Rot_Mat

def Ry(ang, in_radians = False):
    if in_radians == False:
        ang = np.radians(ang)

    Rot_Mat = np.array([
        [np.cos(ang), 0, np.sin(ang)],
        [0, 1, 0],
        [-1*np.sin(ang), 0, np.cos(ang)]
    ])
    return Rot_Mat

def Rz(ang, in_radians = False):
    if in_radians == False:
        ang = np.radians(ang)

    Rot_Mat = np.array([
        [np.cos(ang), -1*np.sin(ang), 0],
        [np.sin(ang), np.cos(ang), 0],
        [0, 0, 1]
    ])
    return Rot_Mat

def Rx_par(ang):

    ang = ang.view(-1) * torch.pi / 180  # Flatten to (batch,)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    
    batch_size = ang.shape[0]
    Rx = torch.zeros((batch_size, 3, 3), device=ang.device, dtype=ang.dtype)
    
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cos
    Rx[:, 1, 2] = -sin
    Rx[:, 2, 1] = sin
    Rx[:, 2, 2] = cos
    
    return Rx

def Ry_par(ang):
    ang = ang.view(-1) * torch.pi / 180  # Flatten to (batch,)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    
    batch_size = ang.shape[0]
    Ry = torch.zeros((batch_size, 3, 3), device=ang.device, dtype=ang.dtype)
    
    Ry[:, 0, 0] = cos
    Ry[:, 0, 2] = sin
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sin
    Ry[:, 2, 2] = cos
    
    return Ry

def Rz_par(ang):
    # ang: (batch, 1)
    # return: (batch, 3, 3)
    ang = ang.view(-1) * torch.pi / 180 # Flatten to (batch,)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    
    batch_size = ang.shape[0]
    Rz = torch.zeros((batch_size, 3, 3), device=ang.device, dtype=ang.dtype)
    
    Rz[:, 0, 0] = cos
    Rz[:, 0, 1] = -sin
    Rz[:, 1, 0] = sin
    Rz[:, 1, 1] = cos
    Rz[:, 2, 2] = 1
    
    return Rz

#the rotation matrices need to be chained according to the order in the file
def _get_rotation_chain(joint_channels, joint_rotations):

    #the rotation matrices are constructed in the order given in the file
    Rot_Mat =  np.array([[1,0,0],[0,1,0],[0,0,1]])#identity matrix 3x3
    order = ''
    index = 0
    for chan in joint_channels: #if file saves xyz ordered rotations, then rotation matrix must be chained as R_x @ R_y @ R_z
        if chan[0].lower() == 'x':
            Rot_Mat = Rot_Mat @ Rx(joint_rotations[index])
            order += 'x'

        elif chan[0].lower() == 'y':
            Rot_Mat = Rot_Mat @ Ry(joint_rotations[index])
            order += 'y'

        elif chan[0].lower() == 'z':
            Rot_Mat = Rot_Mat @ Rz(joint_rotations[index])
            order += 'z'
        index += 1
    #print(order)
    return Rot_Mat

def _get_rotation_chain_parallel(joint_channels, joint_rotations):
    # joint_rotations: (batch * 3)
    #the rotation matrices are constructed in the order given in the file
    # Rot_Mat =  np.array([[1,0,0],[0,1,0],[0,0,1]])#identity matrix 3x3
    batch_size = joint_rotations.shape[0]
    Rot_Mat = torch.eye(3).unsqueeze(0).repeat(batch_size,1,1)
    order = ''
    index = 0
    for chan in joint_channels: #if file saves xyz ordered rotations, then rotation matrix must be chained as R_x @ R_y @ R_z
        if chan[0].lower() == 'x':
            Rot_x = Rx_par(joint_rotations[:,index])
            # Rot_Mat = Rot_Mat @ Rx(joint_rotations[index])
            Rot_Mat = torch.bmm(Rot_Mat, Rot_x)
            order += 'x'

        elif chan[0].lower() == 'y':
            Rot_y = Ry_par(joint_rotations[:,index])
            Rot_Mat = torch.bmm(Rot_Mat, Rot_y)
            order += 'y'

        elif chan[0].lower() == 'z':
            Rot_z = Rz_par(joint_rotations[:,index])
            Rot_Mat = torch.bmm(Rot_Mat,Rot_z)
            order += 'z'
        index += 1
    #print(order)
    return Rot_Mat

#Here root position is used as local coordinate origin.
def _calculate_frame_joint_positions_in_local_space(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy):

    local_positions = {}

    for joint in joints:

        #ignore root joint and set local coordinate to (0,0,0)
        if joint == joints[0]:
            local_positions[joint] = np.array([0,0,0])
            continue

        connected_joints = joints_hierarchy[joint]
        connected_joints = connected_joints[::-1]
        connected_joints.append(joint) #this contains the chain of joints that finally end with the current joint that we want the coordinate of.
        Rot = np.eye(3)
        pos = np.array([0,0,0])
        for i, con_joint in enumerate(connected_joints):
            if i == 0:
                pass
            else:
                parent_joint = connected_joints[i - 1]
                # if parent_joint != joints[0]:
                Rot = Rot @ _get_rotation_chain(joints_saved_angles[parent_joint], frame_joints_rotations[parent_joint])
            joint_pos = joints_offsets[con_joint]
            joint_pos = Rot @ joint_pos
            pos = pos + joint_pos

        local_positions[joint] = pos

    return local_positions

# 2025.04.24 HIT-xiaowangzi
# 并行计算local_joint_positions, 其中frame_joints_rotations包含batch维度
def _calculate_frame_joint_positions_in_local_space_parallel(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy):

    batch_num = frame_joints_rotations.shape[0]
    local_positions = torch.zeros_like(frame_joints_rotations) # (batch * link_num * 3)
    joint_name_to_index = {joint: idx for idx, joint in enumerate(joints)}

    for joint_ind, joint in enumerate(joints):

        #ignore root joint and set local coordinate to (0,0,0)
        # if joint == joints[0]:
        #     local_positions[joint] = np.array([0,0,0])
        #     continue
        if joint != joints[0]:
            connected_joints = joints_hierarchy[joint]
            connected_joints = connected_joints[::-1]
            connected_joints.append(joint) #this contains the chain of joints that finally end with the current joint that we want the coordinate of.
            # Rot = np.eye(3)
            # pos = np.array([0,0,0])
            Rot = torch.eye(3).unsqueeze(0).repeat(batch_num,1,1) # (batch *3*3)
            pos = torch.tensor([0,0,0]).view(1,-1,1).repeat(batch_num, 1, 1) # (batch * 3)
            for i, con_joint in enumerate(connected_joints):
                if i == 0:
                    pass
                else:
                    parent_joint = connected_joints[i - 1]
                    parent_joint_ind = joint_name_to_index[parent_joint]
                    # parent_joint_ind = joints.index(parent_joint)
                    # if parent_joint != joints[0]:
                    # input is batch * 3
                    
                    Rot = torch.bmm(Rot,_get_rotation_chain_parallel
                            (joints_saved_angles[parent_joint], frame_joints_rotations[:,parent_joint_ind,:]))
                joint_pos = torch.from_numpy(joints_offsets[con_joint]).view(1,-1,1).repeat(batch_num,1,1).type(torch.float32)
                # print("joint_pos.shape: ", joint_pos.shape)
                # print("Rot.shape: ", Rot.shape)
                # joint_pos = Rot @ joint_pos
                # joint_pos = torch.bmm(Rot, joint_pos) # (batch * 3)
    
                joint_pos = torch.bmm(Rot, joint_pos)
                # print("joint_pos shape: ",joint_pos.shape)
                # print("pos shape: ",pos.shape)
                pos = pos + joint_pos

            # local_positions[joint] = pos
            local_positions[:,joint_ind,:] = pos.squeeze(2)

    return local_positions


def _calculate_frame_joint_positions_in_world_space(local_positions, root_position, root_rotation, saved_angles):

    world_pos = {}
    for joint in local_positions:
        pos = local_positions[joint]

        Rot = _get_rotation_chain(saved_angles, root_rotation)
        pos = Rot @ pos

        pos = np.array(root_position) + pos
        world_pos[joint] = pos

    return world_pos

def Draw_bvh(joints, joints_offsets, joints_hierarchy, root_positions, joints_rotations, joints_saved_angles):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    frame_joints_rotations = {en:[] for en in joints}

    """
    Number of frames skipped is controlled with this variable below. If you want all frames, set to 1.
    """
    frame_skips = 1

    figure_limit = None #used to set figure axis limits

    for i in range(0,len(joints_rotations), frame_skips):

        frame_data = joints_rotations[i]

        #fill in the rotations dict
        joint_index = 0
        for joint in joints:
            frame_joints_rotations[joint] = frame_data[joint_index:joint_index+3]
            joint_index += 3

        #this returns a dictionary of joint positions in local space. This can be saved to file to get the joint positions.
        local_pos = _calculate_frame_joint_positions_in_local_space(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy)

        #calculate world positions
        world_pos = _calculate_frame_joint_positions_in_world_space(local_pos, root_positions[i], frame_joints_rotations[joints[0]], joints_saved_angles[joints[0]])

        #calculate the limits of the figure. Usually the last joint in the dictionary is one of the feet.
        if figure_limit == None:
            lim_min = np.abs(np.min(local_pos[list(local_pos)[-1]]))
            lim_max = np.abs(np.max(local_pos[list(local_pos)[-1]]))
            lim = lim_min if lim_min > lim_max else lim_max
            figure_limit = lim

        for joint in joints:
            if joint == joints[0]: continue #skip root joint
            parent_joint = joints_hierarchy[joint][0]
            plt.plot(xs = [local_pos[parent_joint][0], local_pos[joint][0]],
                     zs = [local_pos[parent_joint][1], local_pos[joint][1]],
                     ys = [local_pos[parent_joint][2], local_pos[joint][2]], c = 'blue', lw = 2.5)

            #uncomment here if you want to see the world coords. If nothing appears on screen, change the axis limits below!
            plt.plot(xs = [world_pos[parent_joint][0], world_pos[joint][0]],
                     zs = [world_pos[parent_joint][1], world_pos[joint][1]],
                     ys = [world_pos[parent_joint][2], world_pos[joint][2]], c = 'red', lw = 2.5)

        #Depending on the file, the axis limits might be too small or too big. Change accordingly.
        ax.set_axis_off()
        ax.set_xlim(-0.6*figure_limit, 0.6*figure_limit)
        ax.set_ylim(-0.6*figure_limit, 0.6*figure_limit)
        ax.set_zlim(-0.2*figure_limit, 1.*figure_limit)
        plt.title('frame: ' + str(i))
        plt.pause(0.001)
        ax.cla()

    pass
def Get_bvh_joint_local_coord(filename, link_list=SEG_LINKS):
    skeleton_data = ProcessBVH(filename)

    joints = skeleton_data[0]
    print("BVH links: ", joints)
    print("joint_num: ", len(joints))

    joints_offsets = skeleton_data[1]
    joints_hierarchy = skeleton_data[2]
    root_positions = skeleton_data[3]
    joints_rotations = skeleton_data[4] #this contains the angles in degrees
    joints_saved_angles = skeleton_data[5] #this contains channel information. E.g ['Xrotation', 'Yrotation', 'Zrotation']

    frame_joints_rotations = {en:[] for en in joints}

    """
    Number of frames skipped is controlled with this variable below. If you want all frames, set to 1.
    """
    frame_skips = 1

    joints_coord_full = torch.zeros(len(joints_rotations) // frame_skips, len(link_list), 3)

    print("Loading bvh data ... ")
    for i in tqdm(range(0,len(joints_rotations), frame_skips)):

        frame_data = joints_rotations[i]

        #fill in the rotations dict
        joint_index = 0
        for joint in joints:
            frame_joints_rotations[joint] = frame_data[joint_index:joint_index+3]
            joint_index += 3

        #this returns a dictionary of joint positions in local space. This can be saved to file to get the joint positions.
        local_pos = _calculate_frame_joint_positions_in_local_space(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy)

        #calculate world positions
        # world_pos = _calculate_frame_joint_positions_in_world_space(local_pos, root_positions[i], frame_joints_rotations[joints[0]], joints_saved_angles[joints[0]])
        
        joints_coord = []
        for joint in link_list:
            joints_coord.append(torch.from_numpy(local_pos[joint]))
        joints_coord_full[0 + i * frame_skips] = torch.stack(joints_coord, dim=0)
        
    return joints_coord_full / 100


def Get_bvh_joint_local_coord_parallel(filename, link_list=SG_LINKS):
    # 1. preprocess
    skeleton_data = ProcessBVH(filename)

    joints = skeleton_data[0]
    joints_offsets = skeleton_data[1]
    joints_hierarchy = skeleton_data[2]
    root_positions = skeleton_data[3]
    joints_rotations = skeleton_data[4] #this contains the angles in degrees
    joints_saved_angles = skeleton_data[5]
    print("joints: ", joints)
    
    # 2. convert joints_rotations to torch, and reshape it to (batch, link_num,3)
    frame_skips = 1
    joints_rotations = joints_rotations[::frame_skips,:]
    frame_num = joints_rotations.shape[0]
    joints_rotations_batch = torch.from_numpy(joints_rotations).view([frame_num,-1,3])
    
    # 3. send the batch joint rotations to the func
    joint_coords_full = _calculate_frame_joint_positions_in_local_space_parallel(
        joints, joints_offsets, joints_rotations_batch, joints_saved_angles, joints_hierarchy
    )
    # joint_coords_full, joint_rot_full = _calculate_local_pos_and_Rot(
    #     joints, joints_offsets, joints_rotations_batch, joints_saved_angles, joints_hierarchy
    # )
    
    joint_name_to_index = {joint: idx for idx, joint in enumerate(joints)}

    print("joints: ", joints)

    link_indices = [joint_name_to_index[name] for name in link_list]
    return joint_coords_full[:, link_indices, :] / 100
    # return joint_coords_full / 100
    
    

def Get_bvh_joint_world_coord(filename, link_list=SEG_LINKS):
    skeleton_data = ProcessBVH(filename)

    joints = skeleton_data[0]
    joints_offsets = skeleton_data[1]
    joints_hierarchy = skeleton_data[2]
    root_positions = skeleton_data[3]
    joints_rotations = skeleton_data[4] #this contains the angles in degrees
    joints_saved_angles = skeleton_data[5] #this contains channel information. E.g ['Xrotation', 'Yrotation', 'Zrotation']

    frame_joints_rotations = {en:[] for en in joints}

    """
    Number of frames skipped is controlled with this variable below. If you want all frames, set to 1.
    """
    frame_skips = 1

    joints_coord_full = torch.zeros(len(joints_rotations) // frame_skips, len(link_list), 3)


    for i in range(0,len(joints_rotations), frame_skips):

        frame_data = joints_rotations[i]

        #fill in the rotations dict
        joint_index = 0
        for joint in joints:
            frame_joints_rotations[joint] = frame_data[joint_index:joint_index+3]
            joint_index += 3

        #this returns a dictionary of joint positions in local space. This can be saved to file to get the joint positions.
        local_pos = _calculate_frame_joint_positions_in_local_space(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy)

        #calculate world positions
        world_pos = _calculate_frame_joint_positions_in_world_space(local_pos, root_positions[i], frame_joints_rotations[joints[0]], joints_saved_angles[joints[0]])
        
        joints_coord = []
        for joint in link_list:
            joints_coord.append(torch.from_numpy(world_pos[joint]))
        joints_coord_full[0 + i * frame_skips] = torch.stack(joints_coord, dim=0)
        
    return joints_coord_full / 100


def Draw_bvh_frame(joints, joints_offsets, joints_hierarchy, root_positions, joints_rotations, joints_saved_angles, frame_id, fig):

    ax = fig.add_subplot(111, projection='3d')

    frame_joints_rotations = {en:[] for en in joints}

    
    i = frame_id

    frame_data = joints_rotations[i]

    #fill in the rotations dict
    joint_index = 0
    for joint in joints:
        frame_joints_rotations[joint] = frame_data[joint_index:joint_index+3]
        joint_index += 3

    #this returns a dictionary of joint positions in local space. This can be saved to file to get the joint positions.
    local_pos = _calculate_frame_joint_positions_in_local_space(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy)

    #calculate world positions
    world_pos = _calculate_frame_joint_positions_in_world_space(local_pos, root_positions[i], frame_joints_rotations[joints[0]], joints_saved_angles[joints[0]])

    #calculate the limits of the figure. Usually the last joint in the dictionary is one of the feet.
    if True:
        lim_min = np.abs(np.min(local_pos[list(local_pos)[-1]]))
        lim_max = np.abs(np.max(local_pos[list(local_pos)[-1]]))
        lim = lim_min if lim_min > lim_max else lim_max
        figure_limit = lim

    for joint in joints:
        if joint == joints[0]: continue #skip root joint
        parent_joint = joints_hierarchy[joint][0]
        plt.plot(xs = [local_pos[parent_joint][0], local_pos[joint][0]],
                    zs = [local_pos[parent_joint][1], local_pos[joint][1]],
                    ys = [local_pos[parent_joint][2], local_pos[joint][2]], c = 'blue', lw = 2.5)

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
    plt.pause(0.001)

    ax.cla()
    
def Get_bvh_joint_pos_and_Rot(filename, link_list=SG_LINKS):
    # 1. preprocess
    skeleton_data = ProcessBVH(filename)

    joints = skeleton_data[0]
    joints_offsets = skeleton_data[1]
    joints_hierarchy = skeleton_data[2]
    root_positions = skeleton_data[3]
    joints_rotations = skeleton_data[4] #this contains the angles in degrees
    joints_saved_angles = skeleton_data[5]
    print("joints: ", joints)
    
    # 2. convert joints_rotations to torch, and reshape it to (batch, link_num, 3)
    frame_skips = 1
    joints_rotations = joints_rotations[::frame_skips,:]
    frame_num = joints_rotations.shape[0]
    joints_rotations_batch = torch.from_numpy(joints_rotations).view([frame_num,-1,3])
    
    # 3. send the batch joint rotations to the func
    # joint_coords_full = _calculate_frame_joint_positions_in_local_space_parallel(
    #     joints, joints_offsets, joints_rotations_batch, joints_saved_angles, joints_hierarchy
    # )                                                              
    joint_coords_full, joint_rot_full = _calculate_local_pos_and_Rot(
        joints, joints_offsets, joints_rotations_batch, joints_saved_angles, joints_hierarchy
    )
    
    joint_name_to_index = {joint: idx for idx, joint in enumerate(joints)}
                            
    link_indices = [joint_name_to_index[name] for name in link_list]
    return joint_coords_full[:, link_indices, :] / 100, joint_rot_full[:, link_indices, :, :]
    # return joint_coords_full / 100

def Get_bvh_joint_global_pos(filename, link_list=SG_LINKS):
    # 1. preprocess
    skeleton_data = ProcessBVH(filename)

    joints = skeleton_data[0]
    joints_offsets = skeleton_data[1]
    joints_hierarchy = skeleton_data[2]
    root_positions = skeleton_data[3]
    joints_rotations = skeleton_data[4] #this contains the angles in degrees
    joints_saved_angles = skeleton_data[5]
    print("joints: ", joints)
    
    # 2. convert joints_rotations to torch, and reshape it to (batch, link_num,3)
    frame_skips = 1
    joints_rotations = joints_rotations[::frame_skips,:]
    frame_num = joints_rotations.shape[0]
    joints_rotations_batch = torch.from_numpy(joints_rotations).view([frame_num,-1,3])
    
    # 3. send the batch joint rotations to the func
 
    joint_coords_full = _calculate_global_pos(
        joints, joints_offsets, joints_rotations_batch, joints_saved_angles, root_positions, joints_hierarchy
    )
    
    joint_name_to_index = {joint: idx for idx, joint in enumerate(joints)}

    link_indices = [joint_name_to_index[name] for name in link_list]
    return joint_coords_full[:, link_indices, :] / 100

   
def Get_bvh_joint_angles(filename, link_list=SG_LINKS):
     # 1. preprocess
    skeleton_data = ProcessBVH(filename)

    joints = skeleton_data[0]
    joints_offsets = skeleton_data[1]
    joints_hierarchy = skeleton_data[2]
    root_positions = skeleton_data[3]
    joints_rotations = skeleton_data[4] #this contains the angles in degrees
    joints_saved_angles = skeleton_data[5]
    
    frame_num = joints_rotations.shape[0]
    joints_rotations_batch = torch.from_numpy(joints_rotations).view([frame_num,-1,3])
    return joints, joints_rotations_batch
    
    
    
    
    
# 2025.04.28
# Add: calculate the local position and orientation
# because we need the transformation from the hand base link to the fingertip
def _calculate_local_pos_and_Rot(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy):

    batch_num = frame_joints_rotations.shape[0]
    link_num = frame_joints_rotations.shape[1]
    local_positions = torch.zeros_like(frame_joints_rotations) # (batch * link_num * 3)
    local_Rotations = torch.eye(3).view(1,1,3,3).repeat(batch_num, link_num, 1, 1)
    joint_name_to_index = {joint: idx for idx, joint in enumerate(joints)}

    for joint_ind, joint in enumerate(joints):

        if joint != joints[0]:
            connected_joints = joints_hierarchy[joint]
            
            connected_joints = connected_joints[::-1]
            connected_joints.append(joint) #this contains the chain of joints that finally end with the current joint that we want the coordinate of.
            # Rot = np.eye(3)
            # pos = np.array([0,0,0])
            Rot = torch.eye(3).unsqueeze(0).repeat(batch_num,1,1) # (batch *3*3)
            pos = torch.tensor([0,0,0]).view(1,-1,1).repeat(batch_num, 1, 1) # (batch * 3)
            for i, con_joint in enumerate(connected_joints):
                if i == 0:
                    pass
                else:
                    parent_joint = connected_joints[i - 1]
                    parent_joint_ind = joint_name_to_index[parent_joint]
                    # parent_joint_ind = joints.index(parent_joint)
                    # if parent_joint != joints[0]:
                    # input is batch * 3
                    
                    Rot = torch.bmm(Rot,_get_rotation_chain_parallel
                            (joints_saved_angles[parent_joint], frame_joints_rotations[:,parent_joint_ind,:]))
                joint_pos = torch.from_numpy(joints_offsets[con_joint]).view(1,-1,1).repeat(batch_num,1,1).type(torch.float32)
                
                # print("joint_pos.shape: ", joint_pos.shape)
                # print("Rot.shape: ", Rot.shape)
                # joint_pos = Rot @ joint_pos
                # joint_pos = torch.bmm(Rot, joint_pos) # (batch * 3)
    
                joint_pos = torch.bmm(Rot, joint_pos)
                # print("joint_pos shape: ",joint_pos.shape)
                # print("pos shape: ",pos.shape)
                pos = pos + joint_pos
                real_Rot = torch.bmm(Rot,_get_rotation_chain_parallel
                            (joints_saved_angles[joint], frame_joints_rotations[:,joint_ind,:]))

            # local_positions[joint] = pos
            local_positions[:,joint_ind,:] = pos.squeeze(2)
            local_Rotations[:,joint_ind,:,:] = real_Rot

    return local_positions, local_Rotations

def _calculate_global_pos(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, root_positions, joints_hierarchy):

    batch_num = frame_joints_rotations.shape[0]
    link_num = frame_joints_rotations.shape[1]
    local_positions = torch.zeros_like(frame_joints_rotations) # (batch * link_num * 3)
    joint_name_to_index = {joint: idx for idx, joint in enumerate(joints)}

    root_positions = root_positions - root_positions[0] # make the root position relative to the first frame

    for joint_ind, joint in enumerate(joints):

        if joint != joints[0]:
            connected_joints = joints_hierarchy[joint]
            
            connected_joints = connected_joints[::-1]
            connected_joints.append(joint) #this contains the chain of joints that finally end with the current joint that we want the coordinate of.
            # Rot = np.eye(3)
            # pos = np.array([0,0,0])
            Rot = torch.eye(3).unsqueeze(0).repeat(batch_num,1,1) # (batch *3*3)
            pos = torch.tensor([0,0,0]).view(1,-1,1).repeat(batch_num, 1, 1) # (batch * 3)
            for i, con_joint in enumerate(connected_joints):
                if i == 0:
                    pass
                else:
                    parent_joint = connected_joints[i - 1]
                    parent_joint_ind = joint_name_to_index[parent_joint]
                    # parent_joint_ind = joints.index(parent_joint)
                    # if parent_joint != joints[0]:
                    # input is batch * 3
                    
                    Rot = torch.bmm(Rot,_get_rotation_chain_parallel
                            (joints_saved_angles[parent_joint], frame_joints_rotations[:,parent_joint_ind,:]))
                joint_pos = torch.from_numpy(joints_offsets[con_joint]).view(1,-1,1).repeat(batch_num,1,1).type(torch.float32)
                
                # print("joint_pos.shape: ", joint_pos.shape)
                # print("Rot.shape: ", Rot.shape)
                # joint_pos = Rot @ joint_pos
                # joint_pos = torch.bmm(Rot, joint_pos) # (batch * 3)
    
                joint_pos = torch.bmm(Rot, joint_pos)
                # print("joint_pos shape: ",joint_pos.shape)
                # print("pos shape: ",pos.shape)
                pos = pos + joint_pos + torch.from_numpy(root_positions).unsqueeze(2)
               

            # local_positions[joint] = pos
            local_positions[:,joint_ind,:] = pos.squeeze(2)

    return local_positions

def calc_relative_transform(local_positions, local_Rotations, id1, id2):
    """
    Calculate the batchwise transformation from the id2 to id1
    """
    # 1. get the joint ind from the joints list
    pos1 = local_positions[:,id1,:].unsqueeze(2) # (batch * 3 * 1)
    Rot1 = local_Rotations[:,id1,:,:] # (batch * 3 * 3)
    pos2 = local_positions[:,id2,:].unsqueeze(2) # (batch * 3 * 1)
    Rot2 = local_Rotations[:,id2,:,:] # (batch * 3 * 3)
    
    
    # 2. calculate the relative transform
    rel_Rot = torch.bmm(Rot1.transpose(1,2), Rot2)
    # rel_pos = pos1 - torch.bmm(rel_Rot,pos2)
    # rel_pos = rel_pos.squeeze(2)
    rel_pos =torch.bmm(Rot1.transpose(1,2),(pos2-pos1))
    # rel_pos = rel_pos.squeeze(2)
    
    return rel_pos, rel_Rot
 
    

if __name__ == "__main__":

    # if len(sys.argv) != 2:
    #     print('Call the function with the BVH file')
    #     quit()

    # filename = sys.argv[1]
    filename = os.path.join(DATA_ROOT,"motion/human/motion_capture/defense_Skeleton.bvh")
    # filename = os.path.join(DATA_ROOT,"motion/human/SG/output.bvh")
    
    bvh_joint_local_coord_parallel = Get_bvh_joint_local_coord_parallel(filename, link_list = MOTION_CAPTURE_LINKS)
    
    print("Parallel Process: ", bvh_joint_local_coord_parallel[0])
    
    skeleton_data = ProcessBVH(filename)
    print(skeleton_data[4].shape)
    print(skeleton_data[0])
    Draw_bvh(*skeleton_data[:6])