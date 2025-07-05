### TODO: collision loss
###         update limit loss to hard constraints + inverse-like loss
###         the current collision loss is totally nonsense
###             try with better approximation
import math
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pytorch_kinematics as pk

from HRI_retarget.utils.torch_utils.diff_quat import vec6d_to_matrix

from HRI_retarget.config.joint_mapping import G1_LINKS, SG_G1_CORRESPONDENCE
from HRI_retarget.config.joint_mapping import G1_COLLISION_CAPSULE, G1_COLLISION
from HRI_retarget import DATA_ROOT
from HRI_retarget.utils.motion_lib.strechable_chain import load_urdf_as_stretchable_chain

from HRI_retarget.collision.segment_dist_lib import calc_seg2seg_dist,calc_point2seg_dist

class G1_Base_Motion_Model(nn.Module):
    def __init__(self, batch_size=1, joint_correspondence=SG_G1_CORRESPONDENCE, device="cuda:0"):
        super(G1_Base_Motion_Model, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.gt_joint_positions = None

        self.dof = 29

        self.init_angles = torch.zeros(self.batch_size, self.dof)

        self.dof_max_limits = torch.from_numpy(np.array([
                [-2.530, 2.879], ## left_hip_pitch
                [-0.523, 2.967], ## left_hip_roll
                [-0.087267, 2.757], ## left_hip_yaw
                [-0.087267, 2.879], ## left_knee
                [-0.87267, 0.523], ## left_ankle_pitch
                [-0.2618, 0.2618], ## left_ankle_roll
                [-2.5307, 2.8798], ## right_hip_pitch
                [-2.9671, 0.5236], ## right_hip_roll
                [-2.7576, 2.7576], ## right_hip_yaw
                [-0.087267, 2.8798], ## right_knee
                [-0.87267, 0.5236], ## right_ankle_pitch
                [-0.2618, 0.2618], ## right_ankle_roll
                [-2.618, 2.618], ## waist_yaw
                [-0.52, 0.52], ## waist_roll
                [-0.52, 0.52], ## waist_pitch
                [-3.0892, 2.6704], ## left_shoulder_pitch
                [-1.5882, 2.2515], ## left_should_roll
                [-2.618, 2.618], ## left_shoulder_yaw
                [-1.0472, 2.0944], ## left_elbow
                [-1.972222054, 1.972222054], ## left_wrist_roll
                [-1.614429558, 1.614429558], ##left_wrist_pitch
                [-1.614429558, 1.614429558], ## left_wrist_yaw
                [-3.0892, 2.6704], ## right_shoulder_pitch
                [-2.2515, 1.5882], ## right_should_roll
                [-2.618, 2.618], ## right_shoulder_yaw
                [-1.0472, 2.0944], ## right_elbow
                [-1.972222054, 1.972222054], ## right_wrist_roll
                [-1.614429558, 1.614429558], ## right_wrist_pitch
                [-1.614429558, 1.614429558], ## right_wrist_yaw
        ])).repeat(self.batch_size, 1, 1).to(dtype=torch.float32, device=self.device)

        ### soft threshold
        self.dof_limits = self.dof_max_limits * 0.9

       
        ### joint scales upper and lower bound 
        self.joint_scales_min = 0.8
        self.joint_scales_max = 1.2

        ## learnable parameters 
        self.joint_angles = nn.Parameter(torch.zeros(batch_size, self.dof).to(device), requires_grad=True)  # (N, dof)
        self.joint_scales = nn.Parameter(torch.ones(self.dof).to(device), requires_grad=True)  # (dof)

        self.joint_correspondence = joint_correspondence


        self.links = G1_LINKS

        ### apply scale and transformation on robot frame
        self.scale = nn.Parameter(torch.ones(3).to(device), requires_grad=True)
        self.global_rot = nn.Parameter(torch.eye(3)[:, :2].to(device), requires_grad=True)
        self.global_trans = nn.Parameter(torch.zeros(3).to(device), requires_grad=True)

        urdf_rel_path = "resources/robots/g1_asap/g1_29dof_anneal_15dof.urdf"
        self.chain = load_urdf_as_stretchable_chain(os.path.join(DATA_ROOT,urdf_rel_path)).to(dtype=torch.float32, device=self.device)

        
    def forward(self):
        return {
            "joint_angles": self.joint_angles,    
        }

    def set_global_matrix(self, data_dict):
        self.global_trans = nn.Parameter(torch.tensor(data_dict["global_translation"]).to(self.device), requires_grad=True)
        self.global_rot = nn.Parameter(torch.tensor(data_dict["global_rotation"]).to(self.device), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(data_dict["scale"]).to(self.device), requires_grad=True)

    def set_gt_joint_positions(self, gt_joint_positions):
        self.gt_joint_positions = gt_joint_positions.to(self.device)

    def set_angles(self, joint_angles):
        self.joint_angles = nn.Parameter(joint_angles.to(self.device), requires_grad=True)  # (N, dof)
        return

    def forward_kinematics(self):
        """
        chain: pytorch_kinematics.chain.Chain
        joint_angle: (N, dof) 24D vector
        global_translation: (N, 3) 3D vector, root_to_world
        global_orientation: (N, 3) 3D axis-angle, root_to_world

        return: a tensor contains the global poses of 52 links
        """
        R = vec6d_to_matrix(self.global_rot).repeat(self.batch_size, 1, 1) * self.scale.repeat(self.batch_size, 3, 1) # (N_frame, 3, 3)
        t = self.global_trans.reshape(3, 1).repeat(self.batch_size, 1, 1) # (N_frame, 3, 1)
        root_to_world = torch.cat((torch.cat((R, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.batch_size, 1, 1).to(self.device)), dim=1)  # (N_frame, 4, 4)

        link_to_root_dict = self.chain.forward_kinematics(self.joint_angles)  # link to root
        link_to_world_dict = []
        for link_name in self.links:
            T = link_to_root_dict[link_name].get_matrix()  # link to root
            link_to_world_dict.append(torch.einsum('bij,bjk->bik', root_to_world, T))

        link_to_world_dict = torch.stack(link_to_world_dict, dim=1) # (N_frame, 52, 4, 4)
        return link_to_world_dict

    def init_angle_loss(self):
        return (self.joint_angles[0, 3:] - self.init_angle[3:]).abs().sum(dim=-1).mean()

    def joint_local_velocity_loss(self):
        pred_joint_velocities = self.joint_angles[1:, 3:] - self.joint_angles[:-1, 3:]
        pred_joint_accel = pred_joint_velocities[1:] - pred_joint_velocities[:-1]

        ### only regulate on too large vel
        pred_joint_velocities *= (pred_joint_velocities.abs() > 0.05 )

        return pred_joint_velocities.abs().sum(dim=-1).mean(), pred_joint_accel.abs().sum(dim=-1).mean()

    def retarget_joint_loss(self):
        pred_link_global = self.forward_kinematics()
        joint_global_position_loss = 0

        # gt_pos = self.gt_joint_positions @ self.scale + torch.clamp(self.root_diff, -0.1, 0.1)
        # print(self.scale, self.root_diff)
        for joint_corr in self.joint_correspondence:
            joint_global_position_loss += ((pred_link_global[:, joint_corr[1]][:, :3, 3] - self.gt_joint_positions[:, joint_corr[0]])**2).sum(dim=-1).mean() * joint_corr[2]
        return joint_global_position_loss
    

    def dof_limit_loss(self):

        loss =  (self.joint_angles < self.dof_limits[:, :, 0]) * (self.dof_limits[:, :, 0] - self.joint_angles) +\
                (self.joint_angles > self.dof_limits[:, :, 1]) * (self.joint_angles - self.dof_limits[:, :, 1])
        return loss.sum(dim=-1).mean()
    

    # def collision_loss(self):
    #     ### TODO: cuda acceleration
    #     pred_link_global = self.forward_kinematics()

    #     loss = 0

    #     for body1,body2 in G1_COLLISION:
    #         body1_link1,body1_link2,body1_radius = G1_COLLISION_CAPSULE[body1]
    #         body2_link1,body2_link2,body2_radius = G1_COLLISION_CAPSULE[body2]
    #         body1_P1 = pred_link_global[:,body1_link1][:,:3,3]
    #         body1_P2 = pred_link_global[:,body1_link2][:,:3,3]
    #         body2_Q1 = pred_link_global[:,body2_link1][:,:3,3]
    #         body2_Q2 = pred_link_global[:,body2_link2][:,:3,3]
            
    #         ### analytical dist between two capsules
    #         seg_distance = calc_seg2seg_dist(body1_P1,body1_P2,body2_Q1,body2_Q2)
    #         penetrate_dist = (body1_radius + body2_radius - seg_distance).clamp(min=0)
    #         loss += (penetrate_dist ** 2).sum(dim=-1).mean()
        
    #     return loss
    



    # # def elbow_loss(self):
    # #     ### robot specific loss
    # #     ### each elbow should be at least {threshold}m away from spine
    # #     threshold = 0.1
    # #     pred_link_global = self.forward_kinematics()
    # #     elbow_loss = 0.0
    # #     right_elbow_x = pred_link_global[:, GALBOT_CHARLIE_LINKS.index("right_arm_link3"), 0, 3]

    # #     elbow_loss += (threshold - right_elbow_x[right_elbow_x < threshold]).sum()
    # #     left_elbow_x = pred_link_global[:, GALBOT_CHARLIE_LINKS.index("left_arm_link3"), 0, 3]
    # #     elbow_loss += (left_elbow_x[left_elbow_x > -threshold] + threshold).sum()

    # #     return elbow_loss

    def normalize(self):
        ### clip angles within max limits
        ### TODO: torch.clamp on nn.parameter and rename
        self.joint_angles[self.joint_angles < self.dof_max_limits[:, :, 0]] = self.dof_max_limits[:, :, 0][self.joint_angles < self.dof_max_limits[:, :, 0]]
        self.joint_angles[self.joint_angles > self.dof_max_limits[:, :, 1]] = self.dof_max_limits[:, :, 1][self.joint_angles > self.dof_max_limits[:, :, 1]]

        ### clip joint scales
        self.joint_scales[self.joint_scales < self.joint_scales_min] = self.joint_scales_min    
        self.joint_scales[self.joint_scales > self.joint_scales_max] = self.joint_scales_max




