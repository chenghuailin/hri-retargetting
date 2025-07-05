### TODO: estimate global rotation and translation
###     split locomotion and manipulation vel and accel loss
import math
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pytorch_kinematics as pk

from HRI_retarget.utils.torch_utils.diff_quat import vec6d_to_matrix

from HRI_retarget.config.joint_mapping import GALBOT_CHARLIE_LINKS, SG_GALBOT_CHARLIE_CORRESPONDENCE
from HRI_retarget import DATA_ROOT


class Galbot_Charlie_Motion_Model(nn.Module):
    def __init__(self, batch_size=1, joint_correspondence=SG_GALBOT_CHARLIE_CORRESPONDENCE, device="cuda:0"):
        super(Galbot_Charlie_Motion_Model, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.gt_joint_positions = None

        self.dof = 21
        ### TODO: update init angles
        self.init_angle = torch.tensor([
            0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 1.0, -1.0, 0.3, 1.3, 0.0, 0.0, 0.0, 1.0, -1.0, 0.3, 1.3, 0.0, 0.0, 0.0
        ]).to(self.device)


        self.joint_angles = nn.Parameter(torch.zeros(batch_size, self.dof).to(device), requires_grad=True)  # (N, dof)

        self.joint_correspondence = joint_correspondence

        self.chain = None

        self.links = GALBOT_CHARLIE_LINKS

        self.scale = nn.Parameter(torch.ones(3).to(device), requires_grad=True)
        self.global_rot = nn.Parameter(torch.eye(3)[:, :2].to(device), requires_grad=True)
        self.global_trans = nn.Parameter(torch.zeros(3).to(device), requires_grad=True)

        urdf_rel_path = "resources/robots/galbot_one_charlie_10/galbot_one_charlie_retarget.urdf"
        self.load_urdf_as_chain(os.path.join(DATA_ROOT,urdf_rel_path))
        
    
    def forward(self):
        return {
            "joint_angles": self.joint_angles,    
        }

    def load_urdf_as_chain(self, filename):
        with open(filename, 'rb') as file:
            self.chain = pk.build_chain_from_urdf(file.read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.device)

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
        pred_joint_velocities *= (pred_joint_velocities.abs() > 0.2 )

        return pred_joint_velocities.abs().sum(dim=-1).mean(), pred_joint_accel.abs().sum(dim=-1).mean()

    def retarget_joint_loss(self):
        pred_link_global = self.forward_kinematics()
        joint_global_position_loss = 0

        # gt_pos = self.gt_joint_positions @ self.scale + torch.clamp(self.root_diff, -0.1, 0.1)
        # print(self.scale, self.root_diff)
        for joint_corr in self.joint_correspondence:
            joint_global_position_loss += ((pred_link_global[:, joint_corr[1]][:, :3, 3] - self.gt_joint_positions[:, joint_corr[0]])**2).sum(dim=-1).mean() * joint_corr[2]
        return joint_global_position_loss

    def elbow_loss(self):
        ### robot specific loss
        ### each elbow should be at least {threshold}m away from spine
        threshold = 0.1
        pred_link_global = self.forward_kinematics()
        elbow_loss = 0.0
        right_elbow_x = pred_link_global[:, GALBOT_CHARLIE_LINKS.index("right_arm_link3"), 0, 3]

        elbow_loss += (threshold - right_elbow_x[right_elbow_x < threshold]).sum()
        left_elbow_x = pred_link_global[:, GALBOT_CHARLIE_LINKS.index("left_arm_link3"), 0, 3]
        elbow_loss += (left_elbow_x[left_elbow_x > -threshold] + threshold).sum()

        return elbow_loss




