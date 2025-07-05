
import math
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pytorch_kinematics as pk

from HRI_retarget.utils.torch_utils.diff_quat import vec6d_to_matrix

from HRI_retarget.config.joint_mapping import G1_LINKS, G1_LOWERBODY_LINKS, SG_G1_CORRESPONDENCE
from HRI_retarget.config.joint_mapping import G1_COLLISION_CAPSULE, G1_COLLISION
from HRI_retarget import DATA_ROOT

from HRI_retarget.model.g1_base_model import G1_Base_Motion_Model
from HRI_retarget.utils.motion_lib.strechable_chain import load_urdf_as_stretchable_chain


class G1_29_Motion_Model(G1_Base_Motion_Model):
    def __init__(self, batch_size=1, joint_correspondence=SG_G1_CORRESPONDENCE, device="cuda:0"):
        super(G1_29_Motion_Model, self).__init__(batch_size=batch_size, joint_correspondence=joint_correspondence, device=device)
      
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

        self.chain = None

        self.links = G1_LINKS
        self.lower_body_links = G1_LOWERBODY_LINKS

        ### apply scale and transformation on robot frame
        ### full body retargeting should have different rot and trans for each frame
        self.scale = nn.Parameter(torch.ones(3).to(device), requires_grad=True)
        self.global_rot = nn.Parameter(torch.eye(3)[:, :2].repeat(self.batch_size, 1, 1).to(device), requires_grad=True)
        self.global_trans = nn.Parameter(torch.zeros(3).reshape(3, 1).repeat(self.batch_size, 1, 1).to(device), requires_grad=True)

        ### modify lowerbody scale to match robot and human shape
        self.lower_body_scale = torch.ones(3).requires_grad_(False).to(device)

        urdf_rel_path = "resources/robots/g1_asap/g1_29dof.urdf"
        self.chain = load_urdf_as_stretchable_chain(os.path.join(DATA_ROOT,urdf_rel_path)).to(dtype=torch.float32, device=self.device)
        
    

    def forward_kinematics(self):
        """
        chain: pytorch_kinematics.chain.Chain
        joint_angle: (N, dof) 24D vector
        global_translation: (N, 3) 3D vector, root_to_world
        global_orientation: (N, 3) 3D axis-angle, root_to_world

        return: a tensor contains the global poses of 52 links
        """
        R = vec6d_to_matrix(self.global_rot) * self.scale.repeat(self.batch_size, 3, 1) # (N_frame, 3, 3)
        t = self.global_trans # (N_frame, 3, 1)
        root_to_world = torch.cat((torch.cat((R, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.batch_size, 1, 1).to(self.device)), dim=1)  # (N_frame, 4, 4)
        
        R_lower_body = vec6d_to_matrix(self.global_rot) * self.scale.repeat(self.batch_size, 3, 1) * self.lower_body_scale.repeat(self.batch_size, 3, 1)# (N_frame, 3, 3)
        lower_body_root_to_world = torch.cat((torch.cat((R_lower_body, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.batch_size, 1, 1).to(self.device)), dim=1)  # (N_frame, 4, 4)
        

        link_to_root_dict = self.chain.forward_kinematics(self.joint_angles, self.joint_scales)  # link to root
        link_to_world_dict = []
        for link_name in self.links:
            T = link_to_root_dict[link_name].get_matrix()  # link to root
            if link_name in self.lower_body_links:
                link_to_world_dict.append(torch.einsum('bij,bjk->bik', lower_body_root_to_world, T))
            else:
                link_to_world_dict.append(torch.einsum('bij,bjk->bik', root_to_world, T))


        link_to_world_dict = torch.stack(link_to_world_dict, dim=1) # (N_frame, 52, 4, 4)
        return link_to_world_dict




