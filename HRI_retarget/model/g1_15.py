
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
from HRI_retarget.model.g1_base_model import G1_Base_Motion_Model


class G1_15_Motion_Model(G1_Base_Motion_Model):
    def __init__(self, batch_size=1, joint_correspondence=SG_G1_CORRESPONDENCE, device="cuda:0"):
        super(G1_15_Motion_Model, self).__init__(batch_size=batch_size, joint_correspondence=joint_correspondence, device=device)

       
        self.dof = 15

        self.init_angles = torch.zeros(self.batch_size, self.dof)

        self.dof_max_limits = torch.from_numpy(np.array([
            [-2.618, 2.618],#waist_yaw
            [-0.52, 0.52],#waist_roll
            [-0.52, 0.52],#waist_pitch
            [-3.089, 2.670],#left_shoulder_pitch
            [-1.588, 2.251],#left_should_roll
            [-2.618, 2.618],#left_shoulder_yaw
            [-1.047, 2.094],#left_elbow
            [-1.972, 1.972],#left_wrist_roll
            [-1.614, 1.614],#left_wrist_yaw
            [-3.089, 2.670],#right_shoulder_pitch
            [-2.251, 1.588],#right_should_roll
            [-2.618, 2.618],#right_shoulder_yaw
            [-1.047, 2.094],#right_elbow
            [-1.942, 1.972],#rightt_wrist_roll
            [-1.614, 1.614],#right_wrist_yaw
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

        ### apply scale and transformation on robot frame
        self.scale = nn.Parameter(torch.ones(3).to(device), requires_grad=True)
        self.global_rot = nn.Parameter(torch.eye(3)[:, :2].to(device), requires_grad=True)
        self.global_trans = nn.Parameter(torch.zeros(3).to(device), requires_grad=True)

        urdf_rel_path = "resources/robots/g1_asap/g1_29dof_anneal_15dof.urdf"
        self.chain = load_urdf_as_stretchable_chain(os.path.join(DATA_ROOT,urdf_rel_path)).to(dtype=torch.float32, device=self.device)





