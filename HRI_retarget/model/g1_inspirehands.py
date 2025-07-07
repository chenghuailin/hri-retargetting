import math
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pytorch_kinematics as pk

from HRI_retarget.utils.torch_utils.diff_quat import vec6d_to_matrix

from HRI_retarget.config.joint_mapping import G1_INSPIREHANDS_LINKS, BEAT_G1_INSPIREHANDS_CORRESPONDENCE, G1_LOWERBODY_LINKS, \
G1_INSPIREHANDS_COLLISION_CAPSULE, G1_INSPIREHANDS_COLLISION_CUBOID, G1_INSPIREHANDS_ALLOWED_COLLISION_CFG
# from HRI_retarget.config.joint_mapping import G1_COLLISION_CAPSULE, G1_COLLISION
### TODO replace with g1_inspirehand collision
from HRI_retarget import DATA_ROOT

from HRI_retarget.utils.motion_lib.strechable_chain import load_urdf_as_stretchable_chain
from HRI_retarget.model.g1_base_model import G1_Base_Motion_Model

from dex_retargeting.retargeting_config import RetargetingConfig
import yaml
# add collision loss for whole body
from HRI_retarget.collision.segment_dist_lib import calc_seg2cuboid_dist, calc_seg2seg_dist
from HRI_retarget.utils.vis.bvh_vis import calc_relative_transform



class G1_Inspirehands_Motion_Model(G1_Base_Motion_Model):
    def __init__(self, batch_size=1, joint_correspondence=BEAT_G1_INSPIREHANDS_CORRESPONDENCE, device="cuda:0"):
        super(G1_Inspirehands_Motion_Model, self).__init__(batch_size=batch_size, joint_correspondence=joint_correspondence, device=device)

      
        self.dof = 53

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
                [0, 1.7], ##L_index_proximal_joint 22
                [0, 1.7], ##L_index_intermediate_joint 23
                [0, 1.7], ##L_middle_proximal_joint 24
                [0, 1.7], ##L_middle_intermediate_joint 25
                [0, 1.7], ##L_pinky_proximal_joint 26
                [0, 1.7], ##L_pinky_intermediate_joint 27
                [0, 1.7], ##L_ring_proximal_joint 28
                [0, 1.7], ##L_ring_intermediate_joint 29
                [-0.1, 1.3], ## L_thumb_proximal_yaw_joint 30
                [-0.1, 0.6], ##L_thumb_proximal_pitch_joint 31
                [0, 0.8], ##L_thumb_intermediate_joint 32
                [0, 1.2], ##L_thumb_distal_joint 33
                [-3.0892, 2.6704], ## right_shoulder_pitch
                [-2.2515, 1.5882], ## right_should_roll
                [-2.618, 2.618], ## right_shoulder_yaw
                [-1.0472, 2.0944], ## right_elbow
                [-1.972222054, 1.972222054], ## right_wrist_roll
                [-1.614429558, 1.614429558], ## right_wrist_pitch
                [-1.614429558, 1.614429558], ## right_wrist_yaw
                [0, 1.7], ##R_index_proximal_joint 41
                [0, 1.7], ##R_index_intermediate_joint 42
                
                [0, 1.7], ##R_middle_proximal_joint
                [0, 1.7], ##R_middle_intermediate_joint
                [0, 1.7], ##R_pinky_proximal_joint
                [0, 1.7], ##R_pinky_intermediate_joint
                [0, 1.7], ##R_ring_proximal_joint
                [0, 1.7], ##R_ring_intermediate_joint
                [-0.1, 1.3],##R_thumb_proximal_yaw_joint
                [-0.1, 0.6], ##R_thumb_proximal_pitch_joint
                [0, 0.8], ##R_thumb_intermediate_joint
                [0, 1.2], ##R_thumb_distal_joint
        ])).repeat(self.batch_size, 1, 1).to(dtype=torch.float32, device=self.device)


        ### soft threshold
        self.dof_limits = self.dof_max_limits * 0.9
       

        ### joint scales upper and lower bound 
        self.joint_scales_min = 0.7
        self.joint_scales_max = 1.3

        ## learnable parameters 
        self.joint_angles = nn.Parameter(torch.zeros(batch_size, self.dof).to(device), requires_grad=True)  # (N, dof)
        self.joint_scales = nn.Parameter(torch.ones(self.dof).to(device), requires_grad=True)  # (dof)

        self.joint_correspondence = joint_correspondence

        self.chain = None

        self.links = G1_INSPIREHANDS_LINKS
        self.lower_body_links = G1_LOWERBODY_LINKS

        ### apply scale and transformation on robot frame
        ### full body retargeting should have different rot and trans for each frame
        self.scale = nn.Parameter(torch.ones(3).to(device), requires_grad=True)
        # self.global_rot = nn.Parameter(torch.eye(3)[:, :2].repeat(self.batch_size, 1, 1).to(device), requires_grad=True)
        # self.global_trans = nn.Parameter(torch.zeros(3).reshape(3, 1).repeat(self.batch_size, 1, 1).to(device), requires_grad=True)

        ### modify lowerbody scale to match robot and human shape
        self.lower_body_scale = torch.ones(3).requires_grad_(False).to(device)

        urdf_rel_path = "resources/robots/g1_inspirehands/G1_inspire_hands.urdf"
        self.chain = load_urdf_as_stretchable_chain(os.path.join(DATA_ROOT,urdf_rel_path)).to(dtype=torch.float32, device=self.device)
        # self.chain.print_tree()
        
        ### init collision options to realize fully parallelizing
        all_segs, all_radii = [], []
        for g1, g2 in G1_INSPIREHANDS_ALLOWED_COLLISION_CFG['seg2seg']:
            group1 = G1_INSPIREHANDS_COLLISION_CAPSULE[g1]
            group2 = G1_INSPIREHANDS_COLLISION_CAPSULE[g2]
            for (_, (l11, l12, r1)) in group1.items():
                for (_, (l21, l22, r2)) in group2.items():
                    all_segs.append((l11, l12, l21, l22))
                    all_radii.append(r1 + r2)
        self.seg_idx = torch.tensor(all_segs, dtype=torch.long)   # (M,4)
        # self.seg_idx = torch.zeros((self.batch_size * len(all_segs),4))
        self.seg_radii = torch.tensor(all_radii).to(self.device)                 # (M,)
        # print(self.seg_radii)
        # self.seg_radii = torch.zeros((self.batch_size * len(all_radii),1)).to(self.device)
        # for i in range(len(all_radii)):
        #     self.seg_radii[self.batch_size * i:self.batch_size * (i+1),...] = torch.tensor((all_radii[i])).repeat(self.batch_size,1).to(self.device)
        

        # 聚合所有seg2cuboid配置到单一张量
        all_cuboid, cuboid_params = [], []  # 存储 (l1, l2, link, radius)
        self.cuboid_lookup = []             # 存储对应的长方体参数序列
        for g_caps, cub in G1_INSPIREHANDS_ALLOWED_COLLISION_CFG['seg2cuboid']:
            caps = G1_INSPIREHANDS_COLLISION_CAPSULE[g_caps]
            link, xlim, ylim, zlim = G1_INSPIREHANDS_COLLISION_CUBOID[cub]
            for (_, (l1, l2, r)) in caps.items():
                all_cuboid.append((l1, l2, link, r))
                self.cuboid_lookup.append(np.array((xlim[0],xlim[1],ylim[0],ylim[1], zlim[0],zlim[1])))
        self.cuboid_idx = torch.tensor(all_cuboid, dtype=torch.long).to(self.device)  # (N,4)
        self.cuboid_radii = torch.zeros((len(self.cuboid_idx))).to(self.device)
        for i in range(len(self.cuboid_lookup)):
            self.cuboid_radii[i,...] = torch.tensor((all_cuboid[i][3])).to(self.device)
        # self.cuboid_radii = torch.zeros((self.batch_size*len(self.cuboid_idx),1)).to(self.device)
        # for i in range(len(self.cuboid_lookup)):
        #     self.cuboid_radii[self.batch_size * i:self.batch_size * (i+1),...] = torch.tensor((all_cuboid[i][3])).repeat(self.batch_size,1).to(self.device)
        
        self.cuboid_lookup_batch = torch.zeros((self.batch_size*len(self.cuboid_idx),6)).to(self.device)
        for i in range(len(self.cuboid_lookup)):
            self.cuboid_lookup_batch[self.batch_size * i:self.batch_size * (i+1),...] = torch.from_numpy(self.cuboid_lookup[i]).repeat(self.batch_size,1).to(self.device)
        
    

    def forward_kinematics(self):
        """
        chain: pytorch_kinematics.chain.Chain
        joint_angle: (N, dof) 24D vector
        global_translation: (N, 3) 3D vector, root_to_world
        global_orientation: (N, 3) 3D axis-angle, root_to_world

        return: a tensor contains the global poses of 52 links
        """
        ### TODO: adjust the place to multiply the scale
        R = vec6d_to_matrix(self.global_rot).repeat(self.batch_size, 1, 1) * self.scale.repeat(self.batch_size, 3, 1) # (N_frame, 3, 3)
        t = self.global_trans.reshape(3, 1).repeat(self.batch_size, 1, 1) # (N_frame, 3, 1)
        root_to_world = torch.cat((torch.cat((R, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.batch_size, 1, 1).to(self.device)), dim=1)  # (N_frame, 4, 4)
        
        R_lower_body = vec6d_to_matrix(self.global_rot) * self.scale.repeat(self.batch_size, 3, 1) * self.lower_body_scale.repeat(self.batch_size, 3, 1)# (N_frame, 3, 3)
        lower_body_root_to_world = torch.cat((torch.cat((R_lower_body, t), dim=-1), torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.batch_size, 1, 1).to(self.device)), dim=1)  # (N_frame, 4, 4)
        

        link_to_root_dict = self.chain.forward_kinematics(self.joint_angles, self.joint_scales)  # link to root
        # print("link_to_root_dict: ", link_to_root_dict.keys())
        link_to_world_dict = []
        for link_name in self.links:
            T = link_to_root_dict[link_name].get_matrix()  # link to root
            if link_name in self.lower_body_links:
                link_to_world_dict.append(torch.einsum('bij,bjk->bik', lower_body_root_to_world, T))
            else:
                link_to_world_dict.append(torch.einsum('bij,bjk->bik', root_to_world, T))


        link_to_world_dict = torch.stack(link_to_world_dict, dim=1) # (N_frame, 52, 4, 4)
        # print("link_to_world_dict shape: ",link_to_world_dict.shape)
        return link_to_world_dict
    
    
    def set_hand_optimizer(self,config_file_path,default_urdf_dir):
        """
        set the dex-retargeting optimizer
        """
        RetargetingConfig.set_default_urdf_dir(default_urdf_dir)
        with open(config_file_path, 'r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_hand_optimizer = left_retargeting_config.build()
        self.right_hand_optimizer = right_retargeting_config.build()

        # self.left_hand_optimizer = left_retargeting
        # self.right_hand_optimizer = right_retargeting

    def set_hand_tip_positions(self,left_hand_tip_positions, right_hand_tip_positions):
        """ 
        hand_tip_positions:(N_frammes ,5 ,3)
        """
        self.left_hand_tip_positions = left_hand_tip_positions
        self.right_hand_tip_positions = right_hand_tip_positions
        
    def set_hand_rotations(self,left_hand_rotations, right_hand_rotations):
        """
        hand_rotations:(N_frammes ,3 ,3)
        the relative rotation from LeftHand to LeftForeArm; from RightHand to RightForeArm
        """
        self.left_hand_rotations = left_hand_rotations
        self.right_hand_rotations = right_hand_rotations

    def resolve_euler_angles(self,rot):
        """
        resolve the wrist_roll,wrist_pitch, wrist_yaw from the rotations
        rot: (N_frames, 3, 3)
        return : wrist_roll, wrist_pitch, wrist_yaw(N_frames,3)
        """
        r11 = rot[:,0,0]
        r12 = rot[:,0,1]
        r13 = rot[:,0,2]
        r23 = rot[:,1,2]
        r33 = rot[:,2,2]
        
        pitch = torch.atan2(r13, torch.sqrt(r11**2 + r12**2))
        yaw = torch.atan2(-r12/torch.cos(pitch),r11/torch.cos(pitch))
        roll = torch.atan2(-r23/torch.cos(pitch),r33/torch.cos(pitch))
        return torch.stack([roll,pitch,yaw],dim=1)
    
    def refine_wrist_angle(self):
        left_angles = self.resolve_euler_angles(self.left_hand_rotations)
        right_angles = self.resolve_euler_angles(self.right_hand_rotations)
        self.joint_angles[:,19:22] = left_angles # left_wrist_roll, left_wrist_pitch, left_wrist_yaw
        self.joint_angles[:,38:41] = right_angles # right_wrist_roll, right_wrist_pitch, right_wrist_yaw

    def refine_hand_angle(self):
        self.left_qpos_list = np.zeros((self.batch_size,12))
        self.right_qpos_list = np.zeros((self.batch_size,12))
        for i in range(self.batch_size):
            left_ref = self.left_hand_tip_positions[i,:,:].numpy()
            right_ref = self.right_hand_tip_positions[i,:,:].numpy()
            left_qpos = self.left_hand_optimizer.retarget(left_ref)
            right_qpos = self.right_hand_optimizer.retarget(right_ref)
            self.left_qpos_list[i,:] = self.fingerpos_clip(left_qpos[1:])
            self.right_qpos_list[i,:] = self.fingerpos_clip(right_qpos[1:])
            
    def fingerpos_clip(self,qpos):
        limits = np.array([
            [0, 1.7], ##L_index_proximal_joint
            [0, 1.7], ##L_index_intermediate_joint
            [0, 1.7], ##L_middle_proximal_joint
            [0, 1.7], ##L_middle_intermediate_joint
            [0, 1.7], ##L_pinky_proximal_joint
            [0, 1.7], ##L_pinky_intermediate_joint
            [0, 1.7], ##L_ring_proximal_joint
            [0, 1.7], ##L_ring_intermediate_joint
            [-0.1, 1.3], ## L_thumb_proximal_yaw_joint
            [-0.1, 0.6], ##L_thumb_proximal_pitch_joint
            [0, 0.8], ##L_thumb_intermediate_joint
            [0, 1.2], ##L_thumb_distal_joint
        ])
        qpos[qpos < limits[:,0]] = limits[:,0][qpos < limits[:,0]]
        qpos[qpos > limits[:,1]] = limits[:,1][qpos > limits[:,1]]
        return qpos 
    
    # add the constraints for wrist angle
    def set_hand_rotations_world(self,left_hand_rotations, right_hand_rotations):
        """
        hand_rotations:(N_frammes ,3 ,3)
        the relative rotation from LeftHand to Hip; from RightHand to Hip
        """
        self.left_hand_rotations_world = left_hand_rotations.to(self.device)
        self.right_hand_rotations_world = right_hand_rotations.to(self.device)

    def calc_dist_between_rotations(self,rot1,rot2):
        """
        It's a mapping from SO(3)XSO(3)->R+. quantify the distance between two rotations
        rot1: (N_frames, 3, 3)
        rot2: (N_frames, 3, 3)
        return: (N_frames,)
        """
        diff_rot = torch.bmm(rot1.transpose(1,2),rot2) # (N_frames, 3, 3)
        trace = torch.clamp(diff_rot[:,0,0] + diff_rot[:,1,1] + diff_rot[:,2,2], -1.0+1e-6, 3.0-1e-6)
        angle = torch.acos((trace - 1) / 2.0)
        # print("angle: ",angle)
        dist = angle.mean()
        # print("dist: ",dist)
        return dist
    
    def orientation_loss(self):
        # extract the orientation of left_hand, right_hand from the kinematic chain
        left_id = G1_INSPIREHANDS_LINKS.index("L_hand_base_link")
        right_id = G1_INSPIREHANDS_LINKS.index("R_hand_base_link")

        pred_link_global = self.forward_kinematics() # (N_frames, 52, 4, 4)
        loss = 0
        pred_left_hand_rot = pred_link_global[:,left_id][:,:3,:3]
        pred_right_hand_rot = pred_link_global[:,right_id][:,:3,:3]

        # gt_left_hand_rot = 
        
        left_loss = self.calc_dist_between_rotations(pred_left_hand_rot, self.left_hand_rotations_world)
        right_loss = self.calc_dist_between_rotations(pred_right_hand_rot, self.right_hand_rotations_world)
        loss = left_loss + right_loss
        return loss
    
    def copy_qpos_from_bvh(self,left_hand_joints, right_hand_joints):
        left_qpos = torch.zeros((self.batch_size,12))
        right_qpos = torch.zeros((self.batch_size,12))
        
        left_qpos[:,0] = torch.mean(left_hand_joints["index"],dim = 1)
        left_qpos[:,1] = torch.mean(left_hand_joints["index"],dim = 1)
        left_qpos[:,2] = torch.mean(left_hand_joints["middle"],dim = 1)
        left_qpos[:,3] = torch.mean(left_hand_joints["middle"],dim = 1)
        left_qpos[:,4] = torch.mean(left_hand_joints["pinky"],dim = 1)
        left_qpos[:,5] = torch.mean(left_hand_joints["pinky"],dim = 1)
        left_qpos[:,6] = torch.mean(left_hand_joints["ring"],dim = 1)
        left_qpos[:,7] = torch.mean(left_hand_joints["ring"],dim = 1)
        left_qpos[:,8] = left_hand_joints["thumb"][:,0]
        left_qpos[:,9] = (left_hand_joints["thumb"][:,1] + left_hand_joints["thumb"][:,2]) / 5.0
        left_qpos[:,10] = left_qpos[:,9] * 1.6
        left_qpos[:,11] = left_qpos[:,9] * 2.4
        
        right_qpos[:,0] = torch.mean(right_hand_joints["index"],dim = 1)
        right_qpos[:,1] = torch.mean(right_hand_joints["index"],dim = 1)
        right_qpos[:,2] = torch.mean(right_hand_joints["middle"],dim = 1)
        right_qpos[:,3] = torch.mean(right_hand_joints["middle"],dim = 1)
        right_qpos[:,4] = torch.mean(right_hand_joints["pinky"],dim = 1)
        right_qpos[:,5] = torch.mean(right_hand_joints["pinky"],dim = 1)
        right_qpos[:,6] = torch.mean(right_hand_joints["ring"],dim = 1)
        right_qpos[:,7] = torch.mean(right_hand_joints["ring"],dim = 1)
        right_qpos[:,8] = right_hand_joints["thumb"][:,0]
        right_qpos[:,9] = (right_hand_joints["thumb"][:,1] + right_hand_joints["thumb"][:,2]) / 5.0
        right_qpos[:,10] = right_qpos[:,9] * 1.6
        right_qpos[:,11] = right_qpos[:,9] * 2.4
        
        self.left_hand_qpos = left_qpos
        self.right_hand_qpos = right_qpos
        
    # def collision_loss(self):
    #     ### TODO: cuda acceleration
    #     pred_link_global = self.forward_kinematics()
    #     local_pos = pred_link_global[:,:,:3,3]
    #     local_Rot = pred_link_global[:,:,:3,:3]

    #     loss = 0
        
    #     # 1. capsule to capsule loss
    #     for pairs in G1_INSPIREHANDS_ALLOWED_COLLISION_CFG["seg2seg"]:
    #         group1 = G1_INSPIREHANDS_COLLISION_CAPSULE[pairs[0]]
    #         group2 = G1_INSPIREHANDS_COLLISION_CAPSULE[pairs[1]]
    #         for i in group1:
    #             body1_link1, body1_link2 ,body1_radius = group1[i]
    #             for j in group2:
    #                 body2_link1, body2_link2 ,body2_radius = group2[j]
    #                 body1_P1 = pred_link_global[:,body1_link1][:,:3,3]
    #                 body1_P2 = pred_link_global[:,body1_link2][:,:3,3]
    #                 body2_Q1 = pred_link_global[:,body2_link1][:,:3,3]
    #                 body2_Q2 = pred_link_global[:,body2_link2][:,:3,3]
                    
    #                 ### analytical dist between two capsules
    #                 seg_distance = calc_seg2seg_dist(body1_P1,body1_P2,body2_Q1,body2_Q2)
    #                 penetrate_dist = (body1_radius + body2_radius - seg_distance).clamp(min=0)
    #                 loss += (penetrate_dist ** 2).sum(dim=-1).mean()   
        
    #     # 2. capsule to cuboid loss (filtered by colliision group)
    #     for pairs in G1_INSPIREHANDS_ALLOWED_COLLISION_CFG["seg2cuboid"]:
    #         group1 = G1_INSPIREHANDS_COLLISION_CAPSULE[pairs[0]]
    #         group2 = G1_INSPIREHANDS_COLLISION_CUBOID[pairs[1]]
    #         for i in group1:
    #             body1_link1, body1_link2 ,body1_radius = group1[i]
    #             body2_link, xlim,ylim,zlim = group2
    #             body1_relpos1, _ = calc_relative_transform(local_pos, local_Rot, body2_link, body1_link1)
    #             body1_relpos2, _= calc_relative_transform(local_pos, local_Rot, body2_link, body1_link2)
    #             dist = calc_seg2cuboid_dist(body1_relpos1.squeeze(),body1_relpos2.squeeze(), xlim,ylim,zlim)
    #             penetrate_dist = (body1_radius - dist).clamp(min = 0)
    #             loss += (penetrate_dist ** 2).sum(dim=-1).mean()  
                
        
    #     return loss
    def collision_loss(self):
        
        # set safety thres
        safety_thres = 0.03
        # pred_global: (B, L, 4, 4)
        pred_link_global = self.forward_kinematics()
        pos = pred_link_global[:,:,:3,3].to(self.device)  # (B, L, 3)
        R = pred_link_global[:,:,:3, :3].to(self.device)  # (B, L, 3, 3)
        loss = torch.tensor(0.0, device=pos.device)
        B = self.batch_size

        # --- 全局并行：seg2seg ---
        P1 = pos[:, self.seg_idx[:, 0],:].view(-1,3)# (B*M, 3)
        P2 = pos[:, self.seg_idx[:, 1],:].view(-1,3)
        Q1 = pos[:, self.seg_idx[:, 2],:].view(-1,3)
        Q2 = pos[:, self.seg_idx[:, 3],:].view(-1,3)
        dist_seg = calc_seg2seg_dist(P1, P2, Q1, Q2).view(self.batch_size,-1)  # (B * M, 1)
        # print("dist_seg shape:",dist_seg.size())
        pen_seg = ((self.seg_radii- dist_seg) + safety_thres).clamp(min=0)
        # print("self radii shape: ",self.seg_radii.size())
        mask = pen_seg > 0
        if(torch.any(mask)):
            # print("Capture mask > 0!")
        # loss = loss + pen_seg.pow(2).sum(dim = -1).mean()
            loss   += torch.abs(pen_seg)[mask].mean()

        # --- 全局并行：seg2cuboid ---
        C1 = pos[:, self.cuboid_idx[:, 0],:].view(-1,3)  # (B*M,3)
        C2 = pos[:, self.cuboid_idx[:, 1],:].view(-1,3)
        centers = pos[:,self.cuboid_idx[:, 2],:].view(-1,3)   # 长度 N
        # 计算相对坐标：广播批次 B 和 N
        C1_R = R[:,self.cuboid_idx[:, 0],:,:].view(-1,3,3)
        C2_R = R[:,self.cuboid_idx[:, 1],:,:].view(-1,3,3)
        center_R = R[:,self.cuboid_idx[:, 2],:,:].view(-1,3,3) #(B*M, 3,3)
        
        # center: Rot1 C1,C2: Rot2 对应bvh_vis.py里面的calc_relative_transform
        # rel_Rot1 = torch.bmm(center_R.transpose(1,2),C1_R)
        # rel_Rot2 = torch.bmm(center_R.transpose(1,2),C2_R)
        rel_pos1 = torch.bmm(center_R.transpose(1,2), (C1-centers).unsqueeze(2))
        rel_pos2 = torch.bmm(center_R.transpose(1,2), (C2-centers).unsqueeze(2))

        # 分别计算距离并累加
        dist_cub = calc_seg2cuboid_dist(rel_pos1.squeeze(), rel_pos2.squeeze(),
                                        self.cuboid_lookup_batch[:,0:2],
                                        self.cuboid_lookup_batch[:,2:4],
                                        self.cuboid_lookup_batch[:,4:6]).view(self.batch_size,-1)  # (B, N)
        # print("dist_cub.size: ",dist_cub.size())
        # radii_cub = self.cuboid_idx[:, 3].to(dist_cub)
        # print("radii_cub.size: ",self.cuboid_radii.size())
        pen_cub = ((self.cuboid_radii - dist_cub) + safety_thres).clamp(min=0)
        # loss = loss + pen_cub.pow(2).sum(dim = -1). mean()
        mask2 = pen_cub > 0
        if(torch.any(mask2)):
            # print("capture mask2 > 0!")
            loss += torch.abs(pen_cub)[mask2].mean()

        return loss
        
        