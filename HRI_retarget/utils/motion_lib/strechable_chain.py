## modify pytorch_kinematics.chain.Chain so that we can scale each joint of the urdf in fk(if provided)
## or it can be viewed as adding new prismatic joint at each joint at the origin urdf 
## used for retargeting


from pytorch_kinematics.chain import Chain 
from functools import lru_cache
from typing import Optional, Sequence

import copy
import numpy as np
import torch

import pytorch_kinematics.transforms as tf
from pytorch_kinematics import jacobian
from pytorch_kinematics import build_chain_from_urdf
from pytorch_kinematics.frame import Frame, Link, Joint
from pytorch_kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix_44, axis_and_d_to_pris_matrix


### modified from pytorch_kinamatics.chain.Chain.forward_kinematics
class StretchableChain(Chain):
    def forward_kinematics(self, th, scale: Optional = None, frame_indices: Optional = None):
        """
        Compute forward kinematics for the given joint values.

        Args:
            th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched.
            scale: stretch scale for each joint
            frame_indices: A list of frame indices to compute transforms for. If None, all frames are computed.
                Use `get_frame_indices` to convert from frame names to frame indices.

        Returns:
            A dict of Transform3d objects for each frame.

        """
        if frame_indices is None:
            frame_indices = self.get_all_frame_indices()

        if scale is None:
            scale = torch.ones(th.shape[1], dtype=th.dtype, device=th.device)

        th = self.ensure_tensor(th)
        th = torch.atleast_2d(th)

        b = th.shape[0]
        axes_expanded = self.axes.unsqueeze(0).repeat(b, 1, 1)

        # compute all joint transforms at once first
        # in order to handle multiple joint types without branching, we create all possible transforms
        # for all joint types and then select the appropriate one for each joint.
        rev_jnt_transform = axis_and_angle_to_matrix_44(axes_expanded, th)
        pris_jnt_transform = axis_and_d_to_pris_matrix(axes_expanded, th)

        frame_transforms = {}
        b = th.shape[0]
        for frame_idx in frame_indices:
            frame_transform = torch.eye(4).to(th).unsqueeze(0).repeat(b, 1, 1)

            # iterate down the list and compose the transform
            for chain_idx in self.parents_indices[frame_idx.item()]:
                if chain_idx.item() in frame_transforms:
                    frame_transform = frame_transforms[chain_idx.item()]
                else:
                    link_offset_i = self.link_offsets[chain_idx]
                    if link_offset_i is not None:
                        frame_transform = frame_transform @ link_offset_i

                    jnt_idx = self.joint_indices[chain_idx]
                    

                    joint_offset_i = self.joint_offsets[chain_idx]
                    if joint_offset_i is not None:
                        joint_offset_i = joint_offset_i.clone() 
                        joint_offset_i[:,:3,3] *= scale[jnt_idx]
                        frame_transform = frame_transform @ joint_offset_i

                    jnt_type = self.joint_type_indices[chain_idx]
                    if jnt_type == 0:
                        pass
                    elif jnt_type == 1:
                        jnt_transform_i = rev_jnt_transform[:, jnt_idx]
                        frame_transform = frame_transform @ jnt_transform_i
                    elif jnt_type == 2:
                        jnt_transform_i = pris_jnt_transform[:, jnt_idx]
                        frame_transform = frame_transform @ jnt_transform_i

            frame_transforms[frame_idx.item()] = frame_transform

        frame_names_and_transform3ds = {self.idx_to_frame[frame_idx]: tf.Transform3d(matrix=transform) for
                                        frame_idx, transform in frame_transforms.items()}

        return frame_names_and_transform3ds


def load_urdf_as_stretchable_chain(filename):
    with open(filename, 'rb') as file:
        original_chain = build_chain_from_urdf(file.read())  # 获取原始 chain 对象
    strechable_chain = StretchableChain.__new__(StretchableChain)
    strechable_chain.__dict__ = original_chain.__dict__.copy()  # 复制属性
    return strechable_chain