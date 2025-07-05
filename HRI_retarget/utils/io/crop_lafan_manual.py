### convert asap pkl files to unitree csv files

### usage python src/utils/io/corp_lafan_manual.py

import sys
import os
from HRI_retarget import DATA_ROOT

import joblib
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as sRot
from HRI_retarget.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
import torch
from HRI_retarget.utils.isaac_utils.rotations import (
    quaternion_to_matrix,
    wxyz_to_xyzw,
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quat_mul_norm,
    quat_identity_like,
    quat_inverse,
    quat_angle_axis
)


def crop_csv(input_path, output_path):

    ### maunally editing the unitree dancing motion

    full_csv_data = np.genfromtxt(input_path, delimiter=',')
    csv_data = full_csv_data[170:700, :]

    csv_goal = full_csv_data[170]
    csv_goal[:2] = csv_data[-38, :2]
    
    ### add end state
    for i in range(12):
        csv_data[-i - 26] = i / 12 * csv_data[-38] + (12 - i) / 12 * csv_goal
    csv_data = csv_data[:-26]

    ### issue : for the last 12 frames, right foot is sliding
    
    ### computing pose_aa 
    cfg = {
        "assetRoot": os.path.join(DATA_ROOT,"resources/robots/g1_asap"),
        "assetFileName": "g1_29dof_anneal_23dof_fitmotionONLY.xml",
        "extend_config": [{
                "joint_name": "left_hand_link",
                "parent_name": "left_elbow_link",
                "pos": [0.25, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0] # w x y z
            },{
                "joint_name": "right_hand_link",
                "parent_name": "right_elbow_link",
                "pos": [0.25, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0] # w x y z
            },{
                "joint_name": "head_link",
                "parent_name": "torso_link",
                "pos": [0.0, 0.0, 0.42],
                "rot": [1.0, 0.0, 0.0, 0.0] # w x y z
            },
        ]
    }

    mesh_parsers = Humanoid_Batch(cfg)

    root_rot_vec = torch.from_numpy(sRot.from_quat(csv_data[:, 3:7].astype(np.float32)).as_rotvec()).float()
    data_dof = np.zeros((csv_data.shape[0], 23))
    data_dof[:, :19] = csv_data[:, 7:26]
    data_dof[:, 19:23] = csv_data[:, 29:33]

    pose_aa = torch.cat([root_rot_vec[:, None, :], mesh_parsers.dof_axis * data_dof[:, :, None], torch.zeros((csv_data.shape[0], 3, 3))], axis = 1).float()
    print(pose_aa.shape)

    pose_quat = axis_angle_to_quaternion(pose_aa)
    pose_mat = quaternion_to_matrix(pose_quat)


    curr_motion = mesh_parsers.fk_batch(pose_aa[None, ], torch.from_numpy(csv_data[None, :, :3]).float(), return_full= True)
    print(curr_motion.keys())
    print("global_rotation", curr_motion["global_rotation"].shape)
    print("global_translation", curr_motion["global_translation"].shape)

    foot_pos = curr_motion["global_translation"][0, -12, 11, :]
    for i in range(12):
        csv_data[-i, :3] = (torch.tensor(csv_data[-i, :3]) + foot_pos - curr_motion["global_translation"][0, -i, 11, :]).numpy()


    print(curr_motion["global_translation"][0, -10:, 11, :])
    print(curr_motion["global_translation"][0, -10:, 12, :])

    print(curr_motion["global_rotation"][0, -1:, 11, :])
    print(curr_motion["global_rotation"][0, -1:, 12, :])
    
    print(curr_motion["dof_pos"].shape)



    np.savetxt(output_path, csv_data, delimiter=',', fmt='%.8f')



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/g1/LAFAN1/dance1_subject2_crop1.csv"))
    parser.add_argument('--input', type=str, help="csv file name", default=os.path.join(DATA_ROOT,"motion/g1/LAFAN1/dance1_subject2.csv"))
    args = parser.parse_args()

    # asap_to_csv(args.pkl, args.csv)
    crop_csv(args.input, args.output)





