### convert asap pkl files to unitree csv files

### usage python src/utils/io/asap_to_csv.py --pkl {pkl path} --csv {csv_path}

import sys
import os
from HRI_retarget import DATA_ROOT

import joblib
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as sRot
from HRI_retarget.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
import torch


def asap_to_csv(input_path, output_path):
    with open(input_path, "rb") as file:
        data = joblib.load(file)
    # print(data.keys())

    key = list(data.keys())[0]
    csv_data = np.zeros((data[key]["root_trans_offset"].shape[0], 36))
    csv_data[:, :3] = data[key]["root_trans_offset"]
    csv_data[:, 3:7] = data[key]["root_rot"]
    csv_data[:, 7:26] = data[key]["dof"][:, :19]
    csv_data[:, 29:33] = data[key]["dof"][:, 19:23]
    np.savetxt(output_path, csv_data, delimiter=',', fmt='%.8f')

def csv_to_asap(csv_path, pkl_path):
    csv_data = np.genfromtxt(csv_path, delimiter=',')
    exp_name = csv_path
    print(csv_data.shape)



    data = {}
    data["root_trans_offset"] = csv_data[:, :3].astype(np.float32)
    data["root_rot"] = csv_data[:, 3:7].astype(np.float32)
    data["dof"] = np.zeros((csv_data.shape[0], 23))
    data["dof"][:, :19] = csv_data[:, 7:26].astype(np.float32)
    data["dof"][:, 19:23] = csv_data[:, 29:33].astype(np.float32)


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
    root_rot_vec = torch.from_numpy(sRot.from_quat(data["root_rot"]).as_rotvec()).float()

    mesh_parsers = Humanoid_Batch(cfg)
    pose_aa = torch.cat([root_rot_vec[:, None, :], mesh_parsers.dof_axis * data["dof"][:, :, None], torch.zeros((data["root_trans_offset"].shape[0], 3, 3))], axis = 1)
    data["pose_aa"] = pose_aa.numpy().astype(np.float32)
    data["fps"] = 30


    pkl_data = {
        exp_name: data
    }
    joblib.dump(pkl_data, pkl_path)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/g1/LAFAN1/dance1_subject2_v2.pkl"))
    parser.add_argument('--csv', type=str, help="csv file name", default=os.path.join(DATA_ROOT,"motion/g1/LAFAN1/dance1_subject2_crop1.csv"))
    args = parser.parse_args()

    # asap_to_csv(args.pkl, args.csv)
    csv_to_asap(args.csv, args.pkl)





