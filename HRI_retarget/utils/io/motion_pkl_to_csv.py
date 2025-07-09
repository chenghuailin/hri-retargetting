### convert asap pkl files to unitree csv files

### usage python src/utils/io/asap_to_csv.py --pkl {pkl path} --csv {csv_path}

import sys
import os
from HRI_retarget import DATA_ROOT

import joblib
import numpy as np
import argparse
import torch

from HRI_retarget.utils.torch_utils.diff_quat import vec6d_to_quat
from HRI_retarget.config.joint_mapping import G1_29_DOFS, G1_15_DOFS


def load_motion_pkl_as_csv_data(input_pkl_path):
    with open(input_pkl_path, "rb") as file:
        data = joblib.load(file)

    robot_name = data["robot_name"]
    match robot_name:
        case "g1_inspirehands":
            csv_data = np.zeros((data["angles"].shape[0], 60))
            csv_data[:, :3] = data["global_translation"].reshape(-1, 3)
            csv_data[:, 3:7] = vec6d_to_quat(torch.tensor(data['global_rotation'])).numpy()
            csv_data[:, 7:] = data["angles"]
        case "g1_15":
            ### csv correspond to g1_29
            csv_data = np.zeros((data["angles"].shape[0], 36))
            csv_data[:, :3] = data["global_translation"]
            csv_data[:, 3:7] = vec6d_to_quat(torch.tensor(data['global_rotation'])).numpy()
            for idx in range(15):
                csv_data[:, 7 + G1_29_DOFS.index(G1_15_DOFS[idx])] = data["angles"][:, idx]
        case "g1_29":
            csv_data = np.zeros((data["angles"].shape[0], 36))
            csv_data[:, :3] = data["global_translation"][:, :, 0]
            csv_data[:, 3:7] = vec6d_to_quat(torch.tensor(data['global_rotation'])).numpy()
            csv_data[:, 7:] = data["angles"]

   
        case "_":
            print("Undefined robot type: ", robot_name)
            raise ValueError('Invalid robot type')

    return csv_data

def pkl_to_csv(input_path, output_path):

    csv_data = load_motion_pkl_as_csv_data(input_path)
    np.savetxt(output_path, csv_data, delimiter=',', fmt='%.8f')

def pkl_to_npz( input_path, output_path=None):
    
    data = load_motion_pkl_as_csv_data( input_path )
    np.savez(output_path, data=data)


    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--pkl', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle"))
#     # parser.add_argument('--csv', type=str, help="csv file name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.csv"))
#     parser.add_argument('--pkl', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle"))
#     parser.add_argument('--csv', type=str, help="csv file name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.csv"))
#     args = parser.parse_args()

#     pkl_to_csv(args.pkl, args.csv)
#     # csv_to_pkl(args.csv, args.pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle"))
    parser.add_argument('--csv', type=str, help="csv file name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.csv"))
    # parser.add_argument('--pkl', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle"))
    # parser.add_argument('--npz', type=str, help="csv file name", default=os.path.join(DATA_ROOT,"motion/g1/SG/output.npz"))
    args = parser.parse_args()

    pkl_to_csv(args.pkl, args.csv)
    # csv_to_pkl(args.csv, args.pkl)


