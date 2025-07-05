import numpy as np 
import argparse

import os 
from HRI_retarget import DATA_ROOT


def split_npy(folder_pth):
    npy_pth = os.path.join(folder_pth, "results.npy")
    assert os.path.exists(npy_pth)

    data = np.load(npy_pth, allow_pickle=True)
    data = data.tolist()

    num_motions = data['motion'].shape[0]

    for idx in range(num_motions):
        save_pth = os.path.join(folder_pth, f"mdm_{str(idx).zfill(5)}.npy")
        np.save(save_pth, data['motion'][idx].transpose(2, 0, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="File name", default=os.path.join(DATA_ROOT,"motion/human/MDM"))
    args = parser.parse_args()

    split_npy(args.path)