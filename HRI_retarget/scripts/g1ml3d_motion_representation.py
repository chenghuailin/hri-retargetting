import os
import subprocess
from HRI_retarget import DATA_ROOT
from tqdm import tqdm
import time 
from datetime import timedelta
from queue import Queue
from threading import Thread
import pickle
import numpy as np

from HRI_retarget.utils.io.g1_29_humanml3d_representation import data_pkl_to_vec

# 文件夹路径
folder_path = os.path.join(DATA_ROOT,"motion/g1/HumanML3D")
tgt_dir = os.path.join("${HOME}/codebase/HRI_retarget/data/G1ML3D/new_joint_vecs")
starting_time = time.time()

source_list = os.listdir(folder_path)


for source_file in tqdm(source_list):
    try:

        with open(os.path.join(folder_path, source_file), "rb") as file:
            data_dict = pickle.load(file)

        vec = data_pkl_to_vec(data_dict)

        save_file = source_file.split(".")[0] + ".npy"
        np.save(os.path.join(tgt_dir, save_file), vec)
    except Exception as e:
        print(source_file)
        print(e)


