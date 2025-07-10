This is a motion retargetting project, adapted from https://github.com/ritsu-a/HRI_retarget/tree/galbot specifically for humanoid motion retargetting, especially for ultraman posture

Install: 
    tested on python=3.10.13 CUDA=12.1 pytorch==2.6.0

    pip install -r requirements.txt
    pip install -e .

Usage:
    python HRI_retarget/retarget/motioncapture_g1_inspirehands.py data/motion/human/motion_capture/Jappelio_rays_Skeleton.bvh _test1

    the result pickle file will be saved in data/motion/human/motion_capture/Jappelio_rays_Skeleton_retarget_test1.pickle by default

Visualization:
    bvh visualize: https://vrm-c.github.io/bvh2vrma/
    g1 motion visualize: python HRI_retarget/utils/vis/rerun_kinematic.py --file_name data/motion/g1/motion_capture/Jappelio_rays_Skeleton_test.pickle


How to improve retarget result:
    1. add hand retarget, please contact me for more details
    2. adjust HRI_retarget/configs/joint_mapping.json. This file contains the mapping from the source skeleton to the target skeleton, also weights for each mapping. Adjust the mapping should help for better retargeting result.
    3. turing losses in 
