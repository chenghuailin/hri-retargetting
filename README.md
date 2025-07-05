# What's in this repo

1. retarget for galbot from seg_dataset
2. retarget for g1 15 & 29 dof from seg_dataset & SG(semantic gesticulator output) & humanml3d & mdm output 
3. visualization of unitree motion csv via rerun 
4. data conversion between motion pkl(ours) & humanml3d npy & SG bvh & ASAP reference motion
5. dataset preparetion for Humanml3D feature format (buggy now)

### TODO LIST:
- use better log:
    create a log folder out of src folder; the log folder can be placed in data folder
    stop using absolute path 
    you can use from HRI_retarget import ROOT, from HRI_retarget import DATA_ROOT
- fix fps error: 
    fps should be in corespondence with source data; double check each file in HRI_retarget/retarget
    stop abrupt copy functions from one file to another without double-checking. e.g. fps in sg_g1_inspirehands.py is wrong. I have fixed this.
- sorting code
    write a simplified and unified version of {dataset}_g1_inspirehands.py. Current code is hard to adjust to other datasets.
        maybe split the hand part to an independent file
        It's error-prone to rename so many dicts and keys like SG_LINKS.index('RightHandThumb3'). a small typo here may lead to many bugs that is hard to detect
            e.g.    left_wrist = BBDB_LINKS.index("Phy_LeftWrist_Root_end")
                    right_wrist = BBDB_LINKS.index("RightHand")   in bbdb_hand_retarget.py
                why? which joint do you need?
        maybe you need some abstraction
    utils functions should come from a file in the correct utils folder. 
        e.g. from HRI_retarget.utils.vis.bvh_vis import calc_relative_transform
            self.calc_dist_between_rotations in g1_inspirehands.py
        you can move them to a new file in utils/motion_lib
    some file names are purely random or wrong:
        stop naming files like 3_pinocchio_output_collision_links. importing python module with a number at the beginning is not supported
            why you need a random filename with number in it?
        bbdb_hand_retarget.py and sg_g1_inspirehand.py should have same functions, why are they named differently?
            name it in a {dataset}_{robot}.py
    deleting unused functions and files.
        not commentting them. We can recover files from git history. Why do I need 200 lines of commented code that NOBODY REMEMBERS HOW TO USE THEM
        basically I want to delete all the files in retarget/stale. Check if you still need some of them.
        when you update some function in one file, rememeber to also update other retarget files(or simply move them to stale folder)
    split configs/joint_mapping.py.
        now there are so many stuff in this file. It's hard to edit now.
        firstly maybe split it into several smaller configs.
        learn to use hydra/dotmap for config management
    stop making the codebase SHIT CODE if you still need others to use your code
        write readme, usage of files, or even docs. Keep them updated.
- online version of retarget
- rewrite model to use solvers instead of optimization
    maybe we only need a modified version of ik-solver (for faster retarget)
- use different scale parameter for upperbody and lower body
    there is commented experimental code in model/g1_29.py
    but learning on xyz scale of lowerbody will lead to robot foot floating
    i want to learn xy and fix z, but the code have some bug now
- fix 277/280 feature dim error in humanml3d feature computation


# preparation
write current folder path into bashrc, then
    git clone git@github.com:ritsu-a/HRI_retarget.git       
    cd HRI_retarget
    git checkout g1
    cd HRI_retarget 
    pip install -e .


the environment is tested with cuda12.1 with python=3.10
    pip install -r requirements.txt

contact pengyang for data
then create a soft link to HRI_retarget/data




    


# Basic usage:
cd HRI_retarget

semantic gesticulator bvh to galbot dof position retarget:
    python src/retarget/sg_galbot.py data/motion/human/SG/output.bvh

    python src/retarget/seg_galbot.py data/motion/human/SeG_dataset/bvh/ARMS_FOLD-1.bvh
after this, you should generate a file as data/motion/galbot/SG/ARMS_FOLD-1.pickle


visualize kinematic results:
    python src/utils/vis/kinematic_vis.py data/motion/galbot/SG/output.pickle

visualize dynamic results via pybullet and position pd control:
    python src/utils/vis/pybullet_visualize_galbot_dynamic.py   data/motion/galbot/SG/output.pickle


### for g1:
    python src/retarget/sg_g1_15.py data/motion/human/SG/output.bvh


    python src/utils/vis/pybullet_visualize_g1_dynamic.py data/motion/g1/SG/output.pickle



### read here if you want to reproduce the speech+motion example:
1. contact pengyang for SG data, urdf file
2. run python src/retarget/sg_galbot.py {path_to_bvh_file} to generate a pickle file containing robot motion
3. run python src/utils/vis/pybullet_visualize_galbot_dynamic.py {path_to_pickle_file} to visualize robot motion in pybullet
4. find the corresponding audio file, and make your own video

tips: you may need to change some paths for bvh and pickle files, it may take sometime to debug


### use gradio for control robot with mobile phone
1. sudo ufw allow 7990
4. python src/utils/vis/gradio_test.py


### real deployment pipeline:
1. retarget to generate the raw motion 
    src/retarget/sg_g1_15.py 
2. apply clipping and filter of the raw motion
    src/deploy/filter_motion.py #TODO change IO
3. visualize motion either via pybullet(dynamic) or rerun (kinamatic)
    src/utils/vis/rerun_kinematic.py 
    src/utils/vis/pybullet_visualize_g1_dynamic.py  #TODO change IO
4. deploy on real G1 
    src/deploy/deploy_g1.py


### convert fbx to bvh
blender -b -P src/utils/io/fbx_to_bvh.py

### hand_retarget
1. pip install dex_retarget
2. ask HIT-xiaowangzi for inspire_hand_left_virtual.urdf, inspire_hand_right_virtual.urdf, inspire_hand.yml
   and put them into data/resources/robots/g1_inspirehands
3. python bbdb_hand_retarget.py
4. to visualize: python rerun_kinematic.py






