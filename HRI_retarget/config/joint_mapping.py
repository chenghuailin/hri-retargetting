### dataset link names

BBDB_LINKS = ['Hip', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 
              'RightToeBase_end', 'Spine', 'Spine1', 'RightShoulder', 'RightArm', 
              'RightForeArm', 'RightHand', 'RightHandIndex1', 'RightHandIndex2', 
              'RightHandIndex3', 'RightHandIndex4', 'RightHandIndex4_end', 'RightHandMiddle1', 
              'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightHandMiddle4_end', 
              'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 
              'RightHandPinky4_end', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 
              'RightHandRing4', 'RightHandRing4_end', 'RightHandThumb1', 'RightHandThumb2', 
              'RightHandThumb3', 'RightHandThumb4', 'RightHandThumb4_end', 'Phy_RightWrist_Root', 
              'Phy_RightWrist_Root_end', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 
              'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 'LeftHandIndex4_end', 
              'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4', 'LeftHandMiddle4_end', 
              'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 'LeftHandPinky4_end', 
              'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandRing4_end', 
              'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4', 'LeftHandThumb4_end', 
              'Phy_LeftWrist_Root', 'Phy_LeftWrist_Root_end', 'Neck', 'Head', 'Root_1', 'Root_1_end', 
              'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBase_end']

BEAT_LINKS = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 
               'Neck1', 'Head', 'HeadEnd', 'RightShoulder', 'RightArm', 
               'RightForeArm', 'RightHand', 'RightHandMiddle1', 'RightHandMiddle2', 
               'RightHandMiddle3', 'RightHandMiddle4', 'RightHandRing', 
               'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 
               'RightHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 
               'RightHandPinky4', 'RightHandIndex', 'RightHandIndex1', 'RightHandIndex2', 
               'RightHandIndex3', 'RightHandIndex4', 'RightHandThumb1', 'RightHandThumb2', 
               'RightHandThumb3', 'RightHandThumb4', 'LeftShoulder', 'LeftArm', 
               'LeftForeArm', 'LeftHand', 'LeftHandMiddle1', 'LeftHandMiddle2', 
               'LeftHandMiddle3', 'LeftHandMiddle4', 'LeftHandRing', 'LeftHandRing1', 
               'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandPinky', 
               'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 
               'LeftHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 
               'LeftHandIndex4', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 
               'LeftHandThumb4', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightForeFoot', 
               'RightToeBase', 'RightToeBaseEnd', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 
               'LeftForeFoot', 'LeftToeBase', 'LeftToeBaseEnd']

MOTION_CAPTURE_LINKS = ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 
                        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 
                        'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 
                        'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 
                        'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 
                        'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 
                        'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 
                        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 
                        'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 
                        'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 
                        'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 
                        'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 
                        'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 
                        'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

SEG_LINKS = ['Hips', 'Chest', 'Chest2', 'Neck', 'Head',
    'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 
    'LeftFinger0', 'LeftFinger01', 'LeftFinger02', 'LeftFinger1', 
    'LeftFinger11', 'LeftFinger12', 'LeftFinger2', 'LeftFinger21', 
    'LeftFinger22', 'LeftFinger3', 'LeftFinger31', 'LeftFinger32', 
    'LeftFinger4', 'LeftFinger41', 'LeftFinger42', 'RightCollar', 
    'RightShoulder', 'RightElbow', 'RightWrist', 'RightFinger0', 
    'RightFinger01', 'RightFinger02', 'RightFinger1', 'RightFinger11', 
    'RightFinger12', 'RightFinger2', 'RightFinger21', 'RightFinger22', 
    'RightFinger3', 'RightFinger31', 'RightFinger32', 'RightFinger4', 
    'RightFinger41', 'RightFinger42', 'LeftHip', 'LeftKnee', 
    'LeftAnkle', 'LeftToe', 'RightHip', 'RightKnee', 'RightAnkle', 'RightToe']

SG_LINKS = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg',
    'LeftLeg', 'LeftFoot', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 
    'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 
    'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightInHandIndex', 
    'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightInHandMiddle', 
    'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightInHandRing', 
    'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightInHandPinky', 
    'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'LeftShoulder', 
    'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 
    'LeftHandThumb3', 'LeftInHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 
    'LeftHandIndex3', 'LeftInHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 
    'LeftHandMiddle3', 'LeftInHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 
    'LeftInHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3']

SMPL_LINKS = [
    'Pelvis', # 0
    'L_Hip', # 1
    'R_Hip', # 2
    'Spine1', # 3
    'L_Knee', # 4
    'R_Knee', # 5
    'Spine2', # 6
    'L_Ankle', # 7
    'R_Ankle', # 8
    'Spine3', # 9
    'L_Foot', # 10
    'R_Foot', # 11
    'Neck', # 12
    'L_Collar', # 13
    'R_Collar', # 14
    'Head', # 15
    'L_Shoulder', # 16
    'R_Shoulder', # 17
    'L_Elbow', # 18
    'R_Elbow', # 19
    'L_Wrist', # 20
    'R_Wrist', # 21
    # 'L_Hand', # 22
    # 'R_Hand', # 23
]




### robot link names

GALBOT_CHARLIE_LINKS = ['mobile_base', 'base_link_x', 'base_link_y', 'base_link_z',
    'base_link', 'omni_chassis_base_link', 'omni_chassis_leg_mount_link', 'leg_base_link',
    'leg_link1', 'leg_link2', 'leg_link3', 'leg_link4', 
    'leg_end_effector_mount_link', 'torso_base_link', 'torso_head_mount_link', 
    'head_base_link', 'head_link1', 'head_link2', 'head_end_effector_mount_link', 
    'torso_right_arm_mount_link', 'right_arm_base_link', 'right_arm_link1', 
    'right_arm_link2', 'right_arm_link3', 'right_arm_link4', 'right_arm_link5', 
    'right_arm_link6', 'right_arm_link7', 'right_arm_end_effector_mount_link', 
    'right_arm_force_sensor_sim_view_frame', 'right_suction_cup_base_link', 
    'right_suction_cup_link1', 'right_suction_cup_tcp_link', 'right_flange_link', 
    'torso_left_arm_mount_link', 'left_arm_base_link', 'left_arm_link1', 
    'left_arm_link2', 'left_arm_link3', 'left_arm_link4', 'left_arm_link5', 
    'left_arm_link6', 'left_arm_link7', 'left_arm_end_effector_mount_link', 
    'left_arm_force_sensor_sim_view_frame', 'left_gripper_base_link', 'left_gripper_l1_link', 
    'left_gripper_l3_link', 'left_gripper_l2_link', 'left_gripper_r1_link', 'left_gripper_r3_link', 
    'left_gripper_r2_link', 'left_gripper_left_link', 'left_gripper_right_link', 'left_gripper_tcp_link', 
    'left_flange_link']

G1_LINKS = ['pelvis', 'pelvis_contour_link', 'left_hip_pitch_link', 'left_hip_roll_link', 
            'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 
            'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 
            'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 
            'torso_link', 'logo_link', 'head_link', 'waist_support_link', 'imu_link', 'd435_link', 
            'mid360_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 
            'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 
            'left_rubber_hand', 'left_mid_finger_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 
            'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand', 'right_mid_finger_link']

# G1_INSPIREHANDS_LINKS = ['pelvis', 'pelvis_contour_link', 'left_hip_pitch_link', 'left_hip_roll_link', 
#                        'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 
#                        'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 
#                        'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 
#                        'torso_link', 'logo_link', 'head_link', 'waist_support_link', 'imu_in_torso', 'd435_link', 
#                        'mid360_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 
#                        'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 
#                        'L_hand_base_link', 'L_thumb_proximal_base', 'L_thumb_proximal', 'L_thumb_intermediate', 
#                        'L_thumb_distal', 'L_index_proximal', 'L_index_intermediate', 'L_middle_proximal', 
#                        'L_middle_intermediate', 'L_ring_proximal', 'L_ring_intermediate', 'L_pinky_proximal', 
#                        'L_pinky_intermediate', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 
#                        'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 
#                        'right_wrist_yaw_link', 'R_hand_base_link', 'R_thumb_proximal_base', 'R_thumb_proximal', 
#                        'R_thumb_intermediate', 'R_thumb_distal', 'R_index_proximal', 'R_index_intermediate', 
#                        'R_middle_proximal', 'R_middle_intermediate', 'R_ring_proximal', 'R_ring_intermediate', 
#                        'R_pinky_proximal', 'R_pinky_intermediate', 'imu_in_pelvis']
G1_INSPIREHANDS_LINKS = ['pelvis', 'pelvis_contour_link', 'left_hip_pitch_link', 'left_hip_roll_link',
                         'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
                         'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link',
                         'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link',
                         'torso_link', 'logo_link', 'head_link', 'waist_support_link', 'imu_in_torso', 'd435_link',
                         'mid360_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
                         'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 
                         'L_hand_base_link', 'L_thumb_proximal_base', 'L_thumb_proximal', 'L_thumb_intermediate',
                         'L_thumb_distal', 'L_thumb_tip', 'L_index_proximal', 'L_index_intermediate', 'L_index_tip',
                         'L_middle_proximal', 'L_middle_intermediate', 'L_middle_tip', 'L_ring_proximal',
                         'L_ring_intermediate', 'L_ring_tip', 'L_pinky_proximal', 'L_pinky_intermediate', 
                         'L_pinky_tip', 'right_shoulder_pitch_link', 'right_shoulder_roll_link',
                         'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 
                         'right_wrist_pitch_link', 'right_wrist_yaw_link', 'R_hand_base_link', 'R_thumb_proximal_base', 
                         'R_thumb_proximal', 'R_thumb_intermediate', 'R_thumb_distal', 'R_thumb_tip',
                         'R_index_proximal', 'R_index_intermediate', 'R_index_tip', 'R_middle_proximal',
                         'R_middle_intermediate', 'R_middle_tip', 'R_ring_proximal', 'R_ring_intermediate',
                         'R_ring_tip', 'R_pinky_proximal', 'R_pinky_intermediate', 'R_pinky_tip', 'imu_in_pelvis']

G1_LOWERBODY_LINKS = [
    'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 
    'right_ankle_pitch_link', 'right_ankle_roll_link',
]


### robot DOF names

G1_INSPIREHANDS_DOFS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
                        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
                        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
                        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
                        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 
                        'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
                        'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 
                        'L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'L_thumb_distal_joint', 
                        'L_index_proximal_joint', 'L_index_intermediate_joint', 
                        'L_middle_proximal_joint', 'L_middle_intermediate_joint', 
                        'L_ring_proximal_joint', 'L_ring_intermediate_joint', 
                        'L_pinky_proximal_joint', 'L_pinky_intermediate_joint', 
                        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 
                        'right_shoulder_yaw_joint', 'right_elbow_joint', 
                        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 
                        'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_intermediate_joint', 'R_thumb_distal_joint', 
                        'R_index_proximal_joint', 'R_index_intermediate_joint', 'R_middle_proximal_joint', 'R_middle_intermediate_joint', 
                        'R_ring_proximal_joint', 'R_ring_intermediate_joint', 'R_pinky_proximal_joint', 'R_pinky_intermediate_joint'] ## 29 + 12 * 2

G1_29_DOFS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", 
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", 
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


G1_15_DOFS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_yaw_joint",
]


### robot self collision config

G1_COLLISION_CAPSULE = {
    ### name: link1, link2, radius
    "left_thigh": [G1_LINKS.index("left_hip_roll_link"), G1_LINKS.index("left_knee_link"), 0.08],
    "left_hand": [G1_LINKS.index("left_rubber_hand"), G1_LINKS.index("left_mid_finger_link"), 0.04],

    "right_thigh": [G1_LINKS.index("right_hip_roll_link"), G1_LINKS.index("right_knee_link"), 0.08],
    "right_hand": [G1_LINKS.index("right_rubber_hand"), G1_LINKS.index("right_mid_finger_link"), 0.04],
}

G1_COLLISION = [
    ["left_thigh", "left_hand"],
    ["right_thigh", "right_hand"],
    ["left_hand", "right_hand"],
]


### mapping from dataset to robot

SEG_GALBOT_CHARLIE_CORRESPONDENCE = [
    [SEG_LINKS.index("Hips"), GALBOT_CHARLIE_LINKS.index("leg_link3"), 5],
    [SEG_LINKS.index("Neck"), GALBOT_CHARLIE_LINKS.index("head_link1"), 5],

    [SEG_LINKS.index("LeftShoulder"), GALBOT_CHARLIE_LINKS.index("left_arm_link1"), 3],
    [SEG_LINKS.index("LeftElbow"), GALBOT_CHARLIE_LINKS.index("left_arm_link3"), 2],
    [SEG_LINKS.index("LeftWrist"), GALBOT_CHARLIE_LINKS.index("left_arm_link5"), 3],

    [SEG_LINKS.index("RightShoulder"), GALBOT_CHARLIE_LINKS.index("right_arm_link1"), 3],
    [SEG_LINKS.index("RightElbow"), GALBOT_CHARLIE_LINKS.index("right_arm_link3"), 2],
    [SEG_LINKS.index("RightWrist"), GALBOT_CHARLIE_LINKS.index("right_arm_link5"), 3],
]


SG_GALBOT_CHARLIE_CORRESPONDENCE = [
    [SG_LINKS.index("Spine"), GALBOT_CHARLIE_LINKS.index("leg_link3"), 5],
    # [SG_LINKS.index("Neck"), GALBOT_CHARLIE_LINKS.index(""), 5],

    [SG_LINKS.index("LeftArm"), GALBOT_CHARLIE_LINKS.index("left_arm_link1"), 3],
    [SG_LINKS.index("LeftForeArm"), GALBOT_CHARLIE_LINKS.index("left_arm_link3"), 2],
    [SG_LINKS.index("LeftHand"), GALBOT_CHARLIE_LINKS.index("left_arm_link5"), 3],
    
    [SG_LINKS.index("RightArm"), GALBOT_CHARLIE_LINKS.index("right_arm_link1"), 3],
    [SG_LINKS.index("RightForeArm"), GALBOT_CHARLIE_LINKS.index("right_arm_link3"), 2],
    [SG_LINKS.index("RightHand"), GALBOT_CHARLIE_LINKS.index("right_arm_link5"), 3],
]


BBDB_G1_INSPIREHANDS_CORRESPONDENCE = [
    [BBDB_LINKS.index("Spine"), G1_INSPIREHANDS_LINKS.index("pelvis"), 5],
    [BBDB_LINKS.index("Head"), G1_INSPIREHANDS_LINKS.index("mid360_link"), 5],
    # [BEAT_LINKS.index("LeftShoulder"), G1_INSPIREHANDS_LINKS.index(""), 1],
    [BBDB_LINKS.index("LeftArm"), G1_INSPIREHANDS_LINKS.index("left_shoulder_roll_link"), 5],
    [BBDB_LINKS.index("LeftForeArm"), G1_INSPIREHANDS_LINKS.index("left_elbow_link"), 5],
    [BBDB_LINKS.index("LeftHand"), G1_INSPIREHANDS_LINKS.index("L_hand_base_link"), 5],

    # [BEAT_LINKS.index("RightShoulder"), G1_INSPIREHANDS_LINKS.index(""), 1],
    [BBDB_LINKS.index("RightArm"), G1_INSPIREHANDS_LINKS.index("right_shoulder_roll_link"), 5],
    [BBDB_LINKS.index("RightForeArm"), G1_INSPIREHANDS_LINKS.index("right_elbow_link"), 5],
    [BBDB_LINKS.index("RightHand"), G1_INSPIREHANDS_LINKS.index("R_hand_base_link"), 5],


    [BBDB_LINKS.index("LeftUpLeg"), G1_INSPIREHANDS_LINKS.index("left_hip_pitch_link"), 3],
    [BBDB_LINKS.index("RightUpLeg"), G1_INSPIREHANDS_LINKS.index("right_hip_pitch_link"), 3],
    # [SMPL_LINKS.index("Spine1"), G1_LINKS.index(""), 1],
    # [SMPL_LINKS.index("Spine2"), G1_LINKS.index(""), 1],
    # [SMPL_LINKS.index("Spine3"), G1_LINKS.index(""), 1],  
    [BBDB_LINKS.index("LeftLeg"), G1_INSPIREHANDS_LINKS.index("left_knee_link"), 3],
    [BBDB_LINKS.index("RightLeg"), G1_INSPIREHANDS_LINKS.index("right_knee_link"), 3],
    [BBDB_LINKS.index("LeftFoot"), G1_INSPIREHANDS_LINKS.index("left_ankle_roll_link"), 3],
    [BBDB_LINKS.index("RightFoot"), G1_INSPIREHANDS_LINKS.index("right_ankle_roll_link"), 3],



    # [BBDB_LINKS.index("LeftHandThumb2"), G1_INSPIREHANDS_LINKS.index("L_thumb_intermediate"), 1],
    # [BBDB_LINKS.index("LeftHandThumb3"), G1_INSPIREHANDS_LINKS.index("L_thumb_distal"), 1],
    # [BBDB_LINKS.index("LeftHandIndex2"), G1_INSPIREHANDS_LINKS.index("L_index_proximal"), 1],
    # [BBDB_LINKS.index("LeftHandIndex3"), G1_INSPIREHANDS_LINKS.index("L_index_intermediate"), 1],
    [BBDB_LINKS.index("LeftHandMiddle2"), G1_INSPIREHANDS_LINKS.index("L_middle_proximal"), 2],
    [BBDB_LINKS.index("LeftHandMiddle3"), G1_INSPIREHANDS_LINKS.index("L_middle_intermediate"), 2],
    # [BBDB_LINKS.index("LeftHandRing2"), G1_INSPIREHANDS_LINKS.index("L_ring_proximal"), 1],
    # [BBDB_LINKS.index("LeftHandRing3"), G1_INSPIREHANDS_LINKS.index("L_ring_intermediate"), 1],
    # [BBDB_LINKS.index("LeftHandPinky2"), G1_INSPIREHANDS_LINKS.index("L_pinky_proximal"), 1],
    # [BBDB_LINKS.index("LeftHandPinky3"), G1_INSPIREHANDS_LINKS.index("L_pinky_intermediate"), 1],

    # [BBDB_LINKS.index("RightHandThumb2"), G1_INSPIREHANDS_LINKS.index("R_thumb_intermediate"), 1],
    # [BBDB_LINKS.index("RightHandThumb3"), G1_INSPIREHANDS_LINKS.index("R_thumb_distal"), 1],
    [BBDB_LINKS.index("RightHandIndex2"), G1_INSPIREHANDS_LINKS.index("R_index_proximal"), 5],
    [BBDB_LINKS.index("RightHandIndex3"), G1_INSPIREHANDS_LINKS.index("R_index_intermediate"), 5],
    # [BBDB_LINKS.index("RightHandMiddle2"), G1_INSPIREHANDS_LINKS.index("R_middle_proximal"), 5],
    # [BBDB_LINKS.index("RightHandMiddle3"), G1_INSPIREHANDS_LINKS.index("R_middle_intermediate"), 5],
    # [BBDB_LINKS.index("RightHandRing2"), G1_INSPIREHANDS_LINKS.index("R_ring_proximal"), 1],
    # [BBDB_LINKS.index("RightHandRing3"), G1_INSPIREHANDS_LINKS.index("R_ring_intermediate"), 1],
    # [BBDB_LINKS.index("RightHandPinky2"), G1_INSPIREHANDS_LINKS.index("R_pinky_proximal"), 1],
    # [BBDB_LINKS.index("RightHandPinky3"), G1_INSPIREHANDS_LINKS.index("R_pinky_intermediate"), 1],

   
]


BEAT_G1_INSPIREHANDS_CORRESPONDENCE = [
    [BEAT_LINKS.index("Hips"), G1_INSPIREHANDS_LINKS.index("pelvis"), 5],
    [BEAT_LINKS.index("Head"), G1_INSPIREHANDS_LINKS.index("mid360_link"), 5],
    # [BEAT_LINKS.index("LeftShoulder"), G1_INSPIREHANDS_LINKS.index(""), 1],
    [BEAT_LINKS.index("LeftArm"), G1_INSPIREHANDS_LINKS.index("left_shoulder_roll_link"), 5],
    [BEAT_LINKS.index("LeftForeArm"), G1_INSPIREHANDS_LINKS.index("left_elbow_link"), 5],
    [BEAT_LINKS.index("LeftHand"), G1_INSPIREHANDS_LINKS.index("L_hand_base_link"), 5],

    # [BEAT_LINKS.index("RightShoulder"), G1_INSPIREHANDS_LINKS.index(""), 1],
    [BEAT_LINKS.index("RightArm"), G1_INSPIREHANDS_LINKS.index("right_shoulder_roll_link"), 5],
    [BEAT_LINKS.index("RightForeArm"), G1_INSPIREHANDS_LINKS.index("right_elbow_link"), 5],
    [BEAT_LINKS.index("RightHand"), G1_INSPIREHANDS_LINKS.index("R_hand_base_link"), 5],


    [BEAT_LINKS.index("LeftUpLeg"), G1_LINKS.index("left_hip_pitch_link"), 3],
    [BEAT_LINKS.index("RightUpLeg"), G1_LINKS.index("right_hip_pitch_link"), 3],
    # [SMPL_LINKS.index("Spine1"), G1_LINKS.index(""), 1],
    # [SMPL_LINKS.index("Spine2"), G1_LINKS.index(""), 1],
    # [SMPL_LINKS.index("Spine3"), G1_LINKS.index(""), 1],  
    [BEAT_LINKS.index("LeftLeg"), G1_LINKS.index("left_knee_link"), 3],
    [BEAT_LINKS.index("RightLeg"), G1_LINKS.index("right_knee_link"), 3],
    [BEAT_LINKS.index("LeftFoot"), G1_LINKS.index("left_ankle_roll_link"), 3],
    [BEAT_LINKS.index("RightFoot"), G1_LINKS.index("right_ankle_roll_link"), 3],



    [BEAT_LINKS.index("LeftHandThumb2"), G1_INSPIREHANDS_LINKS.index("L_thumb_intermediate"), 1],
    [BEAT_LINKS.index("LeftHandThumb3"), G1_INSPIREHANDS_LINKS.index("L_thumb_distal"), 1],
    [BEAT_LINKS.index("LeftHandIndex2"), G1_INSPIREHANDS_LINKS.index("L_index_proximal"), 1],
    [BEAT_LINKS.index("LeftHandIndex3"), G1_INSPIREHANDS_LINKS.index("L_index_intermediate"), 1],
    [BEAT_LINKS.index("LeftHandMiddle2"), G1_INSPIREHANDS_LINKS.index("L_middle_proximal"), 1],
    [BEAT_LINKS.index("LeftHandMiddle3"), G1_INSPIREHANDS_LINKS.index("L_middle_intermediate"), 1],
    [BEAT_LINKS.index("LeftHandRing2"), G1_INSPIREHANDS_LINKS.index("L_ring_proximal"), 1],
    [BEAT_LINKS.index("LeftHandRing3"), G1_INSPIREHANDS_LINKS.index("L_ring_intermediate"), 1],
    [BEAT_LINKS.index("LeftHandPinky2"), G1_INSPIREHANDS_LINKS.index("L_pinky_proximal"), 1],
    [BEAT_LINKS.index("LeftHandPinky3"), G1_INSPIREHANDS_LINKS.index("L_pinky_intermediate"), 1],

    [BEAT_LINKS.index("RightHandThumb2"), G1_INSPIREHANDS_LINKS.index("R_thumb_intermediate"), 1],
    [BEAT_LINKS.index("RightHandThumb3"), G1_INSPIREHANDS_LINKS.index("R_thumb_distal"), 1],
    [BEAT_LINKS.index("RightHandIndex2"), G1_INSPIREHANDS_LINKS.index("R_index_proximal"), 1],
    [BEAT_LINKS.index("RightHandIndex3"), G1_INSPIREHANDS_LINKS.index("R_index_intermediate"), 1],
    [BEAT_LINKS.index("RightHandMiddle2"), G1_INSPIREHANDS_LINKS.index("R_middle_proximal"), 1],
    [BEAT_LINKS.index("RightHandMiddle3"), G1_INSPIREHANDS_LINKS.index("R_middle_intermediate"), 1],
    [BEAT_LINKS.index("RightHandRing2"), G1_INSPIREHANDS_LINKS.index("R_ring_proximal"), 1],
    [BEAT_LINKS.index("RightHandRing3"), G1_INSPIREHANDS_LINKS.index("R_ring_intermediate"), 1],
    [BEAT_LINKS.index("RightHandPinky2"), G1_INSPIREHANDS_LINKS.index("R_pinky_proximal"), 1],
    [BEAT_LINKS.index("RightHandPinky3"), G1_INSPIREHANDS_LINKS.index("R_pinky_intermediate"), 1],

   
]


MOTION_CAPTURE_G1_INSPIREHANDS_CORRESPONDENCE = [
    [MOTION_CAPTURE_LINKS.index("Spine"), G1_INSPIREHANDS_LINKS.index("pelvis"), 5],
    [MOTION_CAPTURE_LINKS.index("Head"), G1_INSPIREHANDS_LINKS.index("mid360_link"), 5],
    [MOTION_CAPTURE_LINKS.index("LeftArm"), G1_INSPIREHANDS_LINKS.index("left_shoulder_roll_link"), 5],
    [MOTION_CAPTURE_LINKS.index("LeftForeArm"), G1_INSPIREHANDS_LINKS.index("left_elbow_link"), 5],
    [MOTION_CAPTURE_LINKS.index("LeftHand"), G1_INSPIREHANDS_LINKS.index("L_hand_base_link"), 5],

    [MOTION_CAPTURE_LINKS.index("RightArm"), G1_INSPIREHANDS_LINKS.index("right_shoulder_roll_link"), 5],
    [MOTION_CAPTURE_LINKS.index("RightForeArm"), G1_INSPIREHANDS_LINKS.index("right_elbow_link"), 5],
    [MOTION_CAPTURE_LINKS.index("RightHand"), G1_INSPIREHANDS_LINKS.index("R_hand_base_link"), 5],


    [MOTION_CAPTURE_LINKS.index("LeftUpLeg"), G1_INSPIREHANDS_LINKS.index("left_hip_pitch_link"), 5],
    [MOTION_CAPTURE_LINKS.index("RightUpLeg"), G1_INSPIREHANDS_LINKS.index("right_hip_pitch_link"),5],
   
    [MOTION_CAPTURE_LINKS.index("LeftLeg"), G1_INSPIREHANDS_LINKS.index("left_knee_link"), 5],
    [MOTION_CAPTURE_LINKS.index("RightLeg"), G1_INSPIREHANDS_LINKS.index("right_knee_link"), 5],
    [MOTION_CAPTURE_LINKS.index("LeftFoot"), G1_INSPIREHANDS_LINKS.index("left_ankle_roll_link"), 5],
    [MOTION_CAPTURE_LINKS.index("RightFoot"), G1_INSPIREHANDS_LINKS.index("right_ankle_roll_link"), 5],



    [MOTION_CAPTURE_LINKS.index("LeftHandThumb3"), G1_INSPIREHANDS_LINKS.index("L_thumb_tip"), 2],
    [MOTION_CAPTURE_LINKS.index("LeftHandMiddle3"), G1_INSPIREHANDS_LINKS.index("L_middle_tip"), 2],
    [MOTION_CAPTURE_LINKS.index("LeftHandPinky3"), G1_INSPIREHANDS_LINKS.index("L_pinky_tip"), 2],
    
    [MOTION_CAPTURE_LINKS.index("RightHandThumb3"), G1_INSPIREHANDS_LINKS.index("R_thumb_tip"), 2],
    [MOTION_CAPTURE_LINKS.index("RightHandMiddle3"), G1_INSPIREHANDS_LINKS.index("R_middle_tip"), 2],
    [MOTION_CAPTURE_LINKS.index("RightHandPinky3"), G1_INSPIREHANDS_LINKS.index("R_pinky_tip"), 2],
   
]


SG_G1_CORRESPONDENCE = [
    [SG_LINKS.index("Spine"), G1_LINKS.index("pelvis"), 1],
    [SG_LINKS.index("Head"), G1_LINKS.index("mid360_link"), 1],

    [SG_LINKS.index("LeftArm"), G1_LINKS.index("left_shoulder_roll_link"), 3],
    [SG_LINKS.index("LeftForeArm"), G1_LINKS.index("left_elbow_link"), 3],
    # [SG_LINKS.index("LeftHand"), G1_LINKS.index("left_wrist_roll_link"), 1],
    [SG_LINKS.index("LeftHand"), G1_LINKS.index("left_rubber_hand"), 3],
    # [SG_LINKS.index("LeftHand"), G1_LINKS.index("left_wrist_yaw_link"), 1],
    [SG_LINKS.index("LeftHandMiddle3"), G1_LINKS.index("left_mid_finger_link"), 3],
    [SG_LINKS.index("RightArm"), G1_LINKS.index("right_shoulder_roll_link"), 3],
    # [SG_LINKS.index("RightArm"), G1_LINKS.index("right_shoulder_roll_link"), 1],
    # [SG_LINKS.index("RightArm"), G1_LINKS.index("right_shoulder_yaw_link"), 1],
    [SG_LINKS.index("RightForeArm"), G1_LINKS.index("right_elbow_link"), 3],
    # [SG_LINKS.index("RightHand"), G1_LINKS.index("right_wrist_roll_link"), 1],
    [SG_LINKS.index("RightHand"), G1_LINKS.index("right_rubber_hand"), 3],
    # [SG_LINKS.index("RightHand"), G1_LINKS.index("right_wrist_yaw_link"), 1],
    [SG_LINKS.index("RightHandMiddle3"), G1_LINKS.index("right_mid_finger_link"), 3],
]


SG_G1_INSPIREHANDS_CORRESPONDENCE = [
    [SG_LINKS.index("Spine"), G1_INSPIREHANDS_LINKS.index("pelvis"), 1],
    [SG_LINKS.index("Head"), G1_INSPIREHANDS_LINKS.index("mid360_link"), 1],

    [SG_LINKS.index("LeftArm"), G1_INSPIREHANDS_LINKS.index("left_shoulder_roll_link"), 3],
    [SG_LINKS.index("LeftForeArm"), G1_INSPIREHANDS_LINKS.index("left_elbow_link"), 3],
    [SG_LINKS.index("LeftHand"), G1_INSPIREHANDS_LINKS.index("L_hand_base_link"), 3],
    [SG_LINKS.index("LeftHandMiddle2"), G1_INSPIREHANDS_LINKS.index("L_middle_intermediate"), 3],
    [SG_LINKS.index("RightArm"), G1_INSPIREHANDS_LINKS.index("right_shoulder_roll_link"), 3],
    [SG_LINKS.index("RightForeArm"), G1_INSPIREHANDS_LINKS.index("right_elbow_link"), 3],
    [SG_LINKS.index("RightHand"), G1_INSPIREHANDS_LINKS.index("R_hand_base_link"), 3],
    [SG_LINKS.index("RightHandMiddle2"), G1_INSPIREHANDS_LINKS.index("R_middle_intermediate"), 3],
    
    [SG_LINKS.index("LeftUpLeg"), G1_INSPIREHANDS_LINKS.index("left_hip_pitch_link"), 3],
    [SG_LINKS.index("RightUpLeg"), G1_INSPIREHANDS_LINKS.index("right_hip_pitch_link"), 3],
    [SG_LINKS.index("LeftLeg"), G1_INSPIREHANDS_LINKS.index("left_knee_link"), 3],
    [SG_LINKS.index("RightLeg"), G1_INSPIREHANDS_LINKS.index("right_knee_link"), 3],
    [SG_LINKS.index("LeftFoot"), G1_INSPIREHANDS_LINKS.index("left_ankle_roll_link"), 3],
    [SG_LINKS.index("RightFoot"), G1_INSPIREHANDS_LINKS.index("right_ankle_roll_link"), 3],
    
]


SMPL_G1_CORRESPONDENCE = [
    [SMPL_LINKS.index("Pelvis"), G1_LINKS.index("pelvis"), 1],
    [SMPL_LINKS.index("Head"), G1_LINKS.index("mid360_link"), 1],

    [SMPL_LINKS.index("L_Shoulder"), G1_LINKS.index("left_shoulder_roll_link"), 3],
    [SMPL_LINKS.index("L_Elbow"), G1_LINKS.index("left_elbow_link"), 3],
    [SMPL_LINKS.index("L_Wrist"), G1_LINKS.index("left_rubber_hand"), 3],
    # [SMPL_LINKS.index("LeftHandMiddle3"), G1_LINKS.index("left_mid_finger_link"), 3],

    
    [SMPL_LINKS.index("R_Shoulder"), G1_LINKS.index("right_shoulder_roll_link"), 3],
    [SMPL_LINKS.index("R_Elbow"), G1_LINKS.index("right_elbow_link"), 3],
    [SMPL_LINKS.index("R_Wrist"), G1_LINKS.index("right_rubber_hand"), 3],
    # [SMPL_LINKS.index("RightHandMiddle3"), G1_LINKS.index("right_mid_finger_link"), 3],

    ### used for standing straight
    [SMPL_LINKS.index("L_Ankle"), G1_LINKS.index("left_ankle_roll_link"), 1],
    [SMPL_LINKS.index("R_Ankle"), G1_LINKS.index("right_ankle_roll_link"), 1],

]


SMPL_G1_FULLBODY_CORRESPONDENCE = [
    [SMPL_LINKS.index("Pelvis"), G1_LINKS.index("pelvis"), 1],
    [SMPL_LINKS.index("L_Hip"), G1_LINKS.index("left_hip_pitch_link"), 1],
    [SMPL_LINKS.index("R_Hip"), G1_LINKS.index("right_hip_pitch_link"), 1],
    # [SMPL_LINKS.index("Spine1"), G1_LINKS.index(""), 1],
    # [SMPL_LINKS.index("Spine2"), G1_LINKS.index(""), 1],
    # [SMPL_LINKS.index("Spine3"), G1_LINKS.index(""), 1],  
    [SMPL_LINKS.index("L_Knee"), G1_LINKS.index("left_knee_link"), 1],
    [SMPL_LINKS.index("R_Knee"), G1_LINKS.index("right_knee_link"), 1],
    [SMPL_LINKS.index("L_Ankle"), G1_LINKS.index("left_ankle_roll_link"), 3],
    [SMPL_LINKS.index("R_Ankle"), G1_LINKS.index("right_ankle_roll_link"), 3],
    #[SMPL_LINKS.index("L_Foot"), G1_LINKS.index(""), 1],
    #[SMPL_LINKS.index("R_Foot"), G1_LINKS.index(""), 1],
    # [SMPL_LINKS.index("Neck"), G1_LINKS.index(""), 1],
    [SMPL_LINKS.index("Head"), G1_LINKS.index("mid360_link"), 1],

    # [SMPL_LINKS.index("L_Collar"), G1_LINKS.index(""), 3],
    # [SMPL_LINKS.index("R_Collar"), G1_LINKS.index(""), 3],
    [SMPL_LINKS.index("L_Shoulder"), G1_LINKS.index("left_shoulder_roll_link"), 3],
    [SMPL_LINKS.index("R_Shoulder"), G1_LINKS.index("right_shoulder_roll_link"), 3],
    [SMPL_LINKS.index("L_Elbow"), G1_LINKS.index("left_elbow_link"), 3],
    [SMPL_LINKS.index("R_Elbow"), G1_LINKS.index("right_elbow_link"), 3],
    [SMPL_LINKS.index("L_Wrist"), G1_LINKS.index("left_rubber_hand"), 3],
    [SMPL_LINKS.index("R_Wrist"), G1_LINKS.index("right_rubber_hand"), 3],

]



# Add the links for dex_retargeting
BBDB_LEFT_HAND_LINK = {
    "base_link" : BBDB_LINKS.index("LeftHand"),
    "tip_link" : [ BBDB_LINKS.index('LeftHandThumb4_end'), BBDB_LINKS.index('LeftHandIndex4_end'),
                  BBDB_LINKS.index('LeftHandMiddle4_end'), BBDB_LINKS.index('LeftHandRing4_end')
                  , BBDB_LINKS.index('LeftHandPinky4_end'), 
                  ]
}

BBDB_RIGHT_HAND_LINK = {
    "base_link" : BBDB_LINKS.index("RightHand"),
    "tip_link" : [BBDB_LINKS.index('RightHandThumb4_end'),BBDB_LINKS.index('RightHandIndex4_end'), 
                  BBDB_LINKS.index('RightHandMiddle4_end'),BBDB_LINKS.index('RightHandRing4_end'),
                  BBDB_LINKS.index('RightHandPinky4_end')
                  ]
}

BEAT_LEFT_HAND_LINK = {
    "base_link" : BEAT_LINKS.index("LeftHand"),
    "tip_link" : [ BEAT_LINKS.index('LeftHandThumb3'), BEAT_LINKS.index('LeftHandIndex3'),
                  BEAT_LINKS.index('LeftHandMiddle3'), BEAT_LINKS.index('LeftHandRing3')
                  , BEAT_LINKS.index('LeftHandPinky3'), 
                  ]
}

BEAT_RIGHT_HAND_LINK = {
    "base_link" : BEAT_LINKS.index("RightHand"),
    "tip_link" : [BEAT_LINKS.index('RightHandThumb3'),BEAT_LINKS.index('RightHandIndex3'), 
                  BEAT_LINKS.index('RightHandMiddle3'),BEAT_LINKS.index('RightHandRing3'),
                  BEAT_LINKS.index('RightHandPinky3')
                  ]
}


MOTION_CAPTURE_LEFT_HAND_LINK = {
    "base_link" : MOTION_CAPTURE_LINKS.index("LeftHand"),
    "tip_link" : [ MOTION_CAPTURE_LINKS.index('LeftHandThumb3'), MOTION_CAPTURE_LINKS.index('LeftHandIndex3'),
                  MOTION_CAPTURE_LINKS.index('LeftHandMiddle3'), MOTION_CAPTURE_LINKS.index('LeftHandRing3')
                  , MOTION_CAPTURE_LINKS.index('LeftHandPinky3'), 
                  ]
}

MOTION_CAPTURE_RIGHT_HAND_LINK = {
    "base_link" : MOTION_CAPTURE_LINKS.index("RightHand"),
    "tip_link" : [MOTION_CAPTURE_LINKS.index('RightHandThumb3'),MOTION_CAPTURE_LINKS.index('RightHandIndex3'), 
                  MOTION_CAPTURE_LINKS.index('RightHandMiddle3'),MOTION_CAPTURE_LINKS.index('RightHandRing3'),
                  MOTION_CAPTURE_LINKS.index('RightHandPinky3')
                  ]
}


SG_LEFT_HAND_LINK = {
    "base_link" : SG_LINKS.index("LeftHand"),
    "tip_link" : [ SG_LINKS.index('LeftHandThumb3'), SG_LINKS.index('LeftHandIndex3'),
                  SG_LINKS.index('LeftHandMiddle3'), SG_LINKS.index('LeftHandRing3')
                  , SG_LINKS.index('LeftHandPinky3'), 
                  ]
}

SG_RIGHT_HAND_LINK = {
    "base_link" : SG_LINKS.index("RightHand"),
    "tip_link" : [SG_LINKS.index('RightHandThumb3'),SG_LINKS.index('RightHandIndex3'), 
                  SG_LINKS.index('RightHandMiddle3'),SG_LINKS.index('RightHandRing3'),
                  SG_LINKS.index('RightHandPinky3')
                  ]
}



### 2025.05.14
# Full self-collision config for G1 robot

G1_INSPIREHANDS_COLLISION_CAPSULE = {
    ### name: link1, link2, radius
    # left leg collision group
    "group1":{
    "left_thigh": [G1_INSPIREHANDS_LINKS.index("left_hip_roll_link"), G1_INSPIREHANDS_LINKS.index("left_knee_link"), 0.12],
    "left_lowerleg": [G1_INSPIREHANDS_LINKS.index("left_knee_link"), G1_INSPIREHANDS_LINKS.index("left_ankle_roll_link"), 0.12]},
    
    # right leg collision group
    "group2":{
    "right_thigh": [G1_INSPIREHANDS_LINKS.index("right_hip_roll_link"), G1_INSPIREHANDS_LINKS.index("right_knee_link"), 0.10],
    "right_lower_leg": [G1_INSPIREHANDS_LINKS.index("right_knee_link"), G1_INSPIREHANDS_LINKS.index("right_ankle_roll_link"), 0.10]},
    
    # left arm collision group
    "group3":{
    "left_upperarm":[G1_INSPIREHANDS_LINKS.index("left_shoulder_roll_link"), G1_INSPIREHANDS_LINKS.index("left_elbow_link"), 0.08],
    "left_lowerarm":[G1_INSPIREHANDS_LINKS.index("left_elbow_link"), G1_INSPIREHANDS_LINKS.index("left_wrist_pitch_link"), 0.08],
    "left_wrist_2":[G1_INSPIREHANDS_LINKS.index("left_wrist_pitch_link"), G1_INSPIREHANDS_LINKS.index("left_wrist_yaw_link"), 0.06],
    "left_wrist_3":[G1_INSPIREHANDS_LINKS.index("left_wrist_yaw_link"), G1_INSPIREHANDS_LINKS.index("L_hand_base_link"), 0.06],
    "left_thumb1":[G1_INSPIREHANDS_LINKS.index("L_thumb_proximal"), G1_INSPIREHANDS_LINKS.index("L_thumb_intermediate"), 0.032*2],
    "left_thumb2":[G1_INSPIREHANDS_LINKS.index("L_thumb_intermediate"), G1_INSPIREHANDS_LINKS.index("L_thumb_distal"), 0.024*2],
    "left_thumb3":[G1_INSPIREHANDS_LINKS.index("L_thumb_distal"), G1_INSPIREHANDS_LINKS.index("L_thumb_tip"), 0.020*2],
    "left_index1":[G1_INSPIREHANDS_LINKS.index("L_index_proximal"), G1_INSPIREHANDS_LINKS.index("L_index_intermediate"), 0.015*2],
    "left_index2":[G1_INSPIREHANDS_LINKS.index("L_index_intermediate"), G1_INSPIREHANDS_LINKS.index("L_index_tip"), 0.012*2],
    "left_middle1":[G1_INSPIREHANDS_LINKS.index("L_middle_proximal"), G1_INSPIREHANDS_LINKS.index("L_middle_intermediate"), 0.015*2],
    "left_middle2":[G1_INSPIREHANDS_LINKS.index("L_middle_intermediate"), G1_INSPIREHANDS_LINKS.index("L_middle_tip"), 0.012*2],
    "left_ring1":[G1_INSPIREHANDS_LINKS.index("L_ring_proximal"), G1_INSPIREHANDS_LINKS.index("L_ring_intermediate"), 0.015*2],
    "left_ring2":[G1_INSPIREHANDS_LINKS.index("L_ring_intermediate"), G1_INSPIREHANDS_LINKS.index("L_ring_tip"), 0.012*2],
    "left_pinky1":[G1_INSPIREHANDS_LINKS.index("L_pinky_proximal"), G1_INSPIREHANDS_LINKS.index("L_pinky_intermediate"), 0.015*2],
    "left_pinky2":[G1_INSPIREHANDS_LINKS.index("L_pinky_intermediate"), G1_INSPIREHANDS_LINKS.index("L_pinky_tip"), 0.012*2],},
    
    # right arm collision group
    "group4":{
    "right_upperarm":[G1_INSPIREHANDS_LINKS.index("right_shoulder_roll_link"), G1_INSPIREHANDS_LINKS.index("right_elbow_link"), 0.08],
    "right_lowerarm":[G1_INSPIREHANDS_LINKS.index("right_elbow_link"), G1_INSPIREHANDS_LINKS.index("right_wrist_pitch_link"), 0.08],
    "right_wrist_2":[G1_INSPIREHANDS_LINKS.index("right_wrist_pitch_link"), G1_INSPIREHANDS_LINKS.index("right_wrist_yaw_link"), 0.06],
    "right_wrist_3":[G1_INSPIREHANDS_LINKS.index("right_wrist_yaw_link"), G1_INSPIREHANDS_LINKS.index("R_hand_base_link"), 0.06],
    "right_thumb1":[G1_INSPIREHANDS_LINKS.index("R_thumb_proximal"), G1_INSPIREHANDS_LINKS.index("R_thumb_intermediate"), 0.032*2],
    "right_thumb2":[G1_INSPIREHANDS_LINKS.index("R_thumb_intermediate"), G1_INSPIREHANDS_LINKS.index("R_thumb_distal"), 0.024*2],
    "right_thumb3":[G1_INSPIREHANDS_LINKS.index("R_thumb_distal"), G1_INSPIREHANDS_LINKS.index("R_thumb_tip"), 0.020*2],
    "right_index1":[G1_INSPIREHANDS_LINKS.index("R_index_proximal"), G1_INSPIREHANDS_LINKS.index("R_index_intermediate"), 0.015*2],
    "right_index2":[G1_INSPIREHANDS_LINKS.index("R_index_intermediate"), G1_INSPIREHANDS_LINKS.index("R_index_tip"), 0.012*2],
    "right_middle1":[G1_INSPIREHANDS_LINKS.index("R_middle_proximal"), G1_INSPIREHANDS_LINKS.index("R_middle_intermediate"), 0.015*2],
    "right_middle2":[G1_INSPIREHANDS_LINKS.index("R_middle_intermediate"), G1_INSPIREHANDS_LINKS.index("R_middle_tip"), 0.012*2],
    "right_ring1":[G1_INSPIREHANDS_LINKS.index("R_ring_proximal"), G1_INSPIREHANDS_LINKS.index("R_ring_intermediate"), 0.015*2],
    "right_ring2":[G1_INSPIREHANDS_LINKS.index("R_ring_intermediate"), G1_INSPIREHANDS_LINKS.index("R_ring_tip"), 0.012*2],
    "right_pinky1":[G1_INSPIREHANDS_LINKS.index("R_pinky_proximal"), G1_INSPIREHANDS_LINKS.index("R_pinky_intermediate"), 0.015*2],
    "right_pinky2":[G1_INSPIREHANDS_LINKS.index("R_pinky_intermediate"), G1_INSPIREHANDS_LINKS.index("R_pinky_tip"), 0.012*2],},


    
   
}

G1_INSPIREHANDS_COLLISION_CUBOID = {
    "torso":[G1_INSPIREHANDS_LINKS.index("waist_support_link"),
             (-0.07,0.07), # x range
             (-0.105,0.105), # y range
             (0,0.33)], # z range
    "pelvis":[G1_INSPIREHANDS_LINKS.index("pelvis"),
             (-0.06,0.06), # x range
             (-0.065,0.065), # y range
             (-0.15,0)], # z range
    "Lhand":[G1_INSPIREHANDS_LINKS.index("L_hand_base_link"),
             (-0.03,0.03), # x range
             (-0.13,0), # y range
             (-0.04,0.04)], # z range
    "Rhand":[G1_INSPIREHANDS_LINKS.index("R_hand_base_link"),
             (-0.03,0.03), # x range
             (-0.13,0), # y range
             (-0.04,0.04)], # z range
}

G1_INSPIREHANDS_ALLOWED_COLLISION_CFG = {
    "seg2seg":[
        ["group1","group2"],
        ["group1","group3"],
        ["group1","group4"],
        ["group2","group3"],
        ["group2","group4"],
        ["group3","group4"],
    ],
    "seg2cuboid":[
        ["group3","torso"],
        ["group3","pelvis"],
        ["group3","Rhand"],
        ["group4","torso"],
        ["group4","pelvis"],
        ["group4","Lhand"],
    ]
    
}

G1_INSPIREHANDS_COLLISION = [
    ["left_thigh", "left_hand"],
    ["right_thigh", "right_hand"],
    ["left_hand", "right_hand"],
]



