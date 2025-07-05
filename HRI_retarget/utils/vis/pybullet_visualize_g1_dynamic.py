### usage: visualize galbot and bvh at the same time
### specific to /data/SeG_dataset results
### python pybullet_visualize_galbot_dynamic.py data/SeG_dataset/galbot_motion/HAND_FAN-3.pickle
### todo: some variables have wrong name
###    bug! cannot show bvh !too laggy
###    
import pybullet as p
import time
import pickle 
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from HRI_retarget import DATA_ROOT
np.set_printoptions(suppress=True)

if len(sys.argv) != 2:
    print('Call the function with the motion file')
    filename = os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle")

else:
    filename = sys.argv[1]

with open(filename, "rb") as file:
    
    data_dict = pickle.load(file)
    joint_global_pos = data_dict["angles"]
    robot_name = data_dict["robot_name"]
    # fps = data_dict["fps"]
    fps = 120
    



# 连接物理引擎
physicsClient = p.connect(p.GUI)  # 使用 GUI 模式
p.setGravity(0, 0, -9.81)  # 设置重力
# p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1) # collision

### galbot charlie urdf

match robot_name:
    case "g1_15":
        urdf_rel_path = "resources/robots/g1_asap/g1_29dof_anneal_15dof.urdf"
    case "g1_inspirehands":
        urdf_rel_path = "resources/robots/g1_inspirehands/G1_inspire_hands.urdf"
robotId = p.loadURDF(os.path.join(DATA_ROOT,urdf_rel_path), [0, 0, 0], [0, 0, 0, 1])


# 创建固定约束，将base链接固定在世界坐标系的原点
constraint_id = p.createConstraint(
    parentBodyUniqueId=robotId,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,  # -1表示世界坐标系
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,  # 固定关节
    jointAxis=[0, 0, 0],  # 固定关节不需要轴
    parentFramePosition=[0, 0, 0],  # base链接的局部坐标系原点
    childFramePosition=[0, 0, 0]  # 世界坐标系的原点
)

# 获取关节信息
num_joints = p.getNumJoints(robotId)
joint_indices = range(num_joints)
joint_names = [p.getJointInfo(robotId, i)[1].decode("utf-8") for i in joint_indices]
print("Joint Names:", joint_names)





# 创建滑块控件
num_frames = joint_global_pos.shape[0]
frame_slider = p.addUserDebugParameter("frame_id",  0, num_frames-1, 0)

sliders = []
for i in joint_indices:
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode("utf-8")
    lower_limit = joint_info[8]  # 关节下限
    upper_limit = joint_info[9]  # 关节上限
    slider = p.addUserDebugParameter(joint_name, lower_limit, upper_limit, 0)  # 初始/值为0
    sliders.append(slider)

controllable_joints = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    joint_type = joint_info[2]  # 关节类型
    joint_name = joint_info[1].decode('utf-8')  # 关节名称

    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        controllable_joints.append((i, joint_name))
        print(f"Controllable Joint {i}: {joint_name}")
# fig = plt.figure()



init_angles = joint_global_pos[0]

for j, joint in enumerate(controllable_joints):
    i, name = controllable_joints[j]
    p.resetJointState(
        bodyUniqueId=robotId,
        jointIndex=joint_indices[i],
        targetValue=init_angles[j],
        targetVelocity=0
    )

# while True:
p.stepSimulation()
time.sleep(0.05)


frame_id = 0
try:
    while True:
        frame_id += 1
        if frame_id >= num_frames:
            # time.sleep(1)
            frame_id -= num_frames
            init_angles = joint_global_pos[0]

            for j, joint in enumerate(controllable_joints):

                i, name = controllable_joints[j]
                p.resetJointState(
                    bodyUniqueId=robotId,
                    jointIndex=joint_indices[i],
                    targetValue=init_angles[j],
                    targetVelocity=0
                )
            p.stepSimulation()
            
        target_angles = joint_global_pos[int(frame_id)]
        print(frame_id)

        joint_angle_now = []
        for j, joint in enumerate(controllable_joints):
            i, name = controllable_joints[j]
            joint_angle_now.append(p.getJointState(robotId, i)[0])

        print(np.stack([ np.array(range(len(target_angles))), np.array(joint_angle_now), np.array(target_angles),], axis=1))
        print("#" *50)



        ### manually fix the bottom part


        for j, joint in enumerate(controllable_joints):
            i, name = controllable_joints[j]
            print(j, name)
            if j < 3:
                continue
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint_indices[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angles[j]
            )

        time.sleep(1 / fps)

        p.stepSimulation()
except KeyboardInterrupt:
    pass

# 断开连接
p.disconnect()