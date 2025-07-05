import pybullet as p
import time
from HRI_retarget import DATA_ROOT
import os

# 连接物理引擎
physicsClient = p.connect(p.GUI)  # 使用 GUI 模式
# p.setGravity(0, 0, -9.81)  # 设置重力
# p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1) # collision


### galbot charlie urdf
urdf_rel_path = "resources/robots/g1_inspirehands/G1_inspire_hands.urdf"
robotId = p.loadURDF(os.path.join(DATA_ROOT,urdf_rel_path), [0, 0, 0], [0, 0, 0, 1])



# 获取关节信息
num_joints = p.getNumJoints(robotId)
joint_indices = range(num_joints)
joint_names = [p.getJointInfo(robotId, i)[1].decode("utf-8") for i in joint_indices]
print("Joint Names:", joint_names)

dofs = []

# 创建滑块控件
sliders = []
for i in joint_indices:
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode("utf-8")
    lower_limit = joint_info[8]  # 关节下限
    upper_limit = joint_info[9]  # 关节上限
    slider = p.addUserDebugParameter(joint_name, lower_limit, upper_limit, 0)  # 初始值为0
    sliders.append(slider)

    if joint_info[2] != p.JOINT_FIXED:  # 如果不是固定关节
        dofs.append(joint_name)

print("Dof Joints:", dofs)


# # 创建固定约束，将base链接固定在世界坐标系的原点
# constraint_id = p.createConstraint(
#     parentBodyUniqueId=robotId,
#     parentLinkIndex=-1,
#     childBodyUniqueId=-1,  # -1表示世界坐标系
#     childLinkIndex=-1,
#     jointType=p.JOINT_FIXED,  # 固定关节
#     jointAxis=[0, 0, 0],  # 固定关节不需要轴
#     parentFramePosition=[0, 0, 0],  # base链接的局部坐标系原点
#     childFramePosition=[0, 0, 0]  # 世界坐标系的原点
# )



# 主循环
try:
    while True:
        angles = []
        for i, slider in enumerate(sliders):
            target_angle = p.readUserDebugParameter(slider)
            angles.append(target_angle)
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint_indices[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle
            )

       

        # 遍历每个 link
        for link_index in range(num_joints):
            # 获取 link 的状态
            link_state = p.getLinkState(robotId, link_index)
            
            # 提取位置和朝向
            link_pos = link_state[0]  # 位置 (x, y, z)
            link_orn = link_state[1]  # 朝向 (四元数)
            
            # 将四元数转换为欧拉角（可选）
            euler_angles = p.getEulerFromQuaternion(link_orn)
            
            # 在 GUI 中显示位置和朝向
            text_pos = f"Link {joint_names[link_index]}\nPos: {link_pos}\nEuler: {euler_angles}"
            print(text_pos)
            


        
        p.stepSimulation()
except KeyboardInterrupt:
    pass

# 断开连接
p.disconnect()