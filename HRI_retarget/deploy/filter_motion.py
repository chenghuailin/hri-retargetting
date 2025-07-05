import pickle 
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import os
from HRI_retarget import ROOT, DATA_ROOT


with open(os.path.join(DATA_ROOT,"motion/g1/SG/output.pickle"), "rb") as file:
    angles = pickle.load(file)

print(len(angles["angles"]))
# angles["angles"] = angles["angles"][100:500]
# angles["angles"] = gaussian_filter1d(angles["angles"], sigma=2)


### low-pass filter
cutoff_freq = 0.1  # 归一化截止频率（0~1）
b, a = signal.butter(4, cutoff_freq, 'low')  # 4阶Butterworth滤波器
# 应用滤波器
for idx in range(15):
    angles["angles"][:, idx] = signal.filtfilt(b, a, angles["angles"][:, idx])

print(angles["angles"].shape)

dof_names = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
]


x = np.arange(angles["angles"].shape[0])  # 或使用实际横坐标，如时间序列

# 绘制15条折线
plt.figure(figsize=(10, 6))
for i in range(angles["angles"].shape[1]):  # 遍历每一列
    plt.plot(x, angles["angles"][:, i], label=f'{dof_names[i]}')  # 用label添加图例

# 添加标签和标题
plt.xlabel('X-axis (e.g., Time)')
plt.ylabel('Y-axis (e.g., Value)')
plt.title('QPose')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在外侧
plt.grid(True)
plt.tight_layout()  # 防止图例遮挡
plt.show()




with open(os.path.join(ROOT,"output.pickle"), "wb") as file:
    pickle.dump(angles, file)

