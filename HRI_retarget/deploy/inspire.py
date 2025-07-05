import os 
import sys
sys.path.append('/home/galbot/workspace/g1_teleoperation/g1_teleoperation')

import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading
from pygame.time import Clock

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds

from controller.inspire.inspire_controller import Inspire_Controller

if __name__ == '__main__':
    ChannelFactoryInitialize(0)

    hand_ctrl = Inspire_Controller()
    clock = Clock()
    for frame in range(400):
        clock.tick(60)

        # get inspire data
        hand_state = hand_ctrl.get_qpos()
        if frame % 20 == 0:
            print(hand_state)

        # set inspire qpos
        if frame == 0:
            hand_qpos = np.zeros(12)
            hand_ctrl.set_qpos(hand_qpos)
        if frame == 100:
            hand_qpos = np.ones(12)
            hand_ctrl.set_qpos(hand_qpos)
        if frame == 200:
            hand_qpos = np.zeros(12)
            hand_ctrl.set_qpos(hand_qpos)
        if frame == 300:
            hand_qpos = np.ones(12)
            hand_ctrl.set_qpos(hand_qpos)