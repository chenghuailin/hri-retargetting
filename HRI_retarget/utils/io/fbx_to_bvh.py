# https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/blender_rendering/utils/fbx2bvh.py

import bpy
import numpy as np
from os import listdir, path
from HRI_retarget import DATA_ROOT

def fbx2bvh(data_path, file):
    sourcepath = data_path+"/"+file
    bvh_path = data_path+"/"+file.split(".fbx")[0]+".bvh"

    bpy.ops.import_scene.fbx(filepath=sourcepath)

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]

    if  action.frame_range[1] > frame_end:
      frame_end = int(action.frame_range[1])
    if action.frame_range[0] < frame_start:
      frame_start = int(action.frame_range[0])

    frame_end = np.max([60, frame_end])
    bpy.ops.export_anim.bvh(filepath=bvh_path,
                            frame_start=frame_start,
                            frame_end=frame_end, root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])
    print(data_path+"/"+file+" processed.")

if __name__ == '__main__':
    data_path = path.join(DATA_ROOT, "data/motion/g1/motion_capture")


    files = sorted([f for f in listdir(data_path) if f.endswith(".fbx")])
    for file in files:
        print("Processing: ", file)
        fbx2bvh(path.join(data_path), file)