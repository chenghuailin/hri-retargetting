import os

files = [
    "full.obj",
    "gripper_base.obj",
    "gripper_l1.obj",
    "gripper_l2.obj",
    "gripper_l3.obj",
    "gripper_l4.obj",
    "gripper_r1.obj",
    "gripper_r2.obj",
    "gripper_r3.obj",
    "gripper_r4.obj",
]
scale_factor = 1000.0


def scale_obj_file(file_path, new_file_path, scale_factor):
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(new_file_path, "w") as file:
        for line in lines:
            if line.startswith("v "):
                parts = line.split()
                x = float(parts[1]) / scale_factor
                y = float(parts[2]) / scale_factor
                z = float(parts[3]) / scale_factor
                file.write(f"v {-(x+0.0988)} {y} {z}\n")
            else:
                file.write(line)


for file_name in files:
    file_path = os.path.join(
        "/workspace/motion_planning/robot_models/galbot_one_charlie_10/meshes/gripper",
        file_name,
    )
    new_file_path = os.path.join(
        "/workspace/motion_planning/robot_models/galbot_one_charlie_10/meshes/gripper",
        "new_" + file_name,
    )
    scale_obj_file(file_path, new_file_path, scale_factor)
