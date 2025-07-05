# URDFs:

| URDF File                                 | Description                                  | Collision Type | Leg Joints | Right Arm Joints | Left Arm Joints | Gripper Joints | Head Joints |
|-------------------------------------------|----------------------------------------------|----------------|------------|------------------|-----------------|----------------|-------------|
| `galbot_one_charlie_10.urdf`              | Original URDF file                           | Mesh           | √          | √                | √               | √              | ×           |
| `galbot_one_charlie_10_head.urdf`         | Head can move                                | Mesh           | √          | √                | √               | √              | √           |
| `galbot_one_charlie_10_gripper.urdf`      | Lock right arm                               | Sphere         | √          | ×                | √               | √              | ×           |
| `galbot_one_charlie_10_arm.urdf`          | Lock right arm and gripper                   | Sphere         | √          | ×                | √               | ×              | ×           |
| `galbot_lock_leg.urdf`                    | Lock right arm and leg                       | Sphere         | ×          | ×                | √               | √              | ×           |
| `galbot_one_charlie_10_26dof_sphere.urdf` | Head can move, Different right arm angle def | Sphere         | √          | √                | √               | √              | √           |
| `galbot_only_arm.urdf`                    | Only unlock left arm                         | Sphere         | ×          | ×                | √               | ×              | ×           |
| `galbot_right_arm.urdf`                   | Only unlock right arm                        | Sphere         | ×          | √                | ×               | ×              | ×           |