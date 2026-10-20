# justina_description
This repository contains a ROS package with all the robot description files for the Justina robot, such as .xacro, .urdf, .dae, and textures.

## Robot Model
- **Real robot**: URDF description (`urdf/justina.urdf`) – connected to Webots via the **ROS bridge** for hardware‑in‑the‑loop testing.
- **Simulation**: Webots native PROTO files (same kinematic and sensor properties as the real robot).

To launch simulated version of the robot, use launch files located in webots_env folder that loads a low poly version of the robot, which can be used for faster rendering in Gazebo, reducing poly count from 334,489 to 12,372 polygons for all robot mesh.

## Requirements
- **Webots R2025a** (download from [Cyberbotics](https://www.cyberbotics.com/#download)), install via:
  ```bash
  sudo apt install ./webots_2025a_amd64.deb
  ```
- **Instal dependencies**. Change to directory **justina_description/webots_env** and run install.sh file:
  ```bash
  ./install.sh
  ```
------------------
Motor limits conversion to virtual robot 
------------------
lf_shoulder_joint	[-1.5842, 1.7914] → [3080, 880]
ls_shoulder_joint	[-1.5919, 1.5765] → [3085, 1020]
l_arm_twist_joint	[none, none] → [none, none]
l_elbow_joint		[-1.7683, 1.6993] → [3200, 940]* offset 8.9° positivo
l_wrist_joint		[none, none] → [none, none]
l_hand_joint

-----
rf_shoulder_joint	[-1.4461, 1.8681] → [2990, 830]
rs_shoulder_joint	[-1.5919, 1.5765] → [3085, 1020]
r_arm_twist_joint	[none, none] → [none, none]
r_elbow_joint		[-1.7683, 1.73] → [3200, 920] * puede ser igual
r_wrist_joint		[none, none] → [none, none]
r_hand_joint

------------------
