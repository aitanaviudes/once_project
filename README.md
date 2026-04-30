# Setup and Pipeline Worflow

## Camera and Interbotix Arm Setup

run in a terminal:

`roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud` 

in a second terminal:

`roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s use_rviz:=true`

you can check the topics of both the arm and intelisense camera doing:

`rostopic list` 

<br>

## Relevant Scripts:

In a terminal, go to:
`interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos`

here you will see different files but the relevant ones are three:
- demo_collect_current_try_3.py
- vis_velocity_commands.py
- call_replay_live_with_saved_data.py
- inference.py

There is another script we will be using but this one is located in the docker image that has been built, the script can be found here: <br>
`1000_tasks/learning_thousand_tasks/deployments/deploy_mt3.py` <br><br>

<br> 


## Recording a Demo:

Inside the `python_demos` directory run the demonstration collector file to record a demonstration:

`python3 demo_collect_current_try_3.py` 

Example of how to do it: [https://drive.google.com/drive/u/0/home](https://drive.google.com/file/d/1_yAW-ArX_D4vTTU7kbD1qVi1sqcG0pif/view?usp=sharing)

Instructions:

Relevant controls during recording:

- `'o': Open gripper`
- `'c': Close gripper`
- `'s': Start recording demonstration`
- `'z': take photo`

- press `Ctrl + C` to stop the script

The following data will save within the collected_demos/session_[TIMESTAMP]/demo0000/ directory.

  ### Demonstration Data Structure (`demo0000`)

| File Name | Data Type | Dimensions / Size | Description |
| :--- | :--- | :--- | :--- |
| **demo_eef_twists.npy** | NumPy Array (`float64`) | (T, 7) | Time-series of EEF twists: [vx, vy, vz, wx, wy, wz, gripper] |
| **bottleneck_pose.npy** | NumPy Array (`float64`) | (4, 4) | The initial SE(3) transformation matrix of the end-effector |
| **task_name.txt** | Plain Text | N/A | The string name of the task (e.g., pick_up_cube) |
| **head_camera_ws_rgb.png** | Image (uint8) | (720, 1280, 3) | RGB workspace snapshot taken at the "Ready Position" |
| **head_camera_ws_depth_to_rgb.png** | Image (uint16) | (720, 1280) | Aligned depth map in millimeters (mm) |
| **head_camera_ws_segmap.npy** | NumPy Array (`bool`) | (720, 1280) | Boolean mask where True identifies the target object |
| **head_camera_rgb_intrinsic_matrix.npy** | NumPy Array (`float64`) | (3, 3) | The camera intrinsic matrix K |
| **eef_poses.npy** | NumPy Array (`float64`) | (T, 4, 4) | Full sequence of SE(3) matrices for the entire trajectory |
| **timestamps.npy** | NumPy Array (`float64`) | (T,) | Relative time in seconds for each recorded timestep |
| **metadata.pkl** | Python Pickle | Dictionary | Serialized metadata (robot model, joint names, rate, etc.) |

**Note:** `T` represents the total number of timesteps recorded.

<br><br>
## Demonstration Replay Pipeline

Whenever you want to replay a demo, you must stop the current roslaunch of the robot arm:
`roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s use_rviz:=true`
and launch the following:
- replay in simulation: `roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch robot_model:=wx250s use_gazebo:=true dof:=6 use_python_interface:=false gui:=false`
- replay in reality: `roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch robot_model:=wx250s use_actual:=true dof:=6 use_python_interface:=false gui:=true`
  
<br>

If you want to replay a previously recorded demonstration, run the `vis_velocity_commands.py` script in the `python_demos` directory.
(Inside you must update the path of the session you wish to replay).


## Updating Inference and Demonstration Files in Docker Image
If everything looks as expected, then we shouold run the bash script `update_demo2.sh` inside `collected_demos`: <br><br>
`./update_demo.sh -s session....` <br><br>
What this file does is basically update the demonstration `pick_up_shoe` in `1000_tasks/learning_thousand_tasks/assets/demonstrations/pick_up_shoe` and also update the files inside `1000_tasks/learning_thousand_tasks/assets/inference_example`.

If you wish to check the pipeline using a different inference example, after running `update_demo2.sh` also run `update_inference_only2.sh` including in the session flag (-s) a different session. Now you have a different recording between the demostration and inference example.

<br>
## Running inference.py

inside `/python_demos` run `python3 inference.py` and follow the instructions

## Camera Callibraion

Run camera and robot roslaunchs as usual:

Then run:
`roslaunch easy_handeye calibrate_my_robot.launch`

### The Physical Calibration
Open a new terminal and run `rqt_image_view. Select the /tag_detections_image topic. You must see the orange "0" bounding box on your tag before taking any samples.

Go to the `rqt_easy_handeye` GUI window.

Use the RViz GUI to toggle motor torque OFF, physically move the arm around to take samples by clicking the 'Take Sample' button in the GUI.

Repeat for at least 15 different angles (varying translation, tilt, and wrist rotation).

Click Compute, then click Save. This saves the data to ~/.ros/easy_handeye/interbotix_calibration_eye_on_hand.yaml.


# TODO
 
1. Try tests with original camera extrinsics matrix:`T_WC_head.npy` and with the callibrated one: `T_WC_head.npy.callibrated`, these can be found in `/home/aitana_viudes/1000_tasks/learning_thousand_tasks/assets` - DONE
2. Improve / check the depth functions: - DONE
   `_to_uint16_depth_mm()` in: `depth_image = self._to_uint16_depth_mm(depth_image)`
   and
   `_refine_object_depth()` in: `depth_image = self._refine_object_depth(depth_image, segmap)`
   
3. Make tests of how real life displacements affect pointcloud. This can be done by recording demonstrations using the `demo_collect_current_try_2.py` script, then updating demo and inference example: (running `update_demo2.sh` with -s flag of one recorded demo and then `update_inference_only2.sh` with -s flag with another different recorded demo) See how the cube's positions differ. - DONE
4. Try changing the value in `T_WE_copy[:3, 3] += T_WE_copy[:3, :3] @ np.array([0, 0, 0.11])`: - DONE
   <img width="877" height="361" alt="image" src="https://github.com/user-attachments/assets/2fad3f1c-491f-4ea5-b79e-51f48898c1ed" />
5. Record video with 1 demonstran and 2 inferences , show the fotos if MT3 - DONE
6. Object orientation - Done
7. Inference pipeline plug and play - DONE
8. Put MT3 working docker in github
9. improve segmentation, use langsam with necessary python version
10. MT3 in GPU and see inference times in the end-to-end use case workflow
11. Go down and adapt MT3 with Kinova arm to pick and place the plate


interbotix:
1. bottleneck pose in demos has to be recorded from different positions (a bit more above)
2. test orientations generalisation (if doesn't work, record more than 1 per object)
3. test camera, different positions

    -> record bottleneck pose a bit more above that current recordings to see if performance improves

# Checked...

- when same demo used for inference example and demo example and using callibrated matrix it works perfect
- sometimes go to bottleneck pose breaks because it tries to take a path that would go through the table, this then leads to velocity commands starting at wrong position and everything breaks. But this is an easy fix (just make sure bottleneck pose is reached with a sensible path (not going under the table) and if not, plan it again).
intrinsics matrices of demonstrations are the same:

<img width="789" height="183" alt="image" src="https://github.com/user-attachments/assets/9f127af5-e14d-434f-8b0b-f1433b1514b2" />


