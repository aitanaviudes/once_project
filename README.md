# Setup and Pipeline Worflow

## Camera and Interbotix Arm Setup

run in a terminal:

`roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud` 

in a second terminal:

`roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s use_rviz:=true`

you can check the topics of both the arm and intelisense camera doing:

`rostopic list` 

<br>
In another terminal, for simplicity, launch another rviz window: <br><br>

`rviz` 

- In the "Global Options" panel on the left, change the Fixed Frame from `map` to `wx250s/base_link`

- Click the Add button at the bottom left, and select `RobotModel`. You should now see the 3D model of your WidowX arm.

- In the "Displays" panel on the left, find the field labeled "Description Topic" and change it to `/wx250s/robot_description`

- Click "Add" again, go to the "By topic" tab, and find /camera/depth/color/points. Select `PointCloud2`
  
<br>
Connect the Camera to the Wrist running in another terminal:<br><br>

`rosrun tf static_transform_publisher 0.05 0 0.04 0 0 0 wx250s/ee_arm_link camera_link 100` 
`rosrun tf2_ros static_transform_publisher -0.007199 0.035990 0.057238 -0.489602 0.480778 -0.507338 0.521295 wx250s/ee_arm_link camera_color_optical_frame`
<br><br>

## Scripts:

In a terminal, go to:
`interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos`

here you will see different files but the relevant ones are three:
- demo_collect_current_try_2.py
- vis_velocity_commands.py
- call_replay_live_with_saved_data.py

There is another script we will be using but this one is located in the docker image that has been built, the script can be found here: <br>
`1000_tasks/learning_thousand_tasks/deployments/deploy_mt3.py` <br><br>

## Demonstration Collection Pipeline:

### Code Architecture & Key Functions

The `DemonstrationCollectorV2` class manages the lifecycle of a robot recording session. Below are the primary functional blocks:

#### 1. Kinematics & State Estimation
* **`_compute_eef_twist`**: Uses the **Modern Robotics** library to compute the end-effector (EEF) body-frame twist. It calculates the Space Jacobian ($J_s$) from joint positions and converts the resulting spatial twist into the body frame ($V_b$).
* **`_extract_arm_state`**: Parses incoming ROS `JointState` messages to isolate the specific joint positions and velocities for the `wx250s` arm.

#### 2. Computer Vision & Segmentation
* **`_capture_workspace_camera_data`**: A blocking call that synchronizes and captures a single frame of RGB, Depth, and Camera Intrinsics via ROS topics.
* **`simple_orange_segmentation_with_depth`**: The primary perception pipeline. It combines **HSV color thresholding** with **depth-gating** to isolate the target object (orange cube) from the table surface.
* **`_refine_object_depth`**: A cleanup utility that uses a median filter on the segmented depth map to remove "holes" or outliers, ensuring a clean point cloud for later processing.

#### 3. Recording & Hardware Control
* **`start_recording` / `stop_recording`**: Manages the data buffers. Recording only begins once a "Ready Position" camera snapshot is successfully verified.
* **`enable_teaching_mode`**: Disables motor torques on the Interbotix arm, allowing for **kinesthetic teaching** (moving the arm by hand).
* **`record_step`**: The high-frequency loop function (running at the defined `record_rate`) that samples the current twist and appends it to the trajectory buffer.

#### 4. Data Serialization
* **`save_demonstrations`**: Handles the directory creation and converts Python lists into `.npy` (NumPy) and `.png` files. It enforces the `learning_thousand_tasks` naming convention required for training.

<br> 

Inside the `python_demos` directory run the demonstration collector file to record a demonstration:

`python3 demo_collect_current_try_2.py` 

Example of how to do it: [https://drive.google.com/drive/u/0/home](https://drive.google.com/file/d/1_yAW-ArX_D4vTTU7kbD1qVi1sqcG0pif/view?usp=sharing)

Instructions:

Controls during recording:

- `'o': Open gripper`
- `'c': Close gripper`
- `'s': Start recording demonstration`
- `'e': End recording demonstration`
- `'r': Go to ready position`
- `'t': Toggle teaching mode`
- `'p': Print current EEF pose`
- `'q': Quit and save demonstrations`

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

Stop the `roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s` process before moving on to pass our collected demonstration to the mt3 pipeline. We are going to check the the demonstration has been correctly recorded by replaying it in the rviz simulator (real life robot will not replay it): <br><br>
`roslaunch interbotix_xsarm_descriptions xsarm_description.launch robot_model:=wx200 use_joint_pub_gui:=true`
<br><br>
## Demonstration Replay Pipeline

Whenever you want to replay a demo, you must stop the current roslaunch of the robot arm and launch the following:
- replay in simulation: `roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch robot_model:=wx250s use_gazebo:=true dof:=6 use_python_interface:=false gui:=false`
- replay in reality: `roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch robot_model:=wx250s use_actual:=true dof:=6 use_python_interface:=false gui:=false`
  
<br>

To replay a previously recorded demonstration, run the `vis_velocity_commands.py` script in the `python_demos` directory.
(Inside you must update the path of the session you wish to replay).


## Updating Inference and Demonstration Files in Docker Image
If everything looks as expected, then we shouold run the bash script `update_demo2.sh` inside `collected_demos`: <br><br>
`./update_demo.sh -s session....` <br><br>
What this file does is basically update the demonstration `pick_up_shoe` in `1000_tasks/learning_thousand_tasks/assets/demonstrations/pick_up_shoe` and also update the files inside `1000_tasks/learning_thousand_tasks/assets/inference_example`.

If you wish to check the pipeline using a different inference example, after running `update_demo2.sh` also run `update_inference_only2.sh` including in the session flag (-s) a different session. Now you have a different recording between the demostration and inference example.

<br><br>
## Deploying MT3 Pipeline
Now we can execute `make deploy_mt3` (inside `1000_tasks/learning_thousand_tasks/`). This runs the docker image, it is the main entry point of the mt3 pipeline and it also executes the file `1000_tasks/learning_thousand_tasks/deployments/deploy_mt3.py` which we will also be working with. The file has been updated towards the end to inlcude the following:

    save_dir = Path('/workspace/saved_data')  # This is mounted to your host  
    save_dir.mkdir(parents=True, exist_ok=True)  
    
    np.save(save_dir / 'live_bottleneck_pose.npy', live_bottleneck_pose)  
    np.save(save_dir / 'end_effector_twists.npy', end_effector_twists)

This way, once the updated `live_bottleneck_pose` and `end_effector_twists` have been produced, we can save them to a directory in: `1000_tasks/learning_thousand_tasks/saved_data` we can access from outside the docker image.

Now we can run the last script: `python3 call_replay_live_with_saved_data.py` which will basically load saved MT3 data and replay it using the DemoReplayer. This script loads the live_bottleneck_pose and end_effector_twists saved by deploy_mt3.py and passes them to the ROS replay system. 
Make sure to execute this file with rviz simulator first (not real life) to see if the output of the mt3 pipeline looks correct and safe.

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
7. Inference pipeline plug and play
8. Put MT3 working docker in github
9. improve segmentation, use langsam with necessary python version
10. MT3 in GPU and see inference times in the end-to-end use case workflow
11. Go down and adapt MT3 with Kinova arm to pick and place the plate

    -> record bottleneck pose a bit more above that current recordings to see if performance improves

# Checked...

- when same demo used for inference example and demo example and using callibrated matrix it works perfect
- sometimes go to bottleneck pose breaks because it tries to take a path that would go through the table, this then leads to velocity commands starting at wrong position and everything breaks. But this is an easy fix (just make sure bottleneck pose is reached with a sensible path (not going under the table) and if not, plan it again).
intrinsics matrices of demonstrations are the same:

<img width="789" height="183" alt="image" src="https://github.com/user-attachments/assets/9f127af5-e14d-434f-8b0b-f1433b1514b2" />


