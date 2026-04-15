#!/usr/bin/env python3  

# roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch robot_model:=wx250s use_gazebo:=true dof:=6 use_python_interface:=false gui:=false
# roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch robot_model:=wx250s use_actual:=true dof:=6 use_python_interface:=false gui:=false

"""      
Script to load saved MT3 data and replay it using the DemoReplayer.      
This script loads the live_bottleneck_pose and end_effector_twists      
saved by deploy_mt3.py and passes them to the ROS replay system.      
"""      
      
import numpy as np      
import rospy      
from pathlib import Path        
#from inference_pipeline import VelocityTrajectoryReplayer

from vis_velocity_commands import MoveGroupPythonInterfaceTutorial

def replay_saved_trajectory(live_bottleneck_pose, end_effector_twists):
    replayer = MoveGroupPythonInterfaceTutorial()

    print("eef_twists HERE")
    print(end_effector_twists.shape)
    print("=== PHASE 0: RESET TO HOME ===")
    replayer.move_group.set_named_target("Home")
    replayer.move_group.go(wait=True)
    replayer.move_group.stop()

    print("=== PHASE 1: ALIGNMENT ===")
    bottleneck_pose = replayer.mat_to_pose(live_bottleneck_pose)
    replayer.go_to_pose_goal(bottleneck_pose)

    print("=== PHASE 3: INTERACTION ===")
    replayer.execute_true_velocity_phase(end_effector_twists)
    print("Trajectory execution complete.")    

def main():      
    """Load saved data and run replay."""      
    # REMOVE THIS LINE: rospy.init_node('mt3_replay_node')      
          
    # Path to saved data directory - UPDATED PATH    
    save_dir = Path('/home/aitana_viudes/1000_tasks/learning_thousand_tasks/saved_data')      
          
    try:      
        # Load the saved bottleneck pose and twists      
        print(f"Loading data from: {save_dir}")      
        live_bottleneck_pose = np.load(save_dir / 'live_bottleneck_pose.npy')      
        end_effector_twists = np.load(save_dir / 'end_effector_twists.npy')      
              
        print(f"Loaded live_bottleneck_pose: {live_bottleneck_pose.shape}")      
        print(f"Loaded end_effector_twists: {end_effector_twists.shape}")      
        
        replay_saved_trajectory(live_bottleneck_pose, end_effector_twists)  
              
    except FileNotFoundError as e:      
        print(f"Error: Could not find saved data files. {e}")      
        print("Please run deploy_mt3.py first to generate the saved data.")      
    except Exception as e:      
        print(f"Error during replay: {e}")      
        import traceback      
        traceback.print_exc()      
      
      
if __name__ == '__main__':      
    main()