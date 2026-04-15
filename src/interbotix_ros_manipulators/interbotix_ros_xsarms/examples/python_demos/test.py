#!/usr/bin/env python3

import sys
import rospy
import numpy as np
from geometry_msgs.msg import Pose
import moveit_commander
from tf.transformations import quaternion_from_matrix

class RobotAligner(object):
    def __init__(self):
        super(RobotAligner, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("wx250s_align_to_matrix", anonymous=True)

        self.robot_ns = "/wx250s"
        robot_desc = self.robot_ns + "/robot_description"

        # Initialize MoveIt commanders
        self.robot = moveit_commander.RobotCommander(robot_description=robot_desc, ns=self.robot_ns)
        self.move_group = moveit_commander.MoveGroupCommander("interbotix_arm", robot_description=robot_desc, ns=self.robot_ns)
        
        # Configure tolerances for a precise bottleneck reach
        self.move_group.set_goal_position_tolerance(0.001)
        self.move_group.set_goal_orientation_tolerance(0.01)
        self.move_group.set_planning_time(10.0)

    def mat_to_pose(self, T: np.ndarray) -> Pose:
        """Converts a 4x4 numpy matrix to a geometry_msgs/Pose."""
        pose = Pose()
        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])
        
        # quaternion_from_matrix requires a full 4x4
        q = quaternion_from_matrix(T)   
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def go_to_matrix(self, matrix):
        pose_goal = self.mat_to_pose(matrix)
        
        print("============ Planning to Bottleneck Pose...")
        self.move_group.set_pose_target(pose_goal)
        
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        
        if success:
            print("============ Goal Reached Successfully!")
        else:
            print("============ Planning Failed. Check if the pose is within workspace.")

def main():
    # Your specific 4x4 Homogeneous Transformation Matrix
    bottleneck_matrix = np.array([
        [ 0.20851534, -0.73617387,  0.64387063,  0.56490663],
        [ 0.23439103,  0.67677093,  0.69788391, -0.38190134],
        [-0.94951682,  0.005398,    0.31366968,  0.04996001],
        [ 0.0,         0.0,         0.0,         1.0       ]
    ])

    aligner = RobotAligner()

    print("=== MOVING TO HOME FIRST ===")
    aligner.move_group.set_named_target("Home")
    aligner.move_group.go(wait=True)

    print("=== MOVING TO BOTTLENECK POSE ===")
    aligner.go_to_matrix(bottleneck_matrix)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass