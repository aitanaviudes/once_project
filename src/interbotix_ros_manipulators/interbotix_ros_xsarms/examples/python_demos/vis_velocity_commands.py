# roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch robot_model:=wx250s use_gazebo:=true dof:=6 use_python_interface:=false gui:=false
# roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch robot_model:=wx250s use_actual:=true dof:=6 use_python_interface:=false gui:=false

import sys
import rospy
import copy
import numpy as np
from geometry_msgs.msg import Pose
import moveit_commander
from tf.transformations import quaternion_from_matrix
import moveit_msgs.msg
import trajectory_msgs.msg
import tf.transformations as tf_trans

# NEW IMPORTS FOR VELOCITY CONTROL
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

class MoveGroupPythonInterfaceTutorial(object):
    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

        robot_ns = "/wx250s"
        robot_desc = robot_ns + "/robot_description"

        self.robot = moveit_commander.RobotCommander(robot_description=robot_desc, ns=robot_ns)
        self.scene = moveit_commander.PlanningSceneInterface(ns=robot_ns)
        
        # Arm Group
        self.move_group = moveit_commander.MoveGroupCommander("interbotix_arm", robot_description=robot_desc, ns=robot_ns)
        
        # Gripper Group
        self.gripper_group = moveit_commander.MoveGroupCommander("interbotix_gripper", robot_description=robot_desc, ns=robot_ns)

        self.display_trajectory_publisher = rospy.Publisher(
            robot_ns + "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        self.robot_ns = robot_ns

    def load_eef(self):
        #return np.load("/home/aitana_viudes/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos/collected_demos/session_20260407_212059/demo_0000/eef_poses.npy")
        return np.load("/home/aitana_viudes/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos/collected_demos/session_20260414_201103/demo_0000/eef_poses.npy") # 4x4 homogeneous transformation matrix of eef

    def load_twists(self):
        #return np.load("/home/aitana_viudes/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos/collected_demos/session_20260407_212059/demo_0000/demo_eef_twists.npy")
        return np.load("/home/aitana_viudes/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos/collected_demos/session_20260414_201103/demo_0000/demo_eef_twists.npy") # Tx7 velocity commands of eef

    def mat_to_pose(self, T: np.ndarray) -> Pose:
        pose = Pose()
        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])
        q = quaternion_from_matrix(T)   
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
        return pose

    def open_gripper(self):
        print("============ Opening Gripper...")
        self.gripper_group.set_named_target("Open")
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()

    def close_gripper(self):
        print("============ Closing Gripper...")
        self.gripper_group.set_named_target("Closed")
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()

    def go_to_pose_goal(self, pose_goal):
        self.move_group.set_pose_reference_frame(self.robot_ns + "/base_link")
        self.move_group.set_planning_time(15.0)
        self.move_group.set_goal_position_tolerance(0.01)
        self.move_group.set_goal_orientation_tolerance(0.05)
        self.move_group.set_pose_target(pose_goal)
        
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def switch_controllers(self, start_controllers, stop_controllers):
        service_name = self.robot_ns + '/controller_manager/switch_controller'
        print(f"============ Waiting for Controller Manager Service: {service_name}")
        rospy.wait_for_service(service_name)
        
        try:
            switch_srv = rospy.ServiceProxy(service_name, SwitchController)
            req = SwitchControllerRequest()
            req.start_controllers = start_controllers
            req.stop_controllers = stop_controllers
            req.strictness = SwitchControllerRequest.STRICT
            
            response = switch_srv(req)
            if response.ok:
                print(f"============ Successfully switched controllers!")
            else:
                print(f"============ FAILED to switch controllers.")
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

    
    def execute_true_velocity_phase(self, twists, hz=30.0):
        print("============ Starting TRUE Velocity Interaction Phase (Gazebo Streaming)")
        
        # We publish to the existing arm_controller instead of a velocity topic
        traj_pub = rospy.Publisher(
            self.robot_ns + "/arm_controller/command", 
            trajectory_msgs.msg.JointTrajectory, 
            queue_size=10
        )
        
        rate = rospy.Rate(hz)
        dt = 1.0 / hz

        # The controller expects the joints in this exact order
        joint_names = [
            "waist", "shoulder", "elbow", 
            "forearm_roll", "wrist_angle", "wrist_rotate"
        ]

        # Keep track of the joint positions as we integrate them
        current_joints = np.array(self.move_group.get_current_joint_values())

        gripper_closed = False

        for twist in twists:
            # 1. Check and actuate the gripper BEFORE calculating the movement!
            target_gripper_state = float(twist[-1])
            
            if target_gripper_state == 1.0 and not gripper_closed:
                self.close_gripper()
                gripper_closed = True
                # MoveIt pauses the thread to close the gripper, so we reset the 
                # current_joints to exactly where the arm is right now to prevent a jump
                current_joints = np.array(self.move_group.get_current_joint_values())
                
            elif target_gripper_state == 0.0 and gripper_closed:
                self.open_gripper()
                gripper_closed = False
                current_joints = np.array(self.move_group.get_current_joint_values())

            # 2. MT3 Twists are in the End-Effector Frame!
            v_linear_ee = np.array([twist[0], twist[1], twist[2]])
            v_angular_ee = np.array([twist[3], twist[4], twist[5]])
    
            # 2. Get the current orientation of the gripper, which is in world frame (eef with respect to base)
            current_pose = self.move_group.get_current_pose().pose # returns it relative to world frame
            q = [
                current_pose.orientation.x, 
                current_pose.orientation.y, 
                current_pose.orientation.z, 
                current_pose.orientation.w
            ]
            
            # Convert quaternion to a 3x3 Rotation Matrix
            R = tf_trans.quaternion_matrix(q)[:3, :3] 

            # Rotate the Twists from the Hand frame into the Base frame
            v_linear_base = np.dot(R, v_linear_ee)
            v_angular_base = np.dot(R, v_angular_ee)
            
            # Recombine into the 6D target vector the Jacobian expects
            v_target_base = np.concatenate((v_linear_base, v_angular_base))

            # Get the Jacobian
            J = np.array(self.move_group.get_jacobian_matrix(current_joints.tolist()))

            # Inverse of Jacobian
            J_inv = np.linalg.pinv(J, rcond=1e-2) 

            # 7. Calculate final joint velocities
            q_dot = np.dot(J_inv, v_target_base)

            # 8. Integrate to find the next position
            current_joints = current_joints + (q_dot * dt)

            # 9. Send the command to Gazebo
            msg = trajectory_msgs.msg.JointTrajectory()
            msg.joint_names = joint_names
            
            point = trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = current_joints.tolist()
            point.velocities = q_dot.tolist()
            point.time_from_start = rospy.Duration(dt) 
            
            msg.points.append(point)
            traj_pub.publish(msg)

            rate.sleep()


def main():
    mg = MoveGroupPythonInterfaceTutorial()
    waypoints = [mg.mat_to_pose(T) for T in mg.load_eef()]
    twists = mg.load_twists()

    print("=== PHASE 0: RESET TO HOME ===")
    # "Home" or "Sleep" forces the arm to a known, untwisted state
    mg.move_group.set_named_target("Home") 
    mg.move_group.go(wait=True)

    # PHASE 1: ALIGNMENT (Position Control)
    print("=== PHASE 1: ALIGNMENT ===")
    bottleneck_pose = waypoints[0]
    mg.go_to_pose_goal(bottleneck_pose)

    # PHASE 3: INTERACTION (Velocity Control)
    print("=== PHASE 3: INTERACTION ===")
    # TODO: add a flag: sim or real. 
    mg.execute_true_velocity_phase(twists)

    print("============ Demonstration Execution Complete!")

if __name__ == "__main__":
    main()