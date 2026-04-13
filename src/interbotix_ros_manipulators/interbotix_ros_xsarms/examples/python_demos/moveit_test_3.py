#!/usr/bin/env python3

import sys
import copy

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown

ROBOT_NAME = "wx250s"
MOVE_GROUP = "interbotix_arm"
FORWARD_METERS = 0.05


def detect_robot_mode(robot_name, timeout=2.0):

    real_service = "/{}/get_robot_info".format(robot_name)
    try:
        rospy.wait_for_service(real_service, timeout=timeout)
        return "real"
    except rospy.ROSException:
        pass

    sim_topics = {"/{}/joint_states".format(robot_name), "/joint_states"}
    try:
        published_topics = {name for name, _ in rospy.get_published_topics()}
    except rospy.ROSException:
        published_topics = set()

    if sim_topics & published_topics:
        return "simulation"

    return "unknown"


def make_move_group(group_name, robot_name):
    robot_description = "/{}/robot_description".format(robot_name)
    ns = "/{}".format(robot_name)

    try:
        return MoveGroupCommander(group_name, robot_description=robot_description, ns=ns)
    except TypeError:
        return MoveGroupCommander(group_name, robot_description=robot_description)

def T_to_pose(): # discrete poses along the path
    pose = Pose()
    pose.position.x, posep.position.y, pose.position.z = T[0,3], T[1,3], T[2,3]
    quaternion = quaternion_from_matrix(T)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion
    return pose

def ik(demo_dir):
    eef_poses_path = os.path.join(demo_dir, "eef_poses.npy")
    eef_poses = np.load(eef_poses_path)
    waypoints = [T_to_pose(T) for T in eef_poses]

    (plan, fraction) = move_group.compute_cartesian_path(
    waypoints, 0.01  # waypoints to follow  # eef_step
)


def main():
    roscpp_initialize(sys.argv)
    rospy.init_node("moveit_test_forward", anonymous=True)

    try:
        mode = detect_robot_mode(ROBOT_NAME)
        if mode == "real":
            rospy.loginfo("Detected mode: REAL ROBOT")
        elif mode == "simulation":
            rospy.loginfo("Detected mode: SIMULATION")
        else:
            rospy.logwarn("Detected mode: UNKNOWN (check your launch setup)")

        semantic_param = "/{}/robot_description_semantic".format(ROBOT_NAME)
        if not rospy.has_param(semantic_param):
            rospy.logerr("Missing MoveIt semantic param: %s", semantic_param)
            rospy.logerr("Launch MoveIt first, for simulation:")
            rospy.logerr(
                "roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                "robot_model:=%s robot_name:=%s use_fake:=true dof:=6",
                ROBOT_NAME,
                ROBOT_NAME,
            )
            return 1

        group = make_move_group(MOVE_GROUP, ROBOT_NAME)
        group.set_max_velocity_scaling_factor(0.1)
        group.set_max_acceleration_scaling_factor(0.1)
        group.set_start_state_to_current_state()

        rospy.loginfo("Planning frame: %s", group.get_planning_frame()) #
        rospy.loginfo("End-effector link: %s", group.get_end_effector_link())

        current_pose = group.get_current_pose().pose # prbably 4 x 4 matrix of inital pose,
        # what's the difference between that and 
        target_pose = copy.deepcopy(current_pose)
        target_pose.position.x += FORWARD_METERS

        rospy.loginfo("Moving EEF forward by %.3f m", FORWARD_METERS)
        group.set_pose_target(target_pose)
        success = group.go(wait=True)
        group.stop()
        group.clear_pose_targets()
        if not success:
            rospy.logerr("Move failed.")
            return 1

        rospy.loginfo("Done.")
        return 0
    finally:
        roscpp_shutdown()


if __name__ == "__main__":
    sys.exit(main())
