#!/usr/bin/env python3

import sys
import copy

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown

ROBOT_NAME = "wx250s"
MOVE_GROUP = "interbotix_arm"
FORWARD_METERS = 0.05


def detect_robot_mode(robot_name, timeout=2.0):
    """
    Detect whether the robot is running in real hardware mode or simulation.

    Heuristic:
    - REAL: Interbotix xs_sdk service '/<robot_name>/get_robot_info' is available.
    - SIMULATION: no xs_sdk service, but joint states are being published.
    - UNKNOWN: neither condition matched.
    """
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
    """
    Build a MoveGroupCommander while respecting the robot namespace and
    robot_description parameter used by Interbotix MoveIt launches.
    """
    robot_description_ns = "/{}/robot_description".format(robot_name)
    robot_description = (
        robot_description_ns
        if rospy.has_param(robot_description_ns)
        else "robot_description"
    )
    ns = "/{}".format(robot_name)

    # Handle MoveIt API differences across ROS distros.
    try:
        return MoveGroupCommander(
            group_name,
            robot_description=robot_description,
            ns=ns,
        )
    except TypeError:
        try:
            return MoveGroupCommander(
                group_name,
                robot_description=robot_description,
            )
        except TypeError:
            return MoveGroupCommander(group_name)


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

        group = make_move_group(MOVE_GROUP, ROBOT_NAME)
        group.set_max_velocity_scaling_factor(0.2)
        group.set_max_acceleration_scaling_factor(0.2)
        group.set_start_state_to_current_state()

        rospy.loginfo("Planning frame: %s", group.get_planning_frame())
        rospy.loginfo("End-effector link: %s", group.get_end_effector_link())

        current_pose = group.get_current_pose().pose
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
