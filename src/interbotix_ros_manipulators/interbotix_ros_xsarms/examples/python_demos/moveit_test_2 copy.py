#!/usr/bin/env python3

import sys
import copy

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown

ROBOT_NAME = "wx250s"
MOVE_GROUP = "interbotix_arm"
FORWARD_METERS = 0.05


def make_move_group(group_name, robot_name):

    robot_description_ns = "/{}/robot_description".format(robot_name)
    robot_description = (
        robot_description_ns
        if rospy.has_param(robot_description_ns)
        else "robot_description"
    )
    ns = "/{}".format(robot_name)

    move_group = MoveGroupCommander(group_name)
    return move_group

    #try:
    #    return MoveGroupCommander(
    #        group_name,
    #        robot_description=robot_description,
    #        ns=ns,
    #    )
    #except TypeError:
    #    try:
    #        return MoveGroupCommander(
    #            group_name,
    #            robot_description=robot_description,
    #        )
    #    except TypeError:
    #        return MoveGroupCommander(group_name)


def main():
    roscpp_initialize(sys.argv)
    rospy.init_node("moveit_test_forward", anonymous=True)

    try:
        group = make_move_group("interbotix_arm", "wx250s")
        group.set_max_velocity_scaling_factor(0.1)
        group.set_max_acceleration_scaling_factor(0.1)
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
