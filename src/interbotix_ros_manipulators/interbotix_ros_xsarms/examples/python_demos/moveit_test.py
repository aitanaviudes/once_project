#!/usr/bin/env python3

import sys
import copy
import argparse

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown


def make_move_group(group_name, robot_name):

    robot_description_ns = "/{}/robot_description".format(robot_name)
    robot_description = (
        robot_description_ns
        if rospy.has_param(robot_description_ns)
        else "robot_description"
    )
    ns = "/{}".format(robot_name)

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
    parser = argparse.ArgumentParser(
        description="Simple MoveIt test: move EEF a little forward."
    )
    parser.add_argument("--robot_name", default="wx250s")
    parser.add_argument("--move_group", default="interbotix_arm")
    parser.add_argument(
        "--forward",
        type=float,
        default=0.05,
        help="Forward motion in meters along +X of the planning frame.",
    )
    args = parser.parse_args()

    roscpp_initialize(sys.argv)
    rospy.init_node("moveit_test_forward", anonymous=True)

    try:
        group = make_move_group(args.move_group, args.robot_name)
        group.set_max_velocity_scaling_factor(0.2)
        group.set_max_acceleration_scaling_factor(0.2)

        rospy.loginfo("Planning frame: %s", group.get_planning_frame())
        rospy.loginfo("End-effector link: %s", group.get_end_effector_link())
        rospy.loginfo("Group joints: %s", group.get_joints())

        current_pose = group.get_current_pose().pose
        target_pose = copy.deepcopy(current_pose)
        target_pose.position.x += args.forward

        waypoints = [copy.deepcopy(current_pose), copy.deepcopy(target_pose)]
        plan, fraction = group.compute_cartesian_path(
            waypoints,
            0.005,  # eef_step
            0.0,    # jump_threshold
        )

        if fraction > 0.95:
            rospy.loginfo("Executing Cartesian path (fraction=%.2f)", fraction)
            group.execute(plan, wait=True)
        else:
            rospy.logwarn(
                "Cartesian path fraction low (%.2f). Falling back to pose goal.",
                fraction,
            )
            group.set_pose_target(target_pose)
            success = group.go(wait=True)
            group.stop()
            group.clear_pose_targets()
            if not success:
                rospy.logerr("Pose-goal planning/execution failed.")
                return 1

        group.stop()
        rospy.loginfo("Done.")
        return 0
    finally:
        roscpp_shutdown()


if __name__ == "__main__":
    sys.exit(main())
