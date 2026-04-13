#!/usr/bin/env python3

"""
Demonstration Replay Script for Interbotix WX250S

Replays demonstrations saved in learning_thousand_tasks format by
demo_collect_v2.py. Works in two modes, auto-detected at startup:

  1. Real robot (xsarm_control.launch running):
     Uses InterbotixManipulatorXS SDK to command the arm via trajectory messages.

  2. Simulation / RViz (xsarm_description.launch running):
     Publishes JointState messages directly so robot_state_publisher
     updates the TF tree and RViz shows the motion.

Usage:
  # With real robot:
  roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250s
  python replay_demo_v2_last.py -d collected_demos/session_XXXXXXXX/demo_0000

  # With RViz simulation:
  roslaunch interbotix_xsarm_descriptions xsarm_description.launch robot_model:=wx250s use_joint_pub_gui:=false
  python replay_demo_v2_last.py -d collected_demos/session_XXXXXXXX/demo_0000

Options:
  --speed_factor   Speed multiplier (default: 1.0, slower: 0.5, faster: 2.0)
  --downsample     Use every Nth waypoint (default: 3, i.e. ~10Hz from 30Hz)
  --dry_run        Only compute IK, don't move the robot
"""

import rospy
import numpy as np
import modern_robotics as mr
import os
import sys
import argparse
import math
import subprocess
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import interbotix_xs_modules.mr_descriptions as mrd
from tf.transformations import quaternion_from_matrix
from moveit_commander import MoveGroupCommander, roscpp_initialize

# ---------------------------------------------------------------------------
# WX250S constants (used in simulation mode when xs_sdk is not available)
# ---------------------------------------------------------------------------
WX250S_ARM_JOINT_NAMES = [
    'waist', 'shoulder', 'elbow',
    'forearm_roll', 'wrist_angle', 'wrist_rotate'
]

# Dynamixel position register to radians: (val - 2048) * 2pi / 4096
WX250S_JOINT_LOWER_LIMITS = [
    -3.14159,   # waist         (reg 0)
    -1.88496,   # shoulder      (reg 819)
    -2.14675,   # elbow         (reg 648)
    -3.14159,   # forearm_roll  (reg 0)
    -1.74533,   # wrist_angle   (reg 910)
    -3.14159,   # wrist_rotate  (reg 0)
]

WX250S_JOINT_UPPER_LIMITS = [
     3.14159,   # waist         (reg 4095)
     1.98968,   # shoulder      (reg 3345)
     1.60429,   # elbow         (reg 3094)
     3.14159,   # forearm_roll  (reg 4095)
     2.14675,   # wrist_angle   (reg 3447)
     3.14159,   # wrist_rotate  (reg 4095)
]

WX250S_SLEEP_POSITIONS = [0, -1.80, 1.55, 0, 0.8, 0]

# Gripper finger joint limits (prismatic, meters)
GRIPPER_OPEN_POS = 0.037       # left_finger when open
GRIPPER_CLOSED_POS = 0.015     # left_finger when closed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_real_robot(robot_name, timeout=3.0):
    """Return True if the xs_sdk services are reachable (real robot mode)."""
    service_name = "/" + robot_name + "/get_robot_info"
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
        return True
    except rospy.exceptions.ROSException:
        return False


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# DemoReplayer
# ---------------------------------------------------------------------------
class DemoReplayer:
    """Replays recorded EEF demonstrations on the Interbotix arm."""

    def __init__(self, robot_model="wx250s", robot_name="wx250s"):
        self.robot_model = robot_model
        self.robot_name = robot_name

        # Kinematics (always available, no ROS dependency)
        self.robot_des = getattr(mrd, robot_model)
        self.Slist = self.robot_des.Slist
        self.M = self.robot_des.M
        self.rev = 2 * math.pi

        # Initialize ROS node
        rospy.init_node('demo_replayer', anonymous=True)

        # Auto-detect mode
        print("\nDetecting robot mode...")
        self.sim_mode = not detect_real_robot(robot_name)

        if self.sim_mode:
            self._init_sim_mode()
        else:
            self._init_real_mode()

        self._init_moveit()

    # -- Initialisation per mode ------------------------------------------

    def _init_sim_mode(self):
        """Initialise for RViz-only (no xs_sdk)."""
        print("  xs_sdk not found -> SIMULATION mode (RViz)")

        self.joint_names = list(WX250S_ARM_JOINT_NAMES)
        self.num_joints = len(self.joint_names)
        self.joint_lower_limits = list(WX250S_JOINT_LOWER_LIMITS)
        self.joint_upper_limits = list(WX250S_JOINT_UPPER_LIMITS)
        self.sleep_positions = list(WX250S_SLEEP_POSITIONS)

        # All joint names published in the JointState message (arm + fingers)
        self.all_joint_names = self.joint_names + ['left_finger', 'right_finger']
        self.joint_state_topic = '/' + self.robot_name + '/joint_states'

        # Prevent competing JointState sources from fighting this replay node.
        self._stop_competing_joint_state_publishers()

        # Current state for JointState publishing
        self.current_arm_positions = list(self.sleep_positions)
        self.current_gripper_closed = False

        # Publisher for visualisation
        self.js_pub = rospy.Publisher(
            self.joint_state_topic,
            JointState,
            queue_size=10
        )
        # Give publisher time to register with subscribers
        rospy.sleep(0.5)

        print(f"  Publishing JointState on: {self.joint_state_topic}")
        print(f"  Joints: {self.joint_names}")

    def _stop_competing_joint_state_publishers(self):
        """Stop default joint-state publishers that conflict in simulation mode."""
        my_node = rospy.get_name()

        try:
            _, _, system_state = rospy.get_master().getSystemState()
            publishers = system_state[0]
        except Exception as exc:
            print(f"  Warning: could not query ROS master publishers: {exc}")
            return

        competing_nodes = []
        for topic, nodes in publishers:
            if topic != self.joint_state_topic:
                continue
            for node in nodes:
                if node == my_node:
                    continue
                short_name = node.rsplit('/', 1)[-1]
                if short_name in ("joint_state_publisher", "joint_state_publisher_gui"):
                    competing_nodes.append(node)

        if not competing_nodes:
            return

        print(f"  Found competing JointState publishers on {self.joint_state_topic}:")
        for node in competing_nodes:
            print(f"    - {node}")

        for node in competing_nodes:
            try:
                subprocess.check_call(
                    ['rosnode', 'kill', node],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  Stopped conflicting publisher: {node}")
            except Exception as exc:
                print(f"  Warning: failed to stop {node}: {exc}")

        rospy.sleep(0.2)

    def _init_real_mode(self):
        """Initialise for real robot (xs_sdk running)."""
        print("  xs_sdk found -> REAL ROBOT mode")

        from interbotix_xs_modules.arm import InterbotixManipulatorXS
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from interbotix_xs_msgs.msg import JointTrajectoryCommand

        # Store these for later use in real-robot methods
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._JointTrajectoryCommand = JointTrajectoryCommand

        self.bot = InterbotixManipulatorXS(
            robot_model=self.robot_model,
            robot_name=self.robot_name,
            moving_time=2.0,
            accel_time=0.5,
            init_node=False
        )

        self.joint_names = list(self.bot.arm.group_info.joint_names)
        self.num_joints = self.bot.arm.group_info.num_joints
        self.joint_lower_limits = list(self.bot.arm.group_info.joint_lower_limits)
        self.joint_upper_limits = list(self.bot.arm.group_info.joint_upper_limits)

        print(f"  Robot initialised with {self.num_joints} joints: {self.joint_names}")

    def _init_moveit(self):
        """Initialise MoveIt commander for trajectory IK from EEF waypoints."""
        roscpp_initialize(sys.argv)

        robot_description_ns = f"/{self.robot_name}/robot_description"
        semantic_ns = f"/{self.robot_name}/robot_description_semantic"
        moveit_ns = f"/{self.robot_name}"

        if not rospy.has_param(semantic_ns):
            print(f"Error: missing MoveIt semantic parameter: {semantic_ns}")
            print("Launch MoveIt first, e.g.:")
            print(
                f"  roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                f"robot_model:={self.robot_model} robot_name:={self.robot_name} "
                "use_fake:=true dof:=6"
            )
            sys.exit(1)

        try:
            self.move_group = MoveGroupCommander(
                "interbotix_arm",
                robot_description=robot_description_ns,
                ns=moveit_ns,
            )
        except TypeError:
            self.move_group = MoveGroupCommander(
                "interbotix_arm",
                robot_description=robot_description_ns,
            )

        self.move_group.set_max_velocity_scaling_factor(0.2)
        self.move_group.set_max_acceleration_scaling_factor(0.2)
        print("  MoveIt initialised (group: interbotix_arm)")

    # -- Demo loading (shared) --------------------------------------------

    def load_demo(self, demo_dir):
        """Load a demonstration directory into a dict."""
        demo = {}

        eef_poses_path = os.path.join(demo_dir, "eef_poses.npy")
        twists_path = os.path.join(demo_dir, "demo_eef_twists.npy")
        timestamps_path = os.path.join(demo_dir, "timestamps.npy")
        bottleneck_path = os.path.join(demo_dir, "bottleneck_pose.npy")

        if not os.path.exists(eef_poses_path):
            print(f"Error: {eef_poses_path} not found.")
            sys.exit(1)

        demo['eef_poses'] = np.load(eef_poses_path)
        demo['eef_twists'] = np.load(twists_path)
        demo['bottleneck_pose'] = np.load(bottleneck_path)

        if os.path.exists(timestamps_path):
            demo['timestamps'] = np.load(timestamps_path)
        else:
            T = len(demo['eef_poses'])
            demo['timestamps'] = np.arange(T) / 30.0

        print(f"\nLoaded demo from: {demo_dir}")
        print(f"  Timesteps: {len(demo['eef_poses'])}")
        print(f"  Duration:  {demo['timestamps'][-1]:.2f}s")

        return demo

    def set_live_data(self, live_bottleneck_pose, end_effector_twists):
        """Set live bottleneck pose and twists directly."""
        self.live_bottleneck_pose = live_bottleneck_pose
        self.end_effector_twists = end_effector_twists
        print(f"  Set live bottleneck pose: {live_bottleneck_pose.shape}")
        print(f"  Set end-effector twists: {end_effector_twists.shape}")

    def _check_joint_limits(self, theta_list):
        """Check that all joints are within configured limits."""
        for i in range(self.num_joints):
            if theta_list[i] < self.joint_lower_limits[i]:
                return False
            if theta_list[i] > self.joint_upper_limits[i]:
                return False
        return True

    def T_to_pose(self, T):
        """Convert a 4x4 SE(3) matrix to geometry_msgs/Pose."""
        pose = Pose()
        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])
        quat = quaternion_from_matrix(T)
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    def _extract_joint_trajectory_from_plan(self, plan):
        """Convert MoveIt RobotTrajectory to ordered joint positions + times."""
        joint_traj = plan.joint_trajectory
        if not joint_traj.points:
            return [], []

        name_to_idx = {name: idx for idx, name in enumerate(joint_traj.joint_names)}
        missing = [name for name in self.joint_names if name not in name_to_idx]
        if missing:
            print(f"Warning: MoveIt plan missing expected joints: {missing}")
            print("  Falling back to first N trajectory joints by position order.")

        joint_positions = []
        waypoint_times = []
        for point in joint_traj.points:
            if missing:
                if len(point.positions) < self.num_joints:
                    continue
                pos = list(point.positions[:self.num_joints])
            else:
                pos = [point.positions[name_to_idx[name]] for name in self.joint_names]

            if self._check_joint_limits(pos):
                joint_positions.append(pos)
                waypoint_times.append(point.time_from_start.to_sec())

        return joint_positions, waypoint_times

    @staticmethod
    def _state_at_time(query_t, state_times, state_values):
        """Return the last state value at or before query_t."""
        idx = np.searchsorted(state_times, query_t, side='right') - 1
        idx = max(0, min(idx, len(state_values) - 1))
        return state_values[idx]

    def compute_joint_trajectory(self, demo, downsample=3):
        """Pre-compute joint trajectory from EEF poses using MoveIt Cartesian IK."""
        eef_poses = demo['eef_poses']
        timestamps = demo['timestamps']
        gripper_col = demo['eef_twists'][:, 6]

        if len(eef_poses) == 0:
            return [], [], [], 0.0

        indices = list(range(0, len(eef_poses), downsample))
        if indices[-1] != len(eef_poses) - 1:
            indices.append(len(eef_poses) - 1)

        print(
            f"\nComputing MoveIt Cartesian IK for {len(indices)} waypoints "
            f"(downsampled {downsample}x from {len(eef_poses)} frames)..."
        )

        waypoints = [self.T_to_pose(eef_poses[idx]) for idx in indices]
        self.move_group.set_start_state_to_current_state()
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0
        )

        move_group.execute(plan, wait=True)
  
    # =====================================================================
    #  SIMULATION MODE - publish JointState to RViz
    # =====================================================================

    def _publish_joint_state(self, arm_positions, gripper_closed):
        """Publish a single JointState message (arm + gripper fingers)."""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.all_joint_names

        left_finger = GRIPPER_CLOSED_POS if gripper_closed else GRIPPER_OPEN_POS
        right_finger = -left_finger

        msg.position = list(arm_positions) + [left_finger, right_finger]
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)

        self.js_pub.publish(msg)

    def _sim_move_to(self, target_positions, gripper_closed,
                     duration=2.0, rate_hz=30):
        """Smoothly interpolate from current position to target in RViz."""
        start = np.array(self.current_arm_positions)
        end = np.array(target_positions)
        rate = rospy.Rate(rate_hz)
        steps = max(int(duration * rate_hz), 1)

        for i in range(steps + 1):
            if rospy.is_shutdown():
                return
            alpha = float(i) / steps
            interp = start + alpha * (end - start)
            self._publish_joint_state(interp, gripper_closed)
            rate.sleep()

        self.current_arm_positions = list(target_positions)
        self.current_gripper_closed = gripper_closed

    def replay_sim(self, joint_positions, waypoint_times,
                   gripper_states, speed_factor=1.0):
        """Replay in simulation by publishing JointState at the correct rate."""
        if len(joint_positions) < 2:
            print("Not enough waypoints to replay.")
            return

        total_time = (waypoint_times[-1] - waypoint_times[0]) / speed_factor
        print(f"\nReplaying in SIMULATION mode:")
        print(f"  Waypoints: {len(joint_positions)}")
        print(f"  Duration:  {total_time:.2f}s (speed_factor={speed_factor}x)")

        t0 = waypoint_times[0]
        replay_start = rospy.Time.now()

        idx = 0
        rate = rospy.Rate(60)  # 60 Hz for smooth visualisation
        last_pct_logged = -1

        while not rospy.is_shutdown() and idx < len(joint_positions) - 1:
            elapsed = (rospy.Time.now() - replay_start).to_sec()
            target_demo_time = t0 + elapsed * speed_factor

            # Advance index to match target time
            while (idx < len(waypoint_times) - 1 and
                   waypoint_times[idx + 1] <= target_demo_time):
                idx += 1

            # Interpolate between current and next waypoint
            if idx < len(waypoint_times) - 1:
                t_start = waypoint_times[idx]
                t_end = waypoint_times[idx + 1]
                seg_len = t_end - t_start
                if seg_len > 0:
                    alpha = min((target_demo_time - t_start) / seg_len, 1.0)
                else:
                    alpha = 1.0
                pos = (np.array(joint_positions[idx]) * (1 - alpha) +
                       np.array(joint_positions[idx + 1]) * alpha)
            else:
                pos = np.array(joint_positions[idx])

            gripper_closed = gripper_states[idx]

            if gripper_closed != self.current_gripper_closed:
                print(f"  Gripper {'CLOSED' if gripper_closed else 'OPENED'}")
                self.current_gripper_closed = gripper_closed

            self._publish_joint_state(pos, gripper_closed)
            self.current_arm_positions = list(pos)
            rate.sleep()

            # Progress logging (every 25%)
            if total_time > 0:
                pct = int(elapsed / total_time * 4) * 25
                if pct != last_pct_logged and pct <= 100:
                    print(f"  Progress: {pct}%")
                    last_pct_logged = pct

        # Hold final pose briefly
        self._publish_joint_state(joint_positions[-1], gripper_states[-1])
        self.current_arm_positions = list(joint_positions[-1])
        print("  Replay complete.")

    # =====================================================================
    #  REAL ROBOT MODE - use InterbotixManipulatorXS SDK
    # =====================================================================

    def replay_real_trajectory(self, joint_positions, waypoint_times,
                               gripper_states, speed_factor=1.0):
        """Replay on real robot using JointTrajectory messages."""
        JointTrajectory = self._JointTrajectory
        JointTrajectoryPoint = self._JointTrajectoryPoint
        JointTrajectoryCommand = self._JointTrajectoryCommand

        if len(joint_positions) < 2:
            print("Not enough waypoints to replay.")
            return

        # Split at gripper transitions
        segments = []
        seg_start = 0
        for i in range(1, len(gripper_states)):
            if gripper_states[i] != gripper_states[i - 1]:
                segments.append((seg_start, i, gripper_states[i - 1]))
                seg_start = i
        segments.append((seg_start, len(gripper_states), gripper_states[-1]))

        total_time = (waypoint_times[-1] - waypoint_times[0]) / speed_factor
        print(f"\nReplaying on REAL ROBOT:")
        print(f"  Waypoints: {len(joint_positions)}")
        print(f"  Duration:  {total_time:.2f}s (speed_factor={speed_factor}x)")
        print(f"  Segments:  {len(segments)} (split at gripper transitions)")

        for seg_idx, (start, end, gripper_closed) in enumerate(segments):
            seg_positions = joint_positions[start:end]
            seg_times = waypoint_times[start:end]

            if len(seg_positions) < 2:
                if gripper_closed:
                    self.bot.gripper.close(delay=0.5)
                    print("  Gripper CLOSED")
                else:
                    self.bot.gripper.open(delay=0.5)
                    print("  Gripper OPENED")
                continue

            # Build JointTrajectory
            joint_traj = JointTrajectory()
            joint_traj.joint_names = self.joint_names

            t0 = seg_times[0]
            for pos, t in zip(seg_positions, seg_times):
                point = JointTrajectoryPoint()
                point.positions = pos
                point.time_from_start = rospy.Duration.from_sec(
                    (t - t0) / speed_factor
                )
                joint_traj.points.append(point)

            # Snap first point to actual position to avoid a jump
            current_positions = []
            with self.bot.dxl.js_mutex:
                for name in self.joint_names:
                    current_positions.append(
                        self.bot.dxl.joint_states.position[
                            self.bot.dxl.js_index_map[name]
                        ]
                    )
            joint_traj.points[0].positions = current_positions

            seg_duration = (seg_times[-1] - seg_times[0]) / speed_factor
            avg_wp_time = seg_duration / max(len(seg_positions) - 1, 1)
            wp_moving_time = max(avg_wp_time, 0.1)
            wp_accel_time = min(wp_moving_time / 2.0, 0.1)
            self.bot.arm.set_trajectory_time(wp_moving_time, wp_accel_time)

            joint_traj.header.stamp = rospy.Time.now()
            traj_cmd = JointTrajectoryCommand(
                "group", self.bot.arm.group_name, joint_traj
            )
            self.bot.dxl.pub_traj.publish(traj_cmd)

            print(f"  Segment {seg_idx + 1}/{len(segments)}: "
                  f"{len(seg_positions)} wp, {seg_duration:.2f}s, "
                  f"gripper={'CLOSED' if gripper_closed else 'OPEN'}")

            rospy.sleep(seg_duration + wp_moving_time)

            self.bot.arm.joint_commands = list(seg_positions[-1])
            self.bot.arm.T_sb = mr.FKinSpace(
                self.M, self.Slist, seg_positions[-1]
            )

            # Gripper transition
            if seg_idx < len(segments) - 1:
                next_gripper = segments[seg_idx + 1][2]
                if next_gripper and not gripper_closed:
                    self.bot.gripper.close(delay=0.5)
                    print("  Gripper CLOSED")
                elif not next_gripper and gripper_closed:
                    self.bot.gripper.open(delay=0.5)
                    print("  Gripper OPENED")

    def replay_real_point_by_point(self, joint_positions, waypoint_times,
                                    gripper_states, speed_factor=1.0):
        """Simple point-by-point replay on real robot."""
        if not joint_positions:
            print("No waypoints to replay.")
            return

        print(f"\nReplaying point-by-point on REAL ROBOT:")
        print(f"  Waypoints: {len(joint_positions)}")

        gripper_is_closed = False

        for i in range(len(joint_positions)):
            if gripper_states[i] and not gripper_is_closed:
                self.bot.gripper.close(delay=0.3)
                gripper_is_closed = True
                print(f"  [{i}] Gripper CLOSED")
            elif not gripper_states[i] and gripper_is_closed:
                self.bot.gripper.open(delay=0.3)
                gripper_is_closed = False
                print(f"  [{i}] Gripper OPENED")

            if i < len(joint_positions) - 1:
                dt = (waypoint_times[i + 1] - waypoint_times[i]) / speed_factor
                dt = max(dt, 0.05)
            else:
                dt = 0.2

            moving_time = max(dt, 0.1)
            accel_time = min(moving_time / 2.0, 0.1)

            self.bot.arm.publish_positions(
                joint_positions[i],
                moving_time=moving_time,
                accel_time=accel_time,
                blocking=False
            )
            rospy.sleep(dt)

            if (i + 1) % 25 == 0:
                print(f"  Progress: {i + 1}/{len(joint_positions)}")

        rospy.sleep(0.3)
        print("  Replay complete.")

    # =====================================================================
    #  Main entry point
    # =====================================================================

    def run(self, demo_dir, speed_factor=1.0, downsample=3,
            dry_run=False, mode="trajectory"):
        # Load
        demo = self.load_demo(demo_dir)

        # IK
        joint_positions, waypoint_times, gripper_states, success_rate = \
            self.compute_joint_trajectory(demo, downsample)

        if not joint_positions:
            print("Error: IK failed for all waypoints. Cannot replay.")
            return

        if success_rate < 0.5:
            print(f"Warning: IK success rate is low ({success_rate*100:.1f}%).")

        if dry_run:
            print("\n[DRY RUN] IK complete. Not moving the robot.")
            return

        # -- Move to start pose --
        print("\nMoving to start pose...")
        if self.sim_mode:
            self._sim_move_to(joint_positions[0], gripper_states[0], duration=2.0)
        else:
            self.bot.arm.set_trajectory_time(2.0, 0.5)
            self.bot.arm.publish_positions(joint_positions[0], moving_time=2.0,
                                            accel_time=0.5, blocking=True)
            if gripper_states[0]:
                self.bot.gripper.close(delay=0.5)
            else:
                self.bot.gripper.open(delay=0.5)

        # -- Replay --
        if self.sim_mode:
            self.replay_sim(
                joint_positions, waypoint_times, gripper_states, speed_factor
            )
        elif mode == "trajectory":
            self.replay_real_trajectory(
                joint_positions, waypoint_times, gripper_states, speed_factor
            )
        else:
            self.replay_real_point_by_point(
                joint_positions, waypoint_times, gripper_states, speed_factor
            )

        # -- Return to sleep --
        print("\nReturning to sleep pose...")
        if self.sim_mode:
            self._sim_move_to(self.sleep_positions, False, duration=2.0)
        else:
            self.bot.arm.set_trajectory_time(2.0, 0.5)
            self.bot.arm.go_to_sleep_pose()

        print("Done.")


    def run_with_live_data(self, speed_factor=1.0, dry_run=False, mode="trajectory"):
        """Run replay using live bottleneck pose and twists."""
        print(f"\nRunning replay with live data:")
        print(f"  Bottleneck pose: {self.live_bottleneck_pose.shape}")
        print(f"  Twists: {self.end_effector_twists.shape}")

        eef_poses = self.compute_trajectory_from_twists()
        timestamps = np.arange(len(eef_poses)) / 30.0
        demo = {
            'eef_poses': eef_poses,
            'eef_twists': self.end_effector_twists,
            'timestamps': timestamps,
            'bottleneck_pose': self.live_bottleneck_pose,
        }

        joint_positions, waypoint_times, gripper_states, success_rate = \
            self.compute_joint_trajectory(demo, downsample=1)

        if not joint_positions:
            print("Error: IK failed for all waypoints. Cannot replay.")
            return

        if success_rate < 0.5:
            print(f"Warning: IK success rate is low ({success_rate*100:.1f}%).")

        if dry_run:
            print("\n[DRY RUN] IK complete. Not moving the robot.")
            return

        print("\nMoving to bottleneck pose...")
        if self.sim_mode:
            self._sim_move_to(joint_positions[0], gripper_states[0], duration=2.0)
        else:
            self.bot.arm.publish_positions(
                joint_positions[0], moving_time=2.0, accel_time=0.5, blocking=True
            )

        if self.sim_mode:
            self.replay_sim(joint_positions, waypoint_times, gripper_states, speed_factor)
        elif mode == "trajectory":
            self.replay_real_trajectory(
                joint_positions, waypoint_times, gripper_states, speed_factor
            )
        else:
            self.replay_real_point_by_point(
                joint_positions, waypoint_times, gripper_states, speed_factor
            )

        print("\nReturning to sleep pose...")
        if self.sim_mode:
            self._sim_move_to(self.sleep_positions, False, duration=2.0)
        else:
            self.bot.arm.set_trajectory_time(2.0, 0.5)
            self.bot.arm.go_to_sleep_pose()

        print("Done.")
      

def main():
    parser = argparse.ArgumentParser(
        description='Replay demonstrations (real robot or RViz simulation)'
    )
    parser.add_argument(
        '--demo_dir', '-d', type=str, required=True,
        help='Path to demo directory (containing demo_eef_twists.npy, etc.)'
    )
    parser.add_argument(
        '--robot_model', type=str, default='wx250s',
        help='Robot model (default: wx250s)'
    )
    parser.add_argument(
        '--robot_name', type=str, default='wx250s',
        help='Robot name/namespace (default: wx250s)'
    )
    parser.add_argument(
        '--speed_factor', '-s', type=float, default=1.0,
        help='Speed multiplier (default: 1.0, slower: 0.5, faster: 2.0)'
    )
    parser.add_argument(
        '--downsample', '-n', type=int, default=3,
        help='Use every Nth waypoint (default: 3, i.e. ~10Hz from 30Hz)'
    )
    parser.add_argument(
        '--mode', '-m', type=str, default='trajectory',
        choices=['trajectory', 'point_by_point'],
        help='Replay mode for real robot (default: trajectory)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Only compute IK, do not move'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("DEMONSTRATION REPLAY v2")
    print("="*60)
    print(f"Demo:       {args.demo_dir}")
    print(f"Speed:      {args.speed_factor}x")
    print(f"Downsample: {args.downsample}x")
    print(f"Mode:       {args.mode}")
    print(f"Dry run:    {args.dry_run}")
    print("="*60)

    replayer = DemoReplayer(
        robot_model=args.robot_model,
        robot_name=args.robot_name
    )

    replayer.run(
        demo_dir=args.demo_dir,
        speed_factor=args.speed_factor,
        downsample=args.downsample,
        dry_run=args.dry_run,
        mode=args.mode
    )


if __name__ == '__main__':
    main()
