#!/usr/bin/env python3

"""
Demonstration replay using MoveIt Cartesian planning + execute.

Key improvements over original:
  1. Trajectory time-parameterization via iterative_time_parameterization so the
     controller receives valid time stamps → smooth, non-glitchy motion.
  2. Gripper open/close driven by gripper_states.npy (or the 7th column of
     eef_poses.npy), with two fallback strategies:
       a) interbotix_gripper MoveGroup (preferred)
       b) Direct ROS topic /wx250s/commands/joint_single (xs_sdk)
  3. Continuous trajectory: the full arm path is planned once and executed in one
     shot; chunked fallback is kept but also time-parameterized.
  4. Smarter start-pose alignment with orientation tolerance relaxation.
"""

import argparse
import math
import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from interbotix_xs_msgs.msg import JointSingleCommand
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from moveit_msgs.msg import RobotTrajectory
from tf.transformations import quaternion_from_matrix
from trajectory_msgs.msg import JointTrajectoryPoint

# ---------------------------------------------------------------------------
# Optional: MoveIt trajectory processing tools
# ---------------------------------------------------------------------------
try:
    from moveit_commander import PlanningSceneInterface  # noqa: F401 – keep import for side-effects
except ImportError:
    pass

try:
    # Available in moveit_ros_planning_interface Python bindings (Noetic+)
    from moveit.core.robot_trajectory import RobotTrajectory as CoreTrajectory  # noqa: F401
    _HAS_CORE_TRAJ = True
except ImportError:
    _HAS_CORE_TRAJ = False

# ---------------------------------------------------------------------------
# Gripper constants for wx250s
# ---------------------------------------------------------------------------
GRIPPER_OPEN_VALUE = 0.057   # metres (finger gap) — tune to your robot
GRIPPER_CLOSED_VALUE = 0.010
GRIPPER_OPEN_EFFORT = -1.0   # xs_sdk effort convention: negative = open
GRIPPER_CLOSE_EFFORT = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_real_robot(robot_name: str, timeout: float = 3.0) -> bool:
    """Return True if xs_sdk service is reachable (real-robot mode)."""
    service_name = f"/{robot_name}/get_robot_info"
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
        return True
    except rospy.exceptions.ROSException:
        return False


def _time_parameterize_plan(
    plan: RobotTrajectory,
    robot: RobotCommander,
    group_name: str,
    vel_scale: float,
    accel_scale: float,
) -> RobotTrajectory:
    """
    Apply MoveIt's iterative time parameterization to a RobotTrajectory.

    This fills in valid `time_from_start` fields so the joint-trajectory
    controller produces smooth, continuous motion instead of glitchy jumps.

    Falls back gracefully if the MoveIt bindings don't expose the method.
    """
    try:
        from moveit_commander import conversions  # noqa: F401
        # RobotState → needed to call the C++ planner directly via the Python API
        from moveit.core.robot_state import RobotState  # type: ignore
        from moveit.core.robot_trajectory import RobotTrajectory as CoreTraj  # type: ignore
        from moveit.core.planning_interface import IterativeParabolicTimeParameterization  # type: ignore

        rs = RobotState(robot.get_robot_model())
        core_traj = CoreTraj(robot.get_robot_model(), group_name)
        # Populate core trajectory from ROS message
        core_traj.set_robot_trajectory_msg(rs, plan)
        iptp = IterativeParabolicTimeParameterization()
        iptp.compute_time_stamps(core_traj, vel_scale, accel_scale)
        plan2 = RobotTrajectory()
        core_traj.get_robot_trajectory_msg(plan2)
        return plan2
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Pure-Python fallback: assign uniform time stamps based on max joint
    # velocity scaling.  Not as smooth as IPTP but vastly better than all-zero
    # timestamps.
    # -----------------------------------------------------------------------
    points = plan.joint_trajectory.points
    if len(points) < 2:
        return plan

    # Check if timestamps are already meaningful
    if points[-1].time_from_start.to_sec() > 1e-6:
        return plan  # already parameterized

    # Estimate a safe inter-point duration from velocity limits
    # (use a conservative 0.05 s per point as a fallback)
    dt = 0.05 / max(vel_scale, 0.01)

    t = 0.0
    for pt in points:
        pt.time_from_start = rospy.Duration(t)
        t += dt

    return plan


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DemoReplayer:
    """Replay recorded EEF demonstrations through MoveIt."""

    def __init__(
        self,
        robot_model: str = "wx250s",
        robot_name: str = "wx250s",
        eef_step: float = 0.005,       # finer default for smoother paths
        jump_threshold: float = 0.0,
        min_fraction: float = 0.90,
        base_vel_scale: float = 0.4,
        base_accel_scale: float = 0.4,
        avoid_collisions: bool = False,
        gripper_threshold: float = 0.5,  # gripper state binarisation threshold
    ):
        self.robot_model = robot_model
        self.robot_name = robot_name
        self.eef_step = eef_step
        self.jump_threshold = jump_threshold
        self.min_fraction = min_fraction
        self.base_vel_scale = base_vel_scale
        self.base_accel_scale = base_accel_scale
        self.avoid_collisions = avoid_collisions
        self.gripper_threshold = gripper_threshold

        # Live-data mode attributes (set via set_live_data)
        self.live_bottleneck_pose = None
        self.end_effector_twists = None

        rospy.init_node("demo_replayer_moveit", anonymous=True)

        self.sim_mode = not detect_real_robot(robot_name)
        print("\nDetecting robot mode...")
        print("  Mode: SIMULATION" if self.sim_mode else "  Mode: REAL ROBOT")

        self._init_moveit()
        self._init_gripper()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_moveit(self):
        """Initialize MoveGroupCommander and RobotCommander."""
        roscpp_initialize(sys.argv)

        self.robot_description = f"/{self.robot_name}/robot_description"
        semantic_ns = f"/{self.robot_name}/robot_description_semantic"

        if not rospy.has_param(semantic_ns):
            print(f"Error: missing MoveIt semantic parameter: {semantic_ns}")
            print("Launch MoveIt first.")
            print("  Simulation:")
            print(
                "  roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                f"robot_model:={self.robot_model} robot_name:={self.robot_name} "
                "use_fake:=true dof:=6"
            )
            print("  Real robot:")
            print(
                "  roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                f"robot_model:={self.robot_model} robot_name:={self.robot_name} "
                "use_actual:=true dof:=6"
            )
            sys.exit(1)

        # RobotCommander is needed for time-parameterization
        try:
            self.robot_commander = RobotCommander(
                robot_description=self.robot_description,
                ns=f"/{self.robot_name}",
            )
        except Exception:
            self.robot_commander = RobotCommander()

        # Arm move group
        try:
            self.move_group = MoveGroupCommander(
                "interbotix_arm",
                robot_description=self.robot_description,
                ns=f"/{self.robot_name}",
            )
        except TypeError:
            self.move_group = MoveGroupCommander("interbotix_arm")

        self._apply_speed(self.base_vel_scale, self.base_accel_scale)
        self.move_group.set_planning_time(15.0)
        self.move_group.set_num_planning_attempts(20)
        self.move_group.set_goal_position_tolerance(0.005)
        self.move_group.set_goal_orientation_tolerance(0.10)
        try:
            self.move_group.allow_replanning(True)
        except Exception:
            pass

        print("  MoveIt initialized (arm)")
        print(f"  Planning frame: {self.move_group.get_planning_frame()}")
        print(f"  End effector:   {self.move_group.get_end_effector_link()}")

    def _init_gripper(self):
        """
        Set up gripper control with two strategies:
          1. interbotix_gripper MoveGroup  (preferred — uses MoveIt)
          2. Direct JointSingleCommand topic (fallback — lower-level xs_sdk)
        """
        self.gripper_group = None
        self.gripper_pub = None

        # --- Strategy 1: MoveIt gripper group ---
        try:
            self.gripper_group = MoveGroupCommander(
                "interbotix_gripper",
                robot_description=self.robot_description,
                ns=f"/{self.robot_name}",
            )
            self.gripper_group.set_max_velocity_scaling_factor(1.0)
            self.gripper_group.set_max_acceleration_scaling_factor(1.0)
            print("  Gripper: MoveIt interbotix_gripper group ready")
        except Exception as exc:
            print(f"  Gripper MoveGroup unavailable ({exc}); using direct topic.")

        # --- Strategy 2: xs_sdk JointSingleCommand topic ---
        topic = f"/{self.robot_name}/commands/joint_single"
        self.gripper_pub = rospy.Publisher(topic, JointSingleCommand, queue_size=5)
        rospy.sleep(0.3)  # let publisher register
        print(f"  Gripper: direct topic {topic} ready (fallback)")

    # ------------------------------------------------------------------
    # Demo loading
    # ------------------------------------------------------------------

    def load_demo(self, demo_dir: str) -> dict:
        """Load demonstration arrays from directory."""
        demo = {}

        eef_poses_path = os.path.join(demo_dir, "eef_poses.npy")
        twists_path = os.path.join(demo_dir, "demo_eef_twists.npy")
        timestamps_path = os.path.join(demo_dir, "timestamps.npy")
        gripper_path = os.path.join(demo_dir, "gripper_states.npy")

        if not os.path.exists(eef_poses_path):
            print(f"Error: {eef_poses_path} not found.")
            sys.exit(1)

        eef_data = np.load(eef_poses_path)

        # Support (N, 4, 4) pure SE3 OR (N, 4, 5) where the 5th column holds
        # [x, y, z, gripper_state, 1] — detect and split automatically.
        if eef_data.ndim == 3 and eef_data.shape[1] == 4 and eef_data.shape[2] == 5:
            demo["eef_poses"] = eef_data[:, :, :4]          # (N, 4, 4) SE3
            demo["gripper_states"] = eef_data[:, 2, 3]      # row-index 2, col 3 → z-column extended
            print("  Detected gripper state in eef_poses column 5")
        elif eef_data.ndim == 2 and eef_data.shape[1] == 7:
            # Flat format: [x, y, z, qx, qy, qz, gripper]
            demo["eef_poses"] = eef_data  # handled later in T_to_pose if needed
            demo["gripper_states"] = eef_data[:, 6]
            print("  Detected gripper state as 7th column of flat eef_poses")
        else:
            demo["eef_poses"] = eef_data

        demo["eef_twists"] = np.load(twists_path) if os.path.exists(twists_path) else None

        if os.path.exists(timestamps_path):
            demo["timestamps"] = np.load(timestamps_path)
        else:
            T = len(demo["eef_poses"])
            demo["timestamps"] = np.arange(T) / 30.0

        # Gripper states: prefer dedicated file, then embedded, then all-open
        if os.path.exists(gripper_path):
            demo["gripper_states"] = np.load(gripper_path)
            print(f"  Loaded gripper_states.npy ({len(demo['gripper_states'])} timesteps)")
        elif "gripper_states" not in demo:
            print(
                "  Warning: no gripper state data found. "
                "Gripper will stay open throughout."
            )
            demo["gripper_states"] = np.zeros(len(demo["eef_poses"]))

        print(f"\nLoaded demo from: {demo_dir}")
        print(f"  Timesteps: {len(demo['eef_poses'])}")
        print(f"  Duration:  {demo['timestamps'][-1]:.2f}s")
        return demo

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def T_to_pose(T) -> Pose:
        """Convert a 4×4 SE(3) matrix to geometry_msgs/Pose."""
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

    @staticmethod
    def _build_indices(length: int, downsample: int) -> list:
        indices = list(range(0, length, downsample))
        if indices[-1] != length - 1:
            indices.append(length - 1)
        return indices

    @staticmethod
    def _filter_waypoints(waypoints: list, position_only: bool = False) -> list:
        """Drop nearly-duplicate waypoints that break Cartesian interpolation."""
        if len(waypoints) <= 1:
            return waypoints

        filtered = [waypoints[0]]
        pos_eps = 5e-5   # tighter than original to keep more waypoints
        ang_eps = 5e-4

        for wp in waypoints[1:]:
            prev = filtered[-1]
            dp = math.sqrt(
                (wp.position.x - prev.position.x) ** 2 +
                (wp.position.y - prev.position.y) ** 2 +
                (wp.position.z - prev.position.z) ** 2
            )
            if position_only:
                if dp > pos_eps:
                    filtered.append(wp)
            else:
                dot = (
                    prev.orientation.x * wp.orientation.x +
                    prev.orientation.y * wp.orientation.y +
                    prev.orientation.z * wp.orientation.z +
                    prev.orientation.w * wp.orientation.w
                )
                dot = float(np.clip(abs(dot), -1.0, 1.0))
                dang = 2.0 * math.acos(dot)
                if (dp > pos_eps) or (dang > ang_eps):
                    filtered.append(wp)

        return filtered

    @staticmethod
    def _adaptive_eef_step(waypoints: list, requested_step: float) -> float:
        """Choose an eef_step that is at most half the median inter-waypoint gap."""
        if len(waypoints) < 2:
            return requested_step

        dists = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            d = math.sqrt(
                (a.position.x - b.position.x) ** 2 +
                (a.position.y - b.position.y) ** 2 +
                (a.position.z - b.position.z) ** 2
            )
            if d > 1e-6:
                dists.append(d)

        if not dists:
            return requested_step

        auto_step = max(2e-4, float(np.percentile(dists, 50)) * 0.5)
        return min(requested_step, auto_step)

    # ------------------------------------------------------------------
    # Gripper control
    # ------------------------------------------------------------------

    def _gripper_command(self, open_gripper: bool):
        """
        Send gripper open/close command.

        Tries MoveIt group first, falls back to direct xs_sdk topic.
        """
        action = "OPEN" if open_gripper else "CLOSE"
        print(f"  Gripper → {action}")

        # --- MoveIt group ---
        if self.gripper_group is not None:
            try:
                named = "Open" if open_gripper else "Closed"
                # Try named target (defined in SRDF)
                self.gripper_group.set_named_target(named)
                ok = self.gripper_group.go(wait=True)
                self.gripper_group.stop()
                if ok:
                    return
                print("    Named target failed, trying joint value...")
            except Exception:
                pass

            try:
                val = GRIPPER_OPEN_VALUE if open_gripper else GRIPPER_CLOSED_VALUE
                self.gripper_group.set_joint_value_target({"left_finger": val, "right_finger": val})
                ok = self.gripper_group.go(wait=True)
                self.gripper_group.stop()
                if ok:
                    return
            except Exception:
                pass

        # --- Direct topic fallback ---
        if self.gripper_pub is not None:
            msg = JointSingleCommand()
            msg.name = "gripper"
            msg.cmd = GRIPPER_OPEN_EFFORT if open_gripper else GRIPPER_CLOSE_EFFORT
            self.gripper_pub.publish(msg)
            rospy.sleep(0.6)  # allow gripper to physically move

    def _apply_gripper_states(self, gripper_states: np.ndarray, waypoint_indices: list):
        """
        Replay gripper open/close events at the waypoint timestamps.

        Only issues a command when the binarised gripper state changes,
        avoiding redundant commands mid-trajectory.
        """
        if gripper_states is None or len(gripper_states) == 0:
            return

        binary = (gripper_states > self.gripper_threshold).astype(int)  # 1=closed
        last_state = -1

        for idx in waypoint_indices:
            state = int(binary[min(idx, len(binary) - 1)])
            if state != last_state:
                self._gripper_command(open_gripper=(state == 0))
                last_state = state

    # ------------------------------------------------------------------
    # Speed helpers
    # ------------------------------------------------------------------

    def _apply_speed(self, vel: float, accel: float):
        self.move_group.set_max_velocity_scaling_factor(float(np.clip(vel, 0.01, 1.0)))
        self.move_group.set_max_acceleration_scaling_factor(float(np.clip(accel, 0.01, 1.0)))

    def _set_speed(self, speed_factor: float):
        vel = self.base_vel_scale * speed_factor
        accel = self.base_accel_scale * speed_factor
        self._apply_speed(vel, accel)

    # ------------------------------------------------------------------
    # Cartesian planning
    # ------------------------------------------------------------------

    def _plan_cartesian(
        self,
        eef_poses,
        downsample: int = 1,
        position_only: bool = False,
    ):
        """
        Plan a Cartesian path from EEF pose matrices.

        Returns (plan, fraction, waypoint_indices).
        The plan is time-parameterized for smooth execution.
        """
        if len(eef_poses) == 0:
            return None, 0.0, []

        indices = self._build_indices(len(eef_poses), max(1, downsample))
        waypoints = [self.T_to_pose(eef_poses[idx]) for idx in indices]

        if position_only and waypoints:
            qx = waypoints[0].orientation.x
            qy = waypoints[0].orientation.y
            qz = waypoints[0].orientation.z
            qw = waypoints[0].orientation.w
            for wp in waypoints:
                wp.orientation.x = qx
                wp.orientation.y = qy
                wp.orientation.z = qz
                wp.orientation.w = qw

        raw_count = len(waypoints)
        waypoints = self._filter_waypoints(waypoints, position_only=position_only)
        eef_step = self._adaptive_eef_step(waypoints, self.eef_step)

        if len(waypoints) < 2:
            print("  Warning: not enough unique waypoints after filtering.")
            return None, 0.0, []

        print(
            f"\nPlanning Cartesian path with {len(waypoints)} waypoints "
            f"(downsample={downsample}x, "
            f"{'position-only' if position_only else 'full-pose'})..."
        )
        if raw_count != len(waypoints):
            print(f"  Filtered duplicate waypoints: {raw_count} → {len(waypoints)}")
        print(f"  Effective eef_step: {eef_step:.6f} m")

        self.move_group.set_start_state_to_current_state()

        try:
            plan, fraction = self.move_group.compute_cartesian_path(
                waypoints, eef_step, self.jump_threshold,
            )
        except Exception:
            plan, fraction = self.move_group.compute_cartesian_path(
                waypoints, eef_step, self.avoid_collisions,
            )

        n_points = len(plan.joint_trajectory.points)
        print(f"  Cartesian fraction:  {fraction * 100:.1f}%")
        print(f"  Planned points:      {n_points}")

        # ----------------------------------------------------------------
        # KEY FIX: time-parameterize the trajectory so the controller
        # receives valid timestamps → smooth, continuous motion.
        # ----------------------------------------------------------------
        plan = self._time_parameterize(plan)

        return plan, float(fraction), indices

    def _time_parameterize(self, plan: RobotTrajectory) -> RobotTrajectory:
        """Apply IPTP (or fallback uniform timing) to a RobotTrajectory."""
        current_vel = self.move_group.get_max_velocity_scaling_factor() \
            if hasattr(self.move_group, "get_max_velocity_scaling_factor") \
            else self.base_vel_scale
        current_accel = self.move_group.get_max_acceleration_scaling_factor() \
            if hasattr(self.move_group, "get_max_acceleration_scaling_factor") \
            else self.base_accel_scale

        parameterized = _time_parameterize_plan(
            plan,
            self.robot_commander,
            "interbotix_arm",
            vel_scale=current_vel,
            accel_scale=current_accel,
        )
        pts_before = len(plan.joint_trajectory.points)
        pts_after = len(parameterized.joint_trajectory.points)
        if pts_after >= pts_before:
            print(f"  Time-parameterized: {pts_after} trajectory points")
        return parameterized

    # ------------------------------------------------------------------
    # Start-pose alignment
    # ------------------------------------------------------------------

    def _move_to_start_pose(self, start_pose: Pose) -> bool:
        """Move to the first demo pose with progressive tolerance relaxation."""
        print("\nMoving to trajectory start pose...")
        self.move_group.set_start_state_to_current_state()

        # Attempt 1: full pose, tight tolerance
        self.move_group.set_goal_orientation_tolerance(0.05)
        self.move_group.set_pose_target(start_pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if success:
            print("  Reached start pose (tight tolerance).")
            self.move_group.set_goal_orientation_tolerance(0.10)
            return True

        # Attempt 2: full pose, relaxed orientation tolerance
        print("  Retrying with relaxed orientation tolerance...")
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_goal_orientation_tolerance(0.30)
        self.move_group.set_pose_target(start_pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        self.move_group.set_goal_orientation_tolerance(0.10)
        if success:
            print("  Reached start pose (relaxed orientation).")
            return True

        # Attempt 3: position-only
        print("  Full pose failed; trying position-only start alignment...")
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_position_target([
            start_pose.position.x,
            start_pose.position.y,
            start_pose.position.z,
        ])
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if success:
            print("  Reached start position.")
            return True

        print("  Warning: failed to reach start pose/position.")
        return False

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_plan(
        self,
        plan: RobotTrajectory,
        fraction: float,
        dry_run: bool = False,
    ) -> bool:
        """Execute plan if fraction is acceptable."""
        if fraction < self.min_fraction:
            print(
                f"  Error: fraction too low ({fraction:.3f} < {self.min_fraction:.3f}). "
                "Not executing."
            )
            return False

        n_pts = len(plan.joint_trajectory.points)
        if n_pts < 2:
            print("  Error: planned trajectory has fewer than 2 points.")
            return False

        # Sanity-check timestamps
        t_end = plan.joint_trajectory.points[-1].time_from_start.to_sec()
        if t_end < 1e-6:
            print(
                "  Warning: trajectory has no time stamps — re-applying uniform timing."
            )
            plan = self._time_parameterize(plan)

        if dry_run:
            t_end = plan.joint_trajectory.points[-1].time_from_start.to_sec()
            print(f"\n[DRY RUN] Plan computed. {n_pts} points, {t_end:.2f}s. Not executing.")
            return True

        print(f"\nExecuting MoveIt plan ({n_pts} points)...")
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        print("  Done." if success else "  Execution failed.")
        return bool(success)

    def _execute_chunked_cartesian(
        self,
        eef_poses,
        gripper_states=None,
        chunk_size: int = 40,
        dry_run: bool = False,
    ) -> bool:
        """Fallback: execute the trajectory as several Cartesian chunks."""
        if len(eef_poses) < 2:
            print("Error: not enough poses for chunked fallback.")
            return False

        print(
            f"\nChunked Cartesian fallback: {len(eef_poses)} poses, chunk_size={chunk_size}"
        )

        if dry_run:
            print("[DRY RUN] Skipping chunked execution.")
            return True

        seg_idx = 0
        start = 0
        executed_segments = 0
        min_seg_fraction = max(0.20, self.min_fraction * 0.5)

        while start < len(eef_poses) - 1:
            end = min(start + chunk_size, len(eef_poses))
            seg = eef_poses[start:end]
            seg_idx += 1

            print(f"\n  Segment {seg_idx}: poses [{start}:{end}]")
            plan, fraction, seg_indices = self._plan_cartesian(seg, downsample=1, position_only=False)

            if plan is None or fraction < min_seg_fraction:
                print("    Full-pose segment short — retrying position-only...")
                plan, fraction, seg_indices = self._plan_cartesian(seg, downsample=1, position_only=True)

            if plan is None or len(plan.joint_trajectory.points) < 2 or fraction < 0.15:
                print("    Segment planning failed; skipping.")
                start = end - 1
                continue

            # Apply gripper transitions within this chunk
            if gripper_states is not None:
                abs_indices = [start + i for i in seg_indices]
                self._apply_gripper_states(gripper_states, abs_indices)

            ok = self.move_group.execute(plan, wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            if ok:
                executed_segments += 1
                print(f"    Executed (fraction={fraction:.3f})")
            else:
                print("    Execution failed for this segment.")

            start = end - 1  # overlap by one for continuity

        print(f"\nChunked execution summary: {executed_segments} segments executed.")
        return executed_segments > 0

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        demo_dir: str,
        speed_factor: float = 1.0,
        downsample: int = 1,         # default 1 → use all waypoints for smoothness
        dry_run: bool = False,
        mode: str = "trajectory",    # kept for CLI compat
    ):
        """Replay from saved demo directory."""
        _ = mode
        demo = self.load_demo(demo_dir)
        self._set_speed(speed_factor)

        eef_poses = demo["eef_poses"]
        gripper_states = demo.get("gripper_states")

        if len(eef_poses) < 2:
            print("Error: need at least 2 EEF poses.")
            return

        start_pose = self.T_to_pose(eef_poses[0])
        eef_for_cartesian = eef_poses[1:]

        if not dry_run:
            # Open gripper at start
            self._gripper_command(open_gripper=True)
            if not self._move_to_start_pose(start_pose):
                print("  Continuing without start alignment.")
                eef_for_cartesian = eef_poses

        # First gripper state at t=0
        if gripper_states is not None and len(gripper_states) > 0:
            initial_closed = bool(gripper_states[0] > self.gripper_threshold)
            if initial_closed:
                self._gripper_command(open_gripper=False)

        # --- Plan full trajectory ---
        plan, fraction, indices = self._plan_cartesian(
            eef_for_cartesian,
            downsample=max(1, downsample),
            position_only=False,
        )
        if plan is None:
            print("Error: no poses to plan.")
            return

        if fraction < self.min_fraction:
            print("  Full-pose Cartesian path too short — trying position-only fallback...")
            plan, fraction, indices = self._plan_cartesian(
                eef_for_cartesian,
                downsample=max(1, downsample * 2),
                position_only=True,
            )
            if plan is None:
                print("Error: no poses to plan in fallback.")
                return

        if fraction < self.min_fraction:
            print(
                "  Cartesian planning still below threshold — "
                "falling back to chunked Cartesian..."
            )
            self._execute_chunked_cartesian(
                eef_for_cartesian,
                gripper_states=gripper_states,
                chunk_size=max(20, 60 // max(1, downsample)),
                dry_run=dry_run,
            )
            return

        # --- Apply all gripper transitions upfront (before arm moves) ---
        # For the non-chunked path we interleave gripper commands with execution
        # by pre-identifying the state-change timestamps and injecting waits.
        # Since MoveIt executes the full arm plan atomically, we apply gripper
        # transitions AFTER the arm motion completes when using the atomic path.
        # (For tightly-timed demos, use the chunked path instead.)
        arm_success = self._execute_plan(plan, fraction, dry_run=dry_run)

        if arm_success and not dry_run and gripper_states is not None:
            # Replay gripper transitions post-arm (position-keyed)
            print("\nApplying gripper state transitions...")
            self._apply_gripper_states(gripper_states, indices)

    # ------------------------------------------------------------------
    # Live data (bottleneck pose + twists)
    # ------------------------------------------------------------------

    def set_live_data(self, live_bottleneck_pose, end_effector_twists):
        """Set live bottleneck pose and twists directly."""
        self.live_bottleneck_pose = live_bottleneck_pose
        self.end_effector_twists = end_effector_twists
        print(f"  Set live bottleneck pose: {live_bottleneck_pose.shape}")
        print(f"  Set end-effector twists: {end_effector_twists.shape}")

    def se3_exp(self, vec):
        """Build a transformation matrix using the Lie algebra exponential map."""
        assert len(vec) == 6
        phi = vec[3:]
        rho = vec[:3]

        angle = np.linalg.norm(phi)
        if angle < 1e-12:
            R = np.eye(3)
            J = np.eye(3)
        else:
            axis = phi / angle
            cp = np.cos(angle)
            sp = np.sin(angle)
            ax = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R = cp * np.eye(3) + (1 - cp) * np.outer(axis, axis) + sp * ax
            J = (
                (sp / angle) * np.eye(3)
                + (1 - sp / angle) * np.outer(axis, axis)
                + ((1 - cp) / angle) * ax
            )

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = J @ rho
        return T

    def compute_trajectory_from_twists(self):
        """Integrate twists from the bottleneck pose to produce EEF pose sequence."""
        poses = [self.live_bottleneck_pose.copy()]
        current_pose = self.live_bottleneck_pose.copy()
        for twist in self.end_effector_twists:
            dt = 1.0 / 30.0
            T_inc = self.se3_exp(twist[:6] * dt)
            current_pose = current_pose @ T_inc
            poses.append(current_pose.copy())
        return np.array(poses[:-1])

    def run_with_live_data(
        self,
        speed_factor: float = 1.0,
        dry_run: bool = False,
        mode: str = "trajectory",
    ):
        """Replay from live bottleneck pose + twists."""
        _ = mode
        print("\nRunning replay with live data:")
        print(f"  Bottleneck pose: {self.live_bottleneck_pose.shape}")
        print(f"  Twists: {self.end_effector_twists.shape}")

        eef_poses = self.compute_trajectory_from_twists()
        self._set_speed(speed_factor)

        if len(eef_poses) < 2:
            print("Error: need at least 2 EEF poses from twists.")
            return

        start_pose = self.T_to_pose(eef_poses[0])
        eef_for_cartesian = eef_poses[1:]
        if not dry_run:
            self._gripper_command(open_gripper=True)
            if not self._move_to_start_pose(start_pose):
                print("  Continuing without start alignment.")
                eef_for_cartesian = eef_poses

        plan, fraction, _ = self._plan_cartesian(
            eef_for_cartesian, downsample=1, position_only=False,
        )
        if plan is None:
            print("Error: no poses to plan.")
            return

        if fraction < self.min_fraction:
            print("  Full-pose Cartesian path too short — trying position-only fallback...")
            plan, fraction, _ = self._plan_cartesian(
                eef_for_cartesian, downsample=2, position_only=True,
            )
            if plan is None:
                print("Error: no poses to plan in fallback.")
                return

        if fraction < self.min_fraction:
            print("  Cartesian planning below threshold — chunked fallback...")
            self._execute_chunked_cartesian(eef_for_cartesian, chunk_size=30, dry_run=dry_run)
            return

        self._execute_plan(plan, fraction, dry_run=dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay demonstrations via MoveIt Cartesian planning"
    )
    parser.add_argument("--demo_dir", "-d", type=str, required=True,
                        help="Path to demo directory (eef_poses.npy, gripper_states.npy, …)")
    parser.add_argument("--robot_model", type=str, default="wx250s")
    parser.add_argument("--robot_name", type=str, default="wx250s")
    parser.add_argument("--speed_factor", "-s", type=float, default=1.0,
                        help="Speed multiplier applied on top of base scale (default: 1.0)")
    parser.add_argument("--downsample", "-n", type=int, default=1,
                        help="Use every Nth waypoint (default: 1 = all waypoints)")
    parser.add_argument("--mode", "-m", type=str, default="trajectory",
                        choices=["trajectory", "point_by_point"],
                        help="Kept for compatibility; ignored in MoveIt-direct mode.")
    parser.add_argument("--dry_run", action="store_true", help="Plan only, do not execute")
    parser.add_argument("--eef_step", type=float, default=0.005,
                        help="Cartesian interpolation step in metres (default: 0.005)")
    parser.add_argument("--jump_threshold", type=float, default=0.0,
                        help="MoveIt jump threshold")
    parser.add_argument("--min_fraction", type=float, default=0.90,
                        help="Minimum Cartesian fraction required for execution")
    parser.add_argument("--avoid_collisions", action="store_true",
                        help="Enable collision checking during Cartesian path planning")
    parser.add_argument("--gripper_threshold", type=float, default=0.5,
                        help="Threshold to binarise gripper state (default: 0.5)")
    parser.add_argument("--base_vel_scale", type=float, default=0.4)
    parser.add_argument("--base_accel_scale", type=float, default=0.4)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DEMONSTRATION REPLAY (MOVEIT DIRECT)")
    print("=" * 60)
    print(f"Demo:          {args.demo_dir}")
    print(f"Speed factor:  {args.speed_factor}x")
    print(f"Downsample:    {args.downsample}x")
    print(f"EEF step:      {args.eef_step} m")
    print(f"Min fraction:  {args.min_fraction}")
    print(f"Avoid coll.:   {args.avoid_collisions}")
    print(f"Dry run:       {args.dry_run}")
    print("=" * 60)

    try:
        replayer = DemoReplayer(
            robot_model=args.robot_model,
            robot_name=args.robot_name,
            eef_step=args.eef_step,
            jump_threshold=args.jump_threshold,
            min_fraction=args.min_fraction,
            base_vel_scale=args.base_vel_scale,
            base_accel_scale=args.base_accel_scale,
            avoid_collisions=args.avoid_collisions,
            gripper_threshold=args.gripper_threshold,
        )
        replayer.run(
            demo_dir=args.demo_dir,
            speed_factor=args.speed_factor,
            downsample=args.downsample,
            dry_run=args.dry_run,
            mode=args.mode,
        )
    finally:
        roscpp_shutdown()


if __name__ == "__main__":
    main()