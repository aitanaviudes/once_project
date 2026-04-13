#!/usr/bin/env python3

"""
Demonstration replay using MoveIt Cartesian planning.

Flow:
  eef_poses.npy  ->  mat_to_pose()  ->  go(Sleep)  ->  compute_cartesian_path()  ->  execute()
  (T x 4x4 SE3)      (Cartesian        (joint-space    (IK runs internally           (joints
                       waypoints)        move to          for all waypoints)            move)
                                         Sleep pose,
                                         which is ~waypoints[0])
"""

import argparse
import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from tf.transformations import quaternion_from_matrix


def detect_sim_mode(robot_name: str) -> bool:
    """
    Detect simulation vs real robot.

    Real robot (roslaunch interbotix_xsarm_control xsarm_control.launch):
      xs_sdk hardware driver advertises /{robot_name}/get_robot_info.

    Simulation (roslaunch interbotix_xsarm_moveit xsarm_moveit.launch use_fake:=true dof:=6):
      No hardware driver, so get_robot_info is not available.
    """
    try:
        rospy.wait_for_service(f"/{robot_name}/get_robot_info", timeout=3.0)
        return False  # service found -> real robot
    except rospy.exceptions.ROSException:
        return True   # service not found -> simulation


def mat_to_pose(T: np.ndarray) -> Pose:
    """Convert a 4x4 SE(3) matrix to a geometry_msgs/Pose."""
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])
    q = quaternion_from_matrix(T)  # returns [x, y, z, w]
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


class DemoReplayer:

    def __init__(
        self,
        robot_model: str  = "wx250s",
        robot_name: str   = "wx250s",
        eef_step: float   = 0.005,  # Cartesian interpolation step (metres)
        min_fraction: float = 0.95, # minimum % of path that must be planned
        vel_scale: float  = 0.3,    # velocity scaling (0, 1]
        accel_scale: float = 0.3,   # acceleration scaling (0, 1]
    ):
        self.eef_step     = eef_step
        self.min_fraction = min_fraction

        # ROS must be initialised before any rospy calls (including wait_for_service)
        rospy.init_node("demo_replayer_moveit", anonymous=True)
        roscpp_initialize(sys.argv)

        # Detect mode AFTER rospy.init_node
        self.sim_mode = detect_sim_mode(robot_name)
        print("Mode:", "SIMULATION" if self.sim_mode else "REAL ROBOT")

        # Verify MoveIt is running
        semantic_ns = f"/{robot_name}/robot_description_semantic"
        if not rospy.has_param(semantic_ns):
            print(f"\nError: MoveIt not running (missing {semantic_ns})")
            print("Start MoveIt first:")
            print(f"  roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                  f"robot_model:={robot_model} use_fake:=true dof:=6")
            sys.exit(1)

        self.mg = MoveGroupCommander(
            "interbotix_arm",
            robot_description=f"/{robot_name}/robot_description",
            ns=f"/{robot_name}",
        )
        self.mg.set_max_velocity_scaling_factor(vel_scale)
        self.mg.set_max_acceleration_scaling_factor(accel_scale)
        self.mg.set_planning_time(10.0)

        print(f"Planning frame : {self.mg.get_planning_frame()}")
        print(f"End-effector   : {self.mg.get_end_effector_link()}")

    def run(self, demo_dir: str, dry_run: bool = False):

        # 1. Load the recorded end-effector poses
        path = os.path.join(demo_dir, "eef_poses.npy")
        eef_poses = np.load(path)  # shape: (T, 4, 4)
        print(f"\nLoaded {len(eef_poses)} poses from {path}")

        if len(eef_poses) < 2:
            print("Error: need at least 2 poses.")
            return

        # 2. Convert each 4x4 matrix to a Cartesian waypoint
        waypoints = [mat_to_pose(T) for T in eef_poses]

        # Debug: test IK on first few waypoints individually
        print("\nTesting IK on first 5 waypoints individually:")
        for i in range(min(5, len(waypoints))):
            self.mg.set_pose_target(waypoints[i])
            plan = self.mg.plan()
            # plan() returns (success, plan, time, error) in newer MoveIt, or just plan in older
            if isinstance(plan, tuple):
                success = plan[0]
            else:
                success = len(plan.joint_trajectory.points) > 0
            wp = waypoints[i]
            print(f"  waypoint[{i}]: x={wp.position.x:.4f} y={wp.position.y:.4f} z={wp.position.z:.4f} "
                f"-> IK {'OK' if success else 'FAILED'}")
            self.mg.clear_pose_targets()

        # 3. Move to Sleep pose.
        #
        #    The demo was recorded starting from Sleep, and Sleep (~x=0.12, z=0.08)
        #    is essentially the same position as waypoints[0]. We use the named
        #    target instead of set_pose_target(waypoints[0]) because:
        #      - Sleep is a known joint configuration, so MoveIt plans it reliably
        #      - set_pose_target from the Upright position (all joints=0) times out
        #        because OMPL struggles with the large arm-down motion
        print("\nMoving to Sleep pose...")
        if not dry_run:
            self.mg.set_named_target("Sleep")
            success = self.mg.go(wait=True)
            self.mg.stop()
            self.mg.clear_pose_targets()
            if not success:
                print("Error: could not reach Sleep pose. Aborting.")
                return
            print("  Reached Sleep pose.")

        # 4. Plan Cartesian path for ALL waypoints from current state.
        #
        #    We pass all waypoints (including waypoints[0]) because the arm is
        #    at Sleep which is ~5mm from waypoints[0] — MoveIt handles this tiny
        #    gap in the first interpolation step. IK is solved internally at each
        #    waypoint along the path.
        print("\nPlanning Cartesian path...")
        self.mg.set_start_state_to_current_state()
        plan, fraction = self.mg.compute_cartesian_path(
            waypoints,      # all recorded poses
            self.eef_step,  # interpolation step in metres
            True,           # avoid_collisions
        )
        print(f"Cartesian fraction : {fraction * 100:.1f}%")
        print(f"Trajectory points  : {len(plan.joint_trajectory.points)}")

        if fraction < self.min_fraction:
            print(f"Fraction too low ({fraction:.2f} < {self.min_fraction}). Aborting.")
            return

        # 5. Execute
        if dry_run:
            print("[DRY RUN] Plan looks good. Not executing.")
            return

        print("Executing...")
        self.mg.execute(plan, wait=True)
        self.mg.stop()
        self.mg.clear_pose_targets()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay a demonstration via MoveIt Cartesian IK"
    )
    parser.add_argument("--demo_dir",      "-d", required=True,
                        help="Directory containing eef_poses.npy")
    parser.add_argument("--robot_model",   default="wx250s")
    parser.add_argument("--robot_name",    default="wx250s")
    parser.add_argument("--eef_step",      type=float, default=0.005,
                        help="Cartesian interpolation step in metres (default: 0.005)")
    parser.add_argument("--min_fraction",  type=float, default=0.95,
                        help="Minimum planned fraction before executing (default: 0.95)")
    parser.add_argument("--vel_scale",     type=float, default=0.3)
    parser.add_argument("--accel_scale",   type=float, default=0.3)
    parser.add_argument("--dry_run",       action="store_true",
                        help="Plan only, do not execute")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("DEMONSTRATION REPLAY  (MoveIt Cartesian IK)")
    print("=" * 50)

    try:
        replayer = DemoReplayer(
            robot_model  = args.robot_model,
            robot_name   = args.robot_name,
            eef_step     = args.eef_step,
            min_fraction = args.min_fraction,
            vel_scale    = args.vel_scale,
            accel_scale  = args.accel_scale,
        )
        replayer.run(
            demo_dir = args.demo_dir,
            dry_run  = args.dry_run,
        )
    finally:
        roscpp_shutdown()


if __name__ == "__main__":
    main()