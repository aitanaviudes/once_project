"""
Demonstration replay using MoveIt Cartesian planning.

Flow:
  eef_poses.npy -> mat_to_pose() -> go(Sleep) -> go(start=pose[0])
  -> compute_cartesian_path(poses[1:]) -> execute()
"""
 
import argparse
import inspect
import os
import sys
 
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from tf.transformations import quaternion_from_matrix
 
 
def detect_sim_mode(robot_name: str) -> bool:
    """
    Detect whether we are in simulation or real-robot mode by checking
    which ROS topics are available.
 
    - Real robot (roslaunch interbotix_xsarm_control xsarm_control.launch):
        publishes /{robot_name}/joint_states via the xs_sdk hardware driver.
        The xs_sdk also advertises /{robot_name}/get_robot_info service.
 
    - Simulation (roslaunch interbotix_xsarm_descriptions xsarm_description.launch
        or the MoveIt fake-controllers launch):
        does NOT have get_robot_info, only publishes /joint_states via
        robot_state_publisher / fake joint driver.
 
    We check for the xs_sdk service as the distinguishing signal.
    """
    try:
        rospy.wait_for_service(f"/{robot_name}/get_robot_info", timeout=3.0)
        return False   # service found -> real robot
    except rospy.exceptions.ROSException:
        return True    # service not found -> simulation
 
 
def mat_to_pose(T: np.ndarray) -> Pose:
    """Convert a 4x4 SE(3) matrix to a geometry_msgs/Pose."""
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])
    q = quaternion_from_matrix(T)   # returns [x, y, z, w]
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


def pose_error(a: Pose, b: Pose):
    """Return position error [m] and orientation angular error [deg]."""
    dp = np.array([
        a.position.x - b.position.x,
        a.position.y - b.position.y,
        a.position.z - b.position.z,
    ])
    pos_err = float(np.linalg.norm(dp))

    # Relative quaternion angle
    qa = np.array([a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w], dtype=np.float64)
    qb = np.array([b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w], dtype=np.float64)
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    dot = float(np.abs(np.dot(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    ang_deg = float(np.degrees(2.0 * np.arccos(dot)))
    return pos_err, ang_deg


class DemoReplayer:
 
    def __init__(
        self,
        robot_model: str      = "wx250s",
        robot_name: str       = "wx250s",
        eef_step: float       = 0.005,  # Cartesian interpolation step (metres)
        jump_threshold: float = 5.0,    # reject IK solutions with large joint jumps
        min_fraction: float   = 0.95,   # minimum % of path that must be planned
        vel_scale: float      = 0.3,    # velocity scaling  (0, 1]
        accel_scale: float    = 0.3,    # acceleration scaling (0, 1]
    ):
        self.eef_step       = eef_step
        self.jump_threshold = jump_threshold
        self.min_fraction   = min_fraction
        self.start_idx      = 0
 
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
            print("Start MoveIt first, e.g.:")
            print(f"  roslaunch interbotix_xsarm_moveit xsarm_moveit.launch "
                  f"robot_model:={robot_model} use_fake:=true")
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

    def _plan_succeeds(self, plan_result) -> bool:
        """Handle MoveIt API differences for plan() return values."""
        if isinstance(plan_result, tuple):
            return bool(plan_result[0])
        return len(plan_result.joint_trajectory.points) > 0

    def _probe_ik(self, waypoints, count=15):
        """Quick per-waypoint IK probe to localize failures early."""
        n = min(count, len(waypoints))
        print(f"\nIK probe on first {n} waypoints:")
        first_fail = None
        for i in range(n):
            self.mg.set_pose_target(waypoints[i])
            ok = self._plan_succeeds(self.mg.plan())
            wp = waypoints[i]
            print(
                f"  waypoint[{i:03d}] x={wp.position.x:.4f} y={wp.position.y:.4f} z={wp.position.z:.4f} "
                f"-> IK {'OK' if ok else 'FAILED'}"
            )
            self.mg.clear_pose_targets()
            if (not ok) and first_fail is None:
                first_fail = i
        if first_fail is not None:
            print(f"  First IK probe failure at waypoint[{first_fail}]")
        else:
            print("  IK probe passed for sampled waypoints.")

    def _compute_cartesian(self, waypoints):
        """Compute Cartesian path with API-compatible argument dispatch."""
        param_names = list(inspect.signature(self.mg.compute_cartesian_path).parameters.keys())
        third_param = param_names[2] if len(param_names) >= 3 else ""

        if third_param == "jump_threshold":
            plan, fraction = self.mg.compute_cartesian_path(
                waypoints,
                self.eef_step,
                self.jump_threshold,
            )
            print(f"Cartesian API   : jump_threshold={self.jump_threshold}")
            return plan, fraction

        # Most common in Noetic Python wrapper: third arg is avoid_collisions (bool).
        plan, fraction = self.mg.compute_cartesian_path(
            waypoints,
            self.eef_step,
            True,
        )
        if abs(float(self.jump_threshold)) > 1e-12:
            print(
                "Cartesian API   : third arg is avoid_collisions; "
                "--jump_threshold is not exposed by this MoveIt Python binding."
            )
        else:
            print("Cartesian API   : avoid_collisions=True")
        return plan, fraction

    def run(self, demo_dir: str, dry_run: bool = False):

        print("Named targets:", self.mg.get_named_targets())
        
        # 1. Load the recorded end-effector poses
        path = os.path.join(demo_dir, "eef_poses.npy")
        eef_poses = np.load(path)   # shape: (T, 4, 4)
        print(f"\nLoaded {len(eef_poses)} poses from {path}")
 
        if len(eef_poses) < 2:
            print("Error: need at least 2 poses.")
            return
 
        # 2. Convert each 4x4 matrix to a Cartesian waypoint
        waypoints = [mat_to_pose(T) for T in eef_poses]
        if self.start_idx > 0:
            if self.start_idx >= len(waypoints) - 1:
                print(
                    f"Error: start_idx={self.start_idx} leaves fewer than 2 waypoints "
                    f"(total={len(waypoints)})."
                )
                return
            print(f"Applying start_idx={self.start_idx}: skipping initial waypoints.")
            waypoints = waypoints[self.start_idx:]
            print(f"Remaining waypoints: {len(waypoints)}")
 
 
        # 3. Move to the first pose using normal joint-space planning.
        #
        #    This is critical: compute_cartesian_path() assumes the robot is
        #    already sitting exactly at waypoints[0]. If we skip this step and
        #    the arm is somewhere else, the very first movement will be a large
        #    uncontrolled jump to catch up.
        #
        #    We use a regular go() for this — MoveIt will find a safe joint-space
        #    path to reach the start pose regardless of where the arm currently is.
 
        current_pose = self.mg.get_current_pose().pose
        print("Current EEF pose:")
        print(f"  x={current_pose.position.x:.4f}, y={current_pose.position.y:.4f}, z={current_pose.position.z:.4f}")
        print("Target start pose:")
        print(f"  x={waypoints[0].position.x:.4f}, y={waypoints[0].position.y:.4f}, z={waypoints[0].position.z:.4f}")
 
        self._probe_ik(waypoints, count=15)

        # 4. Move to start pose via Sleep as an intermediate waypoint
        print("\nMoving to Sleep pose first (intermediate)...")
        if not dry_run:
            self.mg.set_named_target("Sleep")
            success = self.mg.go(wait=True)
            sleep_pose = self.mg.get_current_pose().pose
            print(f"Sleep EEF pose: x={sleep_pose.position.x:.4f}, y={sleep_pose.position.y:.4f}, z={sleep_pose.position.z:.4f}")
            print(f"Sleep orientation: x={sleep_pose.orientation.x:.4f}, y={sleep_pose.orientation.y:.4f}, z={sleep_pose.orientation.z:.4f}, w={sleep_pose.orientation.w:.4f}")
            print(f"waypoints[0] orientation: x={waypoints[0].orientation.x:.4f}, y={waypoints[0].orientation.y:.4f}, z={waypoints[0].orientation.z:.4f}, w={waypoints[0].orientation.w:.4f}")
            self.mg.stop()
            self.mg.clear_pose_targets()
            if not success:
                print("Error: could not reach Sleep pose. Aborting.")
                return
            print("  Reached Sleep pose.")

            print("\nMoving to recorded start pose (waypoints[0])...")
            self.mg.set_pose_target(waypoints[0])
            success = self.mg.go(wait=True)
            self.mg.stop()
            self.mg.clear_pose_targets()
            if not success:
                print("Error: could not reach waypoints[0]. Aborting.")
                return

            reached = self.mg.get_current_pose().pose
            pos_err, ang_err = pose_error(reached, waypoints[0])
            print(
                "  Start-pose error after move: "
                f"pos={pos_err:.4f} m, ori={ang_err:.2f} deg"
            )
            if pos_err > 0.015 or ang_err > 10.0:
                print("  Warning: start-pose mismatch is large; Cartesian fraction may be poor.")

        # 5. Plan the Cartesian path for the remaining poses (pose[1] onward).
        #    MoveIt runs IK internally for each segment.
        print("\nPlanning Cartesian path...")
        cart_waypoints = waypoints[1:]
        self.mg.set_start_state_to_current_state()
        plan, fraction = self._compute_cartesian(cart_waypoints)
        print(f"Cartesian fraction : {fraction * 100:.1f}%")
        print(f"Trajectory points  : {len(plan.joint_trajectory.points)}")
        print(f"Cartesian waypoints: {len(cart_waypoints)}")
 
        if fraction < self.min_fraction:
            print(f"Fraction too low ({fraction:.2f} < {self.min_fraction}). Aborting.")
            return
 
        # 6. Execute
        if dry_run:
            print("[DRY RUN] Plan looks good. Not executing.")
            return
 
        print("Executing...")
        # print the IK data here
        self.mg.execute(plan, wait=True)
        self.mg.stop()
        self.mg.clear_pose_targets()
        print("Done.")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Replay a demonstration via MoveIt Cartesian IK"
    )
    parser.add_argument("--demo_dir",        "-d", required=True,
                        help="Directory containing eef_poses.npy")
    parser.add_argument("--robot_model",     default="wx250s")
    parser.add_argument("--robot_name",      default="wx250s")
    parser.add_argument("--eef_step",        type=float, default=0.005,
                        help="Cartesian interpolation step in metres (default: 0.005)")
    parser.add_argument("--jump_threshold",  type=float, default=5.0,
                        help="Reject IK solutions with joint jumps above this factor (default: 5.0)")
    parser.add_argument("--min_fraction",    type=float, default=0.95,
                        help="Minimum planned fraction before executing (default: 0.95)")
    parser.add_argument("--vel_scale",       type=float, default=0.3)
    parser.add_argument("--accel_scale",     type=float, default=0.3)
    parser.add_argument("--dry_run",         action="store_true",
                        help="Plan only, do not execute")
    parser.add_argument("--start_idx",       type=int, default=0,
                        help="Skip the first N recorded waypoints (default: 0)")
    args = parser.parse_args()
 
    print("\n" + "=" * 50)
    print("DEMONSTRATION REPLAY  (MoveIt Cartesian IK)")
    print("=" * 50)
 
    try:
        replayer = DemoReplayer(
            robot_model    = args.robot_model,
            robot_name     = args.robot_name,
            eef_step       = args.eef_step,
            jump_threshold = args.jump_threshold,
            min_fraction   = args.min_fraction,
            vel_scale      = args.vel_scale,
            accel_scale    = args.accel_scale,
        )
        replayer.start_idx = max(0, int(args.start_idx))
        replayer.run(
            demo_dir = args.demo_dir,
            dry_run  = args.dry_run,
        )
    finally:
        roscpp_shutdown()
 
 
if __name__ == "__main__":
    main()
