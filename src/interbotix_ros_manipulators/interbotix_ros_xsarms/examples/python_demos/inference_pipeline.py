#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

import cv2
import moveit_commander
import numpy as np
import rospy
import tf.transformations as tf_trans
import trajectory_msgs.msg
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo, Image
from test_sam import mobile_sam_segmap_function
from tf.transformations import quaternion_from_matrix


LEARNING_TASKS_DIR = Path("/home/aitana_viudes/1000_tasks/learning_thousand_tasks")
INFERENCE_EXAMPLE_DIR = LEARNING_TASKS_DIR / "assets" / "inference_example"
SAVED_DATA_DIR = LEARNING_TASKS_DIR / "saved_data"

CAPTURE_MAX_ATTEMPTS = 3
CAPTURE_SEG_MIN_AREA_PX = 500
CAPTURE_SEG_MAX_AREA_RATIO = 0.08
CAPTURE_SEG_MAX_ASPECT_RATIO = 3.5
CAPTURE_DEPTH_MIN_MM = 250
CAPTURE_DEPTH_MAX_MM = 3500
CAPTURE_DEPTH_VALID_RATIO_MIN = 0.7
CAPTURE_DEPTH_SPREAD_MAX_MM = 350
CAPTURE_DEPTH_BAND_MM = 320
MOBILE_SAM_POINT = (300, 200)


def ensure_ros_node_initialized():
    if not rospy.core.is_initialized():
        rospy.init_node("inference_pipeline", anonymous=True)


def to_uint16_depth_mm(depth_image):
    if depth_image.dtype == np.uint16:
        return depth_image

    depth_float = depth_image.astype(np.float32)
    depth_float[~np.isfinite(depth_float)] = 0.0
    max_depth = float(np.max(depth_float)) if depth_float.size else 0.0

    if max_depth <= 20.0:
        depth_mm = np.rint(depth_float * 1000.0)
    else:
        depth_mm = np.rint(depth_float)

    return np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def refine_object_depth(depth_image, segmap, depth_band_mm=40):
    if not segmap.any():
        return depth_image

    refined = depth_image.copy()
    obj_valid = segmap & (refined > 0)
    if not obj_valid.any():
        return refined

    obj_depths = refined[obj_valid]
    median_depth = int(np.median(obj_depths))
    low = max(1, median_depth - depth_band_mm)
    high = median_depth + depth_band_mm

    # Only clamp valid outliers; do not hallucinate missing depth by filling zeros.
    outlier = segmap & (refined > 0) & ((refined < low) | (refined > high))
    refined[outlier] = np.uint16(median_depth)

    ys, xs = np.where(segmap)
    if ys.size == 0:
        return refined

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    roi = refined[y0:y1 + 1, x0:x1 + 1]
    roi_mask = segmap[y0:y1 + 1, x0:x1 + 1]

    if roi.shape[0] >= 5 and roi.shape[1] >= 5:
        roi_blur = cv2.medianBlur(roi, 5)
        roi[roi_mask] = roi_blur[roi_mask]
        refined[y0:y1 + 1, x0:x1 + 1] = roi

    return refined


def extract_primary_segmentation(segmap, preferred_point=None):
    mask = np.asarray(segmap, dtype=bool)
    if not mask.any():
        return mask

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = mask_u8 > 0
    if not mask.any():
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask

    target_label = None
    if preferred_point is not None:
        px, py = preferred_point
        if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]:
            candidate = labels[py, px]
            if candidate > 0:
                target_label = int(candidate)

    if target_label is None:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        target_label = int(np.argmax(component_areas) + 1)
    clean_mask = labels == target_label
    clean_mask_u8 = (clean_mask.astype(np.uint8) * 255)
    clean_mask_u8 = cv2.morphologyEx(clean_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return clean_mask_u8 > 0


def constrain_segmap_to_depth(segmap, depth_image, preferred_point=None):
    depth_valid = (depth_image > 0) & (depth_image < 65000)
    constrained = np.asarray(segmap, dtype=bool) & depth_valid
    if not constrained.any():
        return constrained

    constrained = extract_primary_segmentation(constrained, preferred_point=preferred_point)
    if not constrained.any():
        return constrained

    kernel = np.ones((3, 3), dtype=np.uint8)
    constrained_u8 = (constrained.astype(np.uint8) * 255)
    constrained_u8 = cv2.morphologyEx(constrained_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    constrained = constrained_u8 > 0

    anchor_depth = estimate_anchor_depth_mm(
        depth_image=depth_image,
        segmap=constrained,
        preferred_point=preferred_point,
    )
    if anchor_depth is not None:
        depth_diff = np.abs(depth_image.astype(np.int32) - int(anchor_depth))
        near_depth = depth_diff <= int(CAPTURE_DEPTH_BAND_MM)
        depth_filtered = constrained & near_depth
        if depth_filtered.any():
            constrained = extract_primary_segmentation(
                depth_filtered,
                preferred_point=preferred_point,
            )
    constrained = trim_segmap_depth_spread(
        constrained,
        depth_image,
        preferred_point=preferred_point,
    )
    return constrained & depth_valid


def trim_segmap_depth_spread(segmap, depth_image, preferred_point=None):
    depth_valid = (depth_image > 0) & (depth_image < 65000)
    working = np.asarray(segmap, dtype=bool) & depth_valid
    vals = depth_image[working]
    if vals.size == 0:
        return working

    p10, p90 = np.percentile(vals, [10, 90])
    spread = float(p90 - p10)
    if spread <= CAPTURE_DEPTH_SPREAD_MAX_MM:
        return working

    center = float(np.median(vals))
    half_band = max(60, int(CAPTURE_DEPTH_SPREAD_MAX_MM * 0.55))
    depth_diff = np.abs(depth_image.astype(np.int32) - int(center))
    tight = working & (depth_diff <= half_band)
    if tight.any():
        tight = extract_primary_segmentation(
            tight,
            preferred_point=preferred_point,
        )
        if tight.any():
            return tight
    return working


def estimate_anchor_depth_mm(depth_image, segmap, preferred_point=None, window_radius=6):
    if preferred_point is not None:
        px, py = preferred_point
        if 0 <= px < depth_image.shape[1] and 0 <= py < depth_image.shape[0]:
            y0 = max(0, py - window_radius)
            y1 = min(depth_image.shape[0], py + window_radius + 1)
            x0 = max(0, px - window_radius)
            x1 = min(depth_image.shape[1], px + window_radius + 1)
            patch_depth = depth_image[y0:y1, x0:x1]
            patch_seg = segmap[y0:y1, x0:x1]
            patch_vals = patch_depth[patch_seg & (patch_depth > 0) & (patch_depth < 65000)]
            if patch_vals.size > 0:
                return float(np.percentile(patch_vals, 35))

    vals = depth_image[segmap & (depth_image > 0) & (depth_image < 65000)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, 30))


def validate_capture_quality(segmap, depth_image):
    total_px = segmap.size
    seg_area = int(np.count_nonzero(segmap))
    max_area = int(total_px * CAPTURE_SEG_MAX_AREA_RATIO)
    if seg_area < CAPTURE_SEG_MIN_AREA_PX:
        return False, f"segmentation too small ({seg_area} px < {CAPTURE_SEG_MIN_AREA_PX} px)", {}
    if seg_area > max_area:
        return False, f"segmentation too large ({seg_area} px > {max_area} px)", {}

    ys, xs = np.where(segmap)
    if ys.size == 0:
        return False, "segmentation is empty", {}
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    aspect_ratio = max(bbox_h / max(1, bbox_w), bbox_w / max(1, bbox_h))
    if aspect_ratio > CAPTURE_SEG_MAX_ASPECT_RATIO:
        return False, f"segmentation shape too elongated (aspect={aspect_ratio:.2f})", {}

    depth_values = depth_image[segmap]
    valid_depths = depth_values[(depth_values > 0) & (depth_values < 65000)]
    if depth_values.size == 0:
        return False, "no depth values under segmentation", {}

    valid_ratio = float(valid_depths.size) / float(depth_values.size)
    if valid_ratio < CAPTURE_DEPTH_VALID_RATIO_MIN:
        return False, (
            f"low valid-depth ratio ({valid_ratio:.2f} < {CAPTURE_DEPTH_VALID_RATIO_MIN:.2f})"
        ), {}

    median_depth = float(np.median(valid_depths))
    if median_depth < CAPTURE_DEPTH_MIN_MM or median_depth > CAPTURE_DEPTH_MAX_MM:
        return False, (
            f"depth median out of range ({median_depth:.1f} mm not in "
            f"[{CAPTURE_DEPTH_MIN_MM}, {CAPTURE_DEPTH_MAX_MM}] mm)"
        ), {}

    p10, p90 = np.percentile(valid_depths, [10, 90])
    depth_spread = float(p90 - p10)
    if depth_spread > CAPTURE_DEPTH_SPREAD_MAX_MM:
        return False, (
            f"depth spread too high ({depth_spread:.1f} mm > {CAPTURE_DEPTH_SPREAD_MAX_MM} mm)"
        ), {}

    stats = {
        "seg_area_px": seg_area,
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "depth_median_mm": median_depth,
        "depth_spread_mm": depth_spread,
        "valid_depth_ratio": valid_ratio,
    }
    return True, "ok", stats


def capture_workspace_camera_data(timeout=20.0):
    bridge = CvBridge()
    for attempt in range(1, CAPTURE_MAX_ATTEMPTS + 1):
        try:
            rgb_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=timeout)
            depth_msg = rospy.wait_for_message(
                "/camera/aligned_depth_to_color/image_raw",
                Image,
                timeout=timeout,
            )
            camera_info_msg = rospy.wait_for_message(
                "/camera/color/camera_info",
                CameraInfo,
                timeout=timeout,
            )
        except rospy.ROSException as exc:
            raise RuntimeError(f"Failed to capture camera data: {exc}") from exc

        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_image = to_uint16_depth_mm(depth_image)

        intrinsic_matrix = np.array(camera_info_msg.K, dtype=np.float64).reshape(3, 3)
        rgb_for_mobile_sam = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        raw_segmap = mobile_sam_segmap_function(
            rgb_for_mobile_sam,
            point_x=MOBILE_SAM_POINT[0],
            point_y=MOBILE_SAM_POINT[1],
        )
        segmap = extract_primary_segmentation(raw_segmap, preferred_point=MOBILE_SAM_POINT)
        segmap = constrain_segmap_to_depth(segmap, depth_image, preferred_point=MOBILE_SAM_POINT)
        capture_ok, reason, stats = validate_capture_quality(segmap, depth_image)
        if not capture_ok:
            rospy.logwarn(
                f"Capture attempt {attempt}/{CAPTURE_MAX_ATTEMPTS} rejected: {reason}. Retrying..."
            )
            continue

        depth_image = refine_object_depth(depth_image, segmap)
        print(
            f"Capture accepted: seg_area={stats['seg_area_px']} px, "
            f"bbox={stats['bbox_h']}x{stats['bbox_w']}, "
            f"depth_median={int(stats['depth_median_mm'])} mm"
        )
        return {
            "rgb_image": rgb_image,
            "depth_image": depth_image,
            "segmap": segmap,
            "intrinsic_matrix": intrinsic_matrix,
        }

    raise RuntimeError(
        f"Failed to capture a valid camera snapshot after {CAPTURE_MAX_ATTEMPTS} attempts."
    )


def update_inference_example_assets(camera_data):
    INFERENCE_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    rgb_path = INFERENCE_EXAMPLE_DIR / "head_camera_ws_rgb.png"
    depth_path = INFERENCE_EXAMPLE_DIR / "head_camera_ws_depth_to_rgb.png"
    segmap_path = INFERENCE_EXAMPLE_DIR / "head_camera_ws_segmap.npy"
    intrinsics_path = INFERENCE_EXAMPLE_DIR / "head_camera_rgb_intrinsic_matrix.npy"

    if not cv2.imwrite(str(rgb_path), camera_data["rgb_image"]):
        raise RuntimeError(f"Failed to write {rgb_path}")
    if not cv2.imwrite(str(depth_path), camera_data["depth_image"]):
        raise RuntimeError(f"Failed to write {depth_path}")
    np.save(segmap_path, camera_data["segmap"])
    np.save(intrinsics_path, camera_data["intrinsic_matrix"])

    print("Updated inference_example assets:")
    print(f"  - {rgb_path}")
    print(f"  - {depth_path}")
    print(f"  - {segmap_path}")
    print(f"  - {intrinsics_path}")


def run_mt3_deployment():
    print(f"Running 'make deploy_mt3' in {LEARNING_TASKS_DIR}")
    env = os.environ.copy()
    # Makefile uses ${PWD} for the /workspace bind mount; force it to the repo root.
    env["PWD"] = str(LEARNING_TASKS_DIR)
    subprocess.run(
        ["make", "deploy_mt3"],
        cwd=str(LEARNING_TASKS_DIR),
        env=env,
        check=True,
    )


def load_saved_mt3_outputs():
    bottleneck_path = SAVED_DATA_DIR / "live_bottleneck_pose.npy"
    twists_path = SAVED_DATA_DIR / "end_effector_twists.npy"

    if not bottleneck_path.exists():
        raise FileNotFoundError(f"Missing MT3 output: {bottleneck_path}")
    if not twists_path.exists():
        raise FileNotFoundError(f"Missing MT3 output: {twists_path}")

    live_bottleneck_pose = np.load(bottleneck_path)
    end_effector_twists = np.load(twists_path)

    if live_bottleneck_pose.shape == (1, 4, 4):
        live_bottleneck_pose = live_bottleneck_pose[0]
    if live_bottleneck_pose.shape != (4, 4):
        raise ValueError(
            f"live_bottleneck_pose has unexpected shape {live_bottleneck_pose.shape}, expected (4, 4)"
        )

    if end_effector_twists.ndim == 1:
        end_effector_twists = end_effector_twists.reshape(1, -1)
    if end_effector_twists.ndim != 2 or end_effector_twists.shape[1] < 6:
        raise ValueError(
            "end_effector_twists must have shape (T, 6+) with [vx, vy, vz, wx, wy, wz, ...]"
        )

    print(f"Loaded live_bottleneck_pose: {live_bottleneck_pose.shape}")
    print(f"Loaded end_effector_twists: {end_effector_twists.shape}")
    return live_bottleneck_pose, end_effector_twists


class VelocityTrajectoryReplayer:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        ensure_ros_node_initialized()

        self.robot_ns = "/wx250s"
        robot_desc = self.robot_ns + "/robot_description"

        self.robot = moveit_commander.RobotCommander(
            robot_description=robot_desc,
            ns=self.robot_ns,
        )
        self.scene = moveit_commander.PlanningSceneInterface(ns=self.robot_ns)
        self.move_group = moveit_commander.MoveGroupCommander(
            "interbotix_arm",
            robot_description=robot_desc,
            ns=self.robot_ns,
        )
        self.gripper_group = moveit_commander.MoveGroupCommander(
            "interbotix_gripper",
            robot_description=robot_desc,
            ns=self.robot_ns,
        )

    def mat_to_pose(self, transform):
        pose = Pose()
        pose.position.x = float(transform[0, 3])
        pose.position.y = float(transform[1, 3])
        pose.position.z = float(transform[2, 3])
        quaternion = quaternion_from_matrix(transform)
        pose.orientation.x = float(quaternion[0])
        pose.orientation.y = float(quaternion[1])
        pose.orientation.z = float(quaternion[2])
        pose.orientation.w = float(quaternion[3])
        return pose

    def open_gripper(self):
        print("Opening gripper...")
        self.gripper_group.set_named_target("Open")
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()

    def close_gripper(self):
        print("Closing gripper...")
        self.gripper_group.set_named_target("Closed")
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()

    def go_to_pose_goal(self, pose_goal):
        self.move_group.set_pose_reference_frame(self.robot_ns + "/base_link")
        self.move_group.set_planning_time(15.0)
        self.move_group.set_goal_position_tolerance(0.01)
        self.move_group.set_goal_orientation_tolerance(0.05)
        self.move_group.set_pose_target(pose_goal)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def execute_true_velocity_phase(self, twists, hz=30.0):
        print("Starting velocity trajectory replay...")
        traj_pub = rospy.Publisher(
            self.robot_ns + "/arm_controller/command",
            trajectory_msgs.msg.JointTrajectory,
            queue_size=10,
        )

        rate = rospy.Rate(hz)
        dt = 1.0 / hz

        joint_names = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ]

        current_joints = np.array(self.move_group.get_current_joint_values(), dtype=np.float64)
        gripper_closed = False

        for twist in twists:
            if twist.shape[0] >= 7:
                target_gripper_state = float(twist[6])
                if target_gripper_state == 1.0 and not gripper_closed:
                    self.close_gripper()
                    gripper_closed = True
                    current_joints = np.array(self.move_group.get_current_joint_values(), dtype=np.float64)
                elif target_gripper_state == 0.0 and gripper_closed:
                    self.open_gripper()
                    gripper_closed = False
                    current_joints = np.array(self.move_group.get_current_joint_values(), dtype=np.float64)

            v_linear_ee = np.array([twist[0], twist[1], twist[2]], dtype=np.float64)
            v_angular_ee = np.array([twist[3], twist[4], twist[5]], dtype=np.float64)

            current_pose = self.move_group.get_current_pose().pose
            quaternion = [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ]
            rotation = tf_trans.quaternion_matrix(quaternion)[:3, :3]

            v_linear_base = np.dot(rotation, v_linear_ee)
            v_angular_base = np.dot(rotation, v_angular_ee)
            v_target_base = np.concatenate((v_linear_base, v_angular_base))

            jacobian = np.array(self.move_group.get_jacobian_matrix(current_joints.tolist()))
            jacobian_inverse = np.linalg.pinv(jacobian, rcond=1e-2)
            q_dot = np.dot(jacobian_inverse, v_target_base)
            current_joints = current_joints + (q_dot * dt)

            msg = trajectory_msgs.msg.JointTrajectory()
            msg.joint_names = joint_names

            point = trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = current_joints.tolist()
            point.velocities = q_dot.tolist()
            point.time_from_start = rospy.Duration(dt)
            msg.points.append(point)

            traj_pub.publish(msg)
            rate.sleep()


def replay_saved_trajectory(live_bottleneck_pose, end_effector_twists):
    replayer = VelocityTrajectoryReplayer()

    print("=== PHASE 0: RESET TO HOME ===")
    replayer.move_group.set_named_target("Home")
    replayer.move_group.go(wait=True)
    replayer.move_group.stop()

    print("=== PHASE 1: ALIGNMENT ===")
    bottleneck_pose = replayer.mat_to_pose(live_bottleneck_pose)
    replayer.go_to_pose_goal(bottleneck_pose)

    print("=== PHASE 3: INTERACTION ===")
    replayer.execute_true_velocity_phase(end_effector_twists)
    print("Trajectory execution complete.")


def main():
    ensure_ros_node_initialized()

    print("Capturing live workspace camera data...")
    camera_data = capture_workspace_camera_data()
    update_inference_example_assets(camera_data)

    run_mt3_deployment()
    live_bottleneck_pose, end_effector_twists = load_saved_mt3_outputs()

    answer = input(
        "Proceed with execution of the trajectory? Type 'yes' to continue: "
    ).strip().lower()
    if answer != "yes":
        print("Trajectory execution cancelled.")
        return

    replay_saved_trajectory(live_bottleneck_pose, end_effector_twists)


if __name__ == "__main__":
    main()
