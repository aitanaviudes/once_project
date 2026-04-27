#!/usr/bin/env python3

"""
Inference pipeline:
  1) Press 's' to capture camera inputs (with MobileSAM GUI point selection).
  2) Save as collected_demos/session_<timestamp>/demo_0000.
  3) Run collected_demos/update_inference_only2.sh to refresh MT3 inference assets.
  4) Run `make deploy_mt3` in learning_thousand_tasks.
  5) Replay using call_replay_live_with_saved_data.py.

Press 'q' to quit.
"""

import argparse
import select
import shlex
import subprocess
import sys
import termios
import time
import traceback
import tty
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from test_sam import mobile_sam_segmap_function


SIM_REPLAY_LAUNCH = (
    "roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch "
    "robot_model:=wx250s use_gazebo:=true dof:=6 use_python_interface:=false gui:=false"
)
REAL_REPLAY_LAUNCH = (
    "roslaunch interbotix_xsarm_moveit_interface xsarm_moveit_interface.launch "
    "robot_model:=wx250s use_actual:=true dof:=6 use_python_interface:=false gui:=false"
)
COLLECT_LAUNCH = (
    "roslaunch interbotix_xsarm_control xsarm_control.launch "
    "robot_model:=wx250s use_rviz:=true"
)


class InferencePipeline:
    def __init__(self, task_name: str, replay_mode: str):
        self.task_name = task_name
        self.replay_mode = replay_mode
        self.cv_bridge = CvBridge()
        self.mobile_sam_point = None
        self.object_depth_band_mm = 40
        self.save_segmentation_debug = True

        self.this_dir = Path(__file__).resolve().parent
        self.collected_demos_dir = self.this_dir / "collected_demos"
        self.update_script = self.collected_demos_dir / "update_inference_only2.sh"
        self.replay_script = self.this_dir / "call_replay_live_with_saved_data.py"
        self.mt3_root = Path("/home/aitana_viudes/1000_tasks/learning_thousand_tasks")

        self.collected_demos_dir.mkdir(parents=True, exist_ok=True)

        rospy.init_node("mt3_inference_pipeline", anonymous=True)

    def print_launch_instructions(self):
        print("\nRequired launch setup before using this script:")
        print(f"  Capture/control launch: {COLLECT_LAUNCH}")
        print("  Replay launch (pick one):")
        print(f"    Simulation: {SIM_REPLAY_LAUNCH}")
        print(f"    Reality:    {REAL_REPLAY_LAUNCH}")
        chosen = SIM_REPLAY_LAUNCH if self.replay_mode == "sim" else REAL_REPLAY_LAUNCH
        print(f"  Selected replay mode for this run: {self.replay_mode}")
        print(f"  Start this one in another terminal: {chosen}")

    @staticmethod
    def _to_uint16_depth_mm(depth_image: np.ndarray) -> np.ndarray:
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

    def mobile_sam_segmap(self, rgb_image: np.ndarray, point=None) -> np.ndarray:
        if point is None:
            return mobile_sam_segmap_function(rgb_image)
        point_x, point_y = point
        return mobile_sam_segmap_function(rgb_image, point_x=point_x, point_y=point_y)

    def _refine_object_depth(self, depth_image: np.ndarray, segmap: np.ndarray) -> np.ndarray:
        if not segmap.any():
            return depth_image

        refined = depth_image.copy()
        obj_valid = segmap & (refined > 0)
        if not obj_valid.any():
            return refined

        obj_depths = refined[obj_valid]
        median_depth = int(np.median(obj_depths))
        low = max(1, median_depth - self.object_depth_band_mm)
        high = median_depth + self.object_depth_band_mm

        outlier_or_missing = segmap & ((refined == 0) | (refined < low) | (refined > high))
        refined[outlier_or_missing] = np.uint16(median_depth)

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

    def _select_mobile_sam_point(self, rgb_image: np.ndarray, timeout: float = 60.0):
        state = {"point": self.mobile_sam_point, "clicked": False}
        window_name = "Select Cube Point For MobileSAM"

        def _mouse_callback(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["point"] = (int(x), int(y))
                state["clicked"] = True

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, _mouse_callback)
        except cv2.error as exc:
            rospy.logwarn(f"Could not open point-selection window: {exc}")
            return self.mobile_sam_point

        print(
            "Point selector: left-click cube location. "
            "Press Enter/Space to confirm current point, 'r' to clear, ESC to cancel."
        )

        start_time = time.time()
        confirmed = False

        try:
            while not rospy.is_shutdown():
                preview = rgb_image.copy()
                if state["point"] is not None:
                    x, y = state["point"]
                    cv2.drawMarker(
                        preview,
                        (x, y),
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=24,
                        thickness=2,
                    )
                    cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
                    cv2.putText(
                        preview,
                        f"Current point: ({x}, {y})",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        preview,
                        "Left-click to select cube point",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, preview)
                if state["clicked"]:
                    confirmed = True
                    break

                key = cv2.waitKey(30) & 0xFF
                if key in (13, 10, 32) and state["point"] is not None:
                    confirmed = True
                    break
                if key == ord("r"):
                    state["point"] = None
                    state["clicked"] = False
                if key == 27:
                    break
                if (time.time() - start_time) > timeout:
                    print("Point selection timed out.")
                    break
        finally:
            try:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

        if confirmed and state["point"] is not None:
            return state["point"]
        return self.mobile_sam_point

    def capture_workspace_camera_data(self, timeout: float = 20.0):
        try:
            rgb_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=timeout)
            depth_msg = rospy.wait_for_message(
                "/camera/aligned_depth_to_color/image_raw", Image, timeout=timeout
            )
            camera_info_msg = rospy.wait_for_message(
                "/camera/color/camera_info", CameraInfo, timeout=timeout
            )
        except rospy.ROSException as exc:
            rospy.logwarn(f"Failed to capture camera data: {exc}")
            return None

        try:
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_image = self._to_uint16_depth_mm(depth_image)
            intrinsic_matrix = np.array(camera_info_msg.K, dtype=np.float64).reshape(3, 3)

            selected_point = self._select_mobile_sam_point(rgb_image)
            if selected_point is not None:
                self.mobile_sam_point = selected_point
                print(f"Using MobileSAM point: ({selected_point[0]}, {selected_point[1]})")

            segmap = self.mobile_sam_segmap(rgb_image, point=self.mobile_sam_point)
            depth_image = self._refine_object_depth(depth_image, segmap)

            print(f"RGB SHAPE: {rgb_image.shape}")
            print(f"DEPTH SHAPE: {depth_image.shape}")
            print(f"SEGMAP SHAPE: {segmap.shape}, TRUE PIXELS: {np.count_nonzero(segmap)}")

            if self.save_segmentation_debug:
                debug_vis = rgb_image.copy()
                debug_vis[segmap] = (
                    0.35 * debug_vis[segmap] + 0.65 * np.array([0, 255, 0])
                ).astype(np.uint8)
                debug_path = self.collected_demos_dir / "latest_segmentation_debug.png"
                cv2.imwrite(str(debug_path), debug_vis)
                print(f"Saved segmentation debug overlay: {debug_path}")

            return {
                "rgb_image": rgb_image,
                "depth_image": depth_image,
                "segmap": segmap,
                "intrinsic_matrix": intrinsic_matrix,
            }
        except Exception as exc:
            rospy.logwarn(f"Error converting camera messages: {exc}")
            return None

    def save_session_demo(self, camera_data):
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        demo_dir = self.collected_demos_dir / session_name / "demo_0000"
        demo_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(demo_dir / "head_camera_ws_rgb.png"), camera_data["rgb_image"])
        cv2.imwrite(
            str(demo_dir / "head_camera_ws_depth_to_rgb.png"),
            camera_data["depth_image"],
        )
        np.save(demo_dir / "head_camera_ws_segmap.npy", camera_data["segmap"])
        np.save(
            demo_dir / "head_camera_rgb_intrinsic_matrix.npy",
            camera_data["intrinsic_matrix"],
        )
        with open(demo_dir / "task_name.txt", "w", encoding="utf-8") as file_obj:
            file_obj.write(self.task_name)

        print(f"Saved inference capture to: {demo_dir}")
        return session_name

    @staticmethod
    def _run_command(command, cwd: Path, input_text=None):
        print(f"\n[RUN] cwd={cwd}")
        print(f"[RUN] {' '.join(command)}")

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if input_text is not None:
            output, _ = process.communicate(input_text)
            if output:
                print(output, end="")
        else:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {command}")

    def refresh_inference_assets(self, session_name: str):
        script_quoted = shlex.quote(str(self.update_script))
        session_quoted = shlex.quote(session_name)
        shell_cmd = f"printf '\\nY\\n' | {script_quoted} -s {session_quoted}"
        self._run_command(["bash", "-lc", shell_cmd], cwd=self.collected_demos_dir)

    def run_mt3_deploy(self):
        mt3_root_quoted = shlex.quote(str(self.mt3_root))
        shell_cmd = f"cd {mt3_root_quoted} && make deploy_mt3"
        # `deploy_mt3` in the Makefile uses ${PWD} for docker volume mounting,
        # so we run through `cd` to guarantee PWD points to the MT3 repo root.
        self._run_command(["bash", "-lc", shell_cmd], cwd=self.this_dir)

    def run_replay(self):
        self._run_command([sys.executable, str(self.replay_script)], cwd=self.this_dir)

    def run_cycle(self):
        print("\n==================== NEW INFERENCE CYCLE ====================")
        camera_data = self.capture_workspace_camera_data()
        if camera_data is None:
            print("Cycle aborted: camera capture failed.")
            return

        session_name = self.save_session_demo(camera_data)
        self.refresh_inference_assets(session_name)
        self.run_mt3_deploy()
        self.run_replay()
        print("Cycle complete. Press 's' to run again.")

    def run(self):
        print("\nControls:")
        print("  [s] Capture + update inference assets + run MT3 + replay")
        print("  [q] Quit")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while not rospy.is_shutdown():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue

                key = sys.stdin.read(1).lower()
                if key == "q":
                    print("\nQuitting inference pipeline.")
                    break
                if key == "s":
                    try:
                        self.run_cycle()
                    except Exception as exc:
                        print(f"Cycle failed: {exc}")
                        traceback.print_exc()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except cv2.error:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="MT3 one-key inference pipeline.")
    parser.add_argument(
        "--task-name",
        type=str,
        default="pick_up_cube",
        help="Task label written to task_name.txt in the generated demo folder.",
    )
    parser.add_argument(
        "--replay-mode",
        choices=["sim", "real"],
        default="real",
        help="Only used to print which replay roslaunch command to keep running.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = InferencePipeline(task_name=args.task_name, replay_mode=args.replay_mode)
    pipeline.print_launch_instructions()
    pipeline.run()


if __name__ == "__main__":
    main()
