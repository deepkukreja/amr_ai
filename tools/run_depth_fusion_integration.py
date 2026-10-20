#!/usr/bin/env python3
"""Launch an isolated ROS 2 graph and run the depth-fusion integration test."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def start_process(label, command, env):
    print(f"Starting {label}...", flush=True)
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )


def stop_process(process, label):
    if process is None or process.poll() is not None:
        return

    print(f"Stopping {label}...", flush=True)
    try:
        # SIGINT mirrors normal Ctrl+C shutdown for ROS 2 nodes.
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        print(f"Graceful shutdown timed out for {label}.", flush=True)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            print(f"Force-stopping {label}.", flush=True)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main():
    if shutil.which("ros2") is None:
        print(
            "ERROR: ros2 was not found. Source ROS 2 Jazzy and the "
            "amr_detector workspace first.",
            file=sys.stderr,
        )
        return 2

    env = os.environ.copy()

    # Keep this run separate from any ROS graph already running in WSL.
    domain_id = 80 + (os.getpid() % 20)
    env["ROS_DOMAIN_ID"] = str(domain_id)

    tf_process = None
    node_process = None

    print(f"Using isolated ROS_DOMAIN_ID={domain_id}", flush=True)

    try:
        tf_process = start_process(
            "static test transform",
            [
                "ros2", "run", "tf2_ros", "static_transform_publisher",
                "--x", "0", "--y", "0", "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "base_link",
                "--child-frame-id", "camera_depth_optical_frame",
            ],
            env,
        )

        time.sleep(0.5)
        if tf_process.poll() is not None:
            print("ERROR: Static TF publisher exited unexpectedly.")
            return 1

        node_process = start_process(
            "depth-fusion node",
            ["ros2", "run", "amr_detector", "depth_fusion_3d"],
            env,
        )

        # The existing test waits for subscriptions and the 3D output.
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "test_depth_fusion_integration.py"),
                ],
                cwd=ROOT,
                env=env,
                timeout=25,
                check=False,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            print("FAIL: Integration test exceeded 25 seconds.")
            return 124

    except KeyboardInterrupt:
        print("Integration test interrupted.", flush=True)
        return 130

    finally:
        stop_process(node_process, "depth-fusion node")
        stop_process(tf_process, "static test transform")


if __name__ == "__main__":
    sys.exit(main())
