"""Exercise the world reader with Jazzy map_server and planner_server only."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import subprocess
import time

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import PoseStamped, TransformStamped
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from nav2_msgs.action import ComputePathToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from tf2_ros import StaticTransformBroadcaster
import yaml

from read_nav2 import WorldReader, write_capture


def stop_owned(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def run_fixture(args, executor, driver, reader, stack):
    directory = args.output
    parameters = yaml.safe_load(args.parameters.read_text(encoding="utf-8"))
    planner = parameters["planner_server"]
    costmap = parameters["global_costmap"]["global_costmap"]["ros__parameters"]
    planner["ros__parameters"]["use_sim_time"] = False
    planner["ros__parameters"]["GridBased"]["allow_unknown"] = False
    costmap["use_sim_time"] = False
    costmap["plugins"].remove("obstacle_layer")
    config = directory / "fixture-params.yaml"
    config.write_text(yaml.safe_dump({"planner_server": planner, "global_costmap": parameters["global_costmap"]}), encoding="utf-8")

    def start(package, name, arguments):
        executable = Path(get_package_prefix(package)) / "lib" / package / name
        log = stack.enter_context((directory / f"{name}.log").open("w", encoding="utf-8"))
        process = subprocess.Popen([str(executable), "--ros-args", *arguments], stdout=log, stderr=subprocess.STDOUT)
        stack.callback(stop_owned, process)

    def wait(future, message, timeout=12):
        executor.spin_until_future_complete(future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError(message)
        return future.result()

    def activate(name):
        service = driver.create_client(ChangeState, f"/{name}/change_state")
        if not service.wait_for_service(timeout_sec=8):
            raise RuntimeError(f"{name} lifecycle service unavailable")
        for transition in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
            request = ChangeState.Request()
            request.transition.id = transition
            if not wait(service.call_async(request), f"{name} lifecycle timed out").success:
                raise RuntimeError(f"{name} lifecycle rejected transition {transition}")

    broadcaster = StaticTransformBroadcaster(driver)
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.header.stamp = driver.get_clock().now().to_msg()
    transform.child_frame_id = "base_footprint"
    transform.transform.rotation.w = 1.0
    broadcaster.sendTransform(transform)
    start("nav2_map_server", "map_server", ["-p", f"yaml_filename:={args.map.resolve()}"])
    activate("map_server")
    start("nav2_planner", "planner_server", ["--params-file", str(config.resolve())])
    activate("planner_server")

    def pose(x, y, yaw):
        target = PoseStamped()
        target.header.frame_id = "map"
        target.header.stamp = driver.get_clock().now().to_msg()
        target.pose.position.x, target.pose.position.y = x, y
        target.pose.orientation.z = math.sin(yaw / 2)
        target.pose.orientation.w = math.cos(yaw / 2)
        return target

    action = ActionClient(driver, ComputePathToPose, "/compute_path_to_pose")
    if not action.wait_for_server(timeout_sec=8):
        raise RuntimeError("ComputePathToPose unavailable")
    request = ComputePathToPose.Goal()
    request.use_start = True
    request.start = pose(0.0, 0.0, 0.0)
    request.goal = pose(*args.goal, -math.pi / 2)
    request.planner_id = "GridBased"
    goal = wait(action.send_goal_async(request), "Planner goal timed out")
    if not goal.accepted:
        raise RuntimeError("Planner rejected goal")
    result = wait(goal.get_result_async(), "Planner result timed out")
    if result.status != GoalStatus.STATUS_SUCCEEDED:
        raise RuntimeError(f"Planner failed: {result.result}")

    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.2)
        capture = reader.capture("Small warehouse", source="NAV2_PLANNING_FIXTURE")
        if capture is not None:
            break
    else:
        raise RuntimeError(reader.reason)
    expected = [[value.pose.position.x, value.pose.position.y, value.pose.position.z] for value in result.result.path.poses]
    if capture["plan"]["points"] != expected or math.dist(expected[-1][:2], args.goal) > 0.5:
        raise RuntimeError("Published path differs from the planner result or goal")
    if math.dist([capture["pose"]["x"], capture["pose"]["y"]], expected[0][:2]) > 0.1:
        raise RuntimeError("TF robot position and planner start are misaligned")
    if any(topic.rsplit("/", 1)[-1].startswith("cmd_vel") for topic, _ in driver.get_topic_names_and_types()):
        raise RuntimeError("Unexpected velocity interface in the planning fixture")
    reader.base = "rbw_missing_transform_test"
    if reader.capture("Small warehouse") is not None or "No robot transform" not in reader.reason:
        raise RuntimeError("Missing-transform regression")

    write_capture(directory / "nav2-world.json", capture)
    path_m = sum(math.dist(a, b) for a, b in zip(expected, expected[1:]))
    report = {"pathPoses": len(expected), "pathMetres": path_m, "goal": args.goal, "mapFrame": capture["frame"],
              "mapDimensions": [capture["map"]["width"], capture["map"]["height"]],
              "pathTopicMatchesAction": True, "missingTransformRejected": True, "velocityTopics": 0}
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal", nargs=2, type=float, default=[4.0, -3.0])
    args = parser.parse_args()
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST" or not 100 <= int(os.environ.get("ROS_DOMAIN_ID", "-1")) <= 199:
        parser.error("Select an unused ROS_DOMAIN_ID between 100 and 199 and set discovery to LOCALHOST")
    args.output.mkdir()
    rclpy.init()
    driver = rclpy.create_node("rbw_planner_fixture")
    reader = WorldReader()
    executor = SingleThreadedExecutor()
    executor.add_node(driver)
    executor.add_node(reader)
    try:
        probe_end = time.monotonic() + 1
        while time.monotonic() < probe_end:
            executor.spin_once(timeout_sec=0.15)
        if set(driver.get_node_names()) - {driver.get_name(), reader.get_name()}:
            raise RuntimeError("Selected ROS domain is occupied")
        with ExitStack() as stack:
            run_fixture(args, executor, driver, reader, stack)
    finally:
        executor.shutdown()
        reader.destroy_node()
        driver.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
