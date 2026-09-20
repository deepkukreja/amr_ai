from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import (
    get_package_share_directory,
    get_package_prefix,
)
import os


def generate_launch_description():

    package_share = get_package_share_directory("amr_detector")
    package_prefix = get_package_prefix("amr_detector")

    detector_script = os.path.join(
        package_prefix,
        "lib",
        "amr_detector",
        "detector_node",
    )

    depth_fusion_config = os.path.join(
        package_share,
        "config",
        "depth_fusion.yaml",
    )

    return LaunchDescription([
        Node(
            executable="/home/deepk12341234/amr_ai/.venv/bin/python",
            arguments=[detector_script],
            name="amr_detector",
            output="screen",
        ),

        Node(
            package="amr_detector",
            executable="depth_fusion_3d",
            name="depth_fusion_3d",
            output="screen",
            parameters=[depth_fusion_config],
        ),
    ])
