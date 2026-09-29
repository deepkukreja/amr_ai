from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    package_share = get_package_share_directory("amr_detector")

    depth_fusion_config = os.path.join(
        package_share,
        "config",
        "depth_fusion.yaml",
    )

    model_path = LaunchConfiguration("model_path")

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_path",
            description="Absolute path to the YOLO model checkpoint",
        ),

        Node(
            package="amr_detector",
            executable="detector_node",
            name="amr_detector",
            output="screen",
            parameters=[{
                "model_path": model_path,
            }],
        ),

        Node(
            package="amr_detector",
            executable="depth_fusion_3d",
            name="depth_fusion_3d",
            output="screen",
            parameters=[depth_fusion_config],
        ),
    ])
