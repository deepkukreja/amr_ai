import os
from setuptools import setup

package_name = "amr_detector"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join("share", package_name, "config"),
            ["config/depth_fusion.yaml"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            ["launch/amr_perception.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "detector_node = amr_detector.detector_node:main",
            "depth_fusion_3d = amr_detector.depth_fusion_3d:main",
        ],
    },
)
