from setuptools import setup, find_packages

setup(
    name="lidar_det",
    version="0.0.1",
    author="Dan Jia",
    author_email="jia@vision.rwth-aachen.de",
    packages=find_packages(
        include=["lidar_det", "lidar_det.*", "lidar_det.*.*"]
    ),
    license="LICENSE.txt",
    description="Object detection from LiDAR point cloud.",
)
