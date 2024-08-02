from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolov5_detection',
            executable='yolov5_detection_node'
        ),
    ])
