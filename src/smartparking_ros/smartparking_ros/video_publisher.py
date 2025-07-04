#!/usr/bin/env python3
"""
video_publisher.py

Reads frames from small_parking.mp4 and publishes them as
sensor_msgs/Image on /camera/image_raw at ~30Hz, looping forever.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        # Publisher for raw camera frames
        self.pub = self.create_publisher(Image, 'camera/image_raw', 10)
        # Bridge to convert between ROS Images and OpenCV
        self.bridge = CvBridge()

        # Path to your MP4 inside the container
        video_path = '/root/ros2_ws/src/SMARTPARKING/training/small_parking.mp4'
        if not os.path.isfile(video_path):
            self.get_logger().error(f'Video file not found: {video_path}')
            rclpy.shutdown()
            return

        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open video: {video_path}')
            rclpy.shutdown()
            return

        # Schedule frame publication at ~30Hz
        self.create_timer(1.0 / 30.0, self.publish_frame)
        self.get_logger().info(f'Starting video publisher on {video_path}')

    def publish_frame(self):
        """
        Timer callback: read the next frame, loop if at end,
        convert to ROS Image, and publish.
        """
        ret, frame = self.cap.read()
        if not ret:
            # Loop back to the first frame instead of exiting
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Convert OpenCV BGR image to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub.publish(msg)

    def destroy_node(self):
        # Ensure VideoCapture is released on shutdown
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
