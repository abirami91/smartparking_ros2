#!/usr/bin/env python3
"""
Self-contained video_detector node.

– Subscribes to /camera/image_raw (sensor_msgs/Image)
– Runs YOLOv8 inference on each frame
– Uses your PKLot YAML to define parking‐slot bounding boxes
– Publishes:
    • visualization_msgs/MarkerArray on /parking_markers
    • sensor_msgs/Image (annotated) on /camera/annotated
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import yaml
import os

from ultralytics import YOLO

# Constants
CAR_CLASS_ID = 2        # COCO class index for “car”
CONF_THRESH  = 0.20     # YOLO confidence threshold
TOTAL_SPOTS  = 60       # adjust to your lot’s capacity

class VideoDetector(Node):
    def __init__(self):
        super().__init__('video_detector')
        self.bridge = CvBridge()

        # 1) Load YOLOv8 model
        weights = os.path.expanduser(
            '/root/ros2_ws/src/SMARTPARKING/training/yolov8n.pt'
        )
        if not os.path.isfile(weights):
            self.get_logger().error(f'Weights not found: {weights}')
            rclpy.shutdown()
            return
        self.model = YOLO(weights)
        self.get_logger().info('YOLO model loaded.')

        # 2) Load parking‐slot ROIs from YAML
        yaml_path = os.path.expanduser(
            '/root/ros2_ws/src/SMARTPARKING/training/pklot50.yaml'
        )
        if not os.path.isfile(yaml_path):
            self.get_logger().error(f'ROI YAML not found: {yaml_path}')
            rclpy.shutdown()
            return
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        # Assume data has a list of slots under key "rois", each [x,y,w,h]
        self.rois = data.get('rois', [])
        self.get_logger().info(f'Loaded {len(self.rois)} slot ROIs.')

        # 3) Create ROS interfaces
        self.sub  = self.create_subscription(
            Image, 'camera/image_raw', self.cb_image, 10)
        self.pub_markers   = self.create_publisher(MarkerArray, 'parking_markers', 10)
        self.pub_annotated = self.create_publisher(Image,       'camera/annotated',   10)

        self.get_logger().info('video_detector ready. Waiting for images…')

    def cb_image(self, msg: Image):
        # Convert ROS → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run YOLO inference
        results = self.model(frame, conf=CONF_THRESH)[0]

        # Count and draw car detections
        occupied = 0
        for box, conf, cls in zip(
            results.boxes.xyxy,
            results.boxes.conf,
            results.boxes.cls
        ):
            if int(cls) != CAR_CLASS_ID:
                continue
            occupied += 1
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 0, 255), 2)
            label = f'car {conf:.2f}'
            cv2.putText(frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,255),1,cv2.LINE_AA)

        # Overlay occupancy stats
        free = max(TOTAL_SPOTS - occupied, 0)
        cv2.putText(frame, f'Occupied: {occupied}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,0,255),2,cv2.LINE_AA)
        cv2.putText(frame, f'Free:     {free}', (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0),2,cv2.LINE_AA)

        # Publish MarkerArray for slots
        ma = MarkerArray()
        for idx, (x,y,w,h) in enumerate(self.rois):
            m = Marker()
            m.header = msg.header
            m.ns     = 'slots'
            m.id     = idx
            m.type   = Marker.CUBE
            m.action = Marker.ADD
            # Center & scale
            m.pose.position.x = x + w/2
            m.pose.position.y = y + h/2
            m.pose.position.z = 0.1
            m.scale.x = w
            m.scale.y = h
            m.scale.z = 0.1
            # Color by occupancy
            if idx < occupied:
                m.color.r = 1.0; m.color.g = 0.0; m.color.a = 0.5
            else:
                m.color.r = 0.0; m.color.g = 1.0; m.color.a = 0.5
            ma.markers.append(m)
        self.pub_markers.publish(ma)

        # Publish annotated frame
        ann = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        ann.header = msg.header
        self.pub_annotated.publish(ann)

def main(args=None):
    rclpy.init(args=args)
    node = VideoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
