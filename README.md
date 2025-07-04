# SmartParking ROS2 Pipeline

An end-to-end, containerized ROS 2 Humble pipeline for real-time parking‑lot occupancy detection using YOLOv8 and OpenCV, 
deployed on ARM64 (Raspberry Pi) and streamed to any browser via MJPEG.

## Overview
This repository demonstrates a complete workflow:

* **Training** on the PKLot public dataset with YOLOv8 (not included in this repo).
* **ROS 2 Nodes**:

    * VideoPublisher: loops an MP4 (recorded or test video) and publishes raw frames on /camera/image_raw.

    * VideoDetector: subscribes to /camera/image_raw, runs YOLOv8 inference and parking‑slot logic, publishes annotated frames on /camera/annotated and slot‐status markers on /parking_markers.

* **Web Streaming**:
    * **rosbridge_server** exposes a WebSocket (port 9090).

    * **web_video_server** serves any sensor_msgs/Image as MJPEG on HTTP (port 8080).

* View the live feed in any browser:
    * http://<RASPBERRY_PI_IP>:8080/stream?topic=/camera/annotated

## Prerequisites:

* **HOST**: Any Linux desktop (for development and Medium/GitHub docs).
* **Raspberry Pi**: ARM64 (e.g. Pi 4) running Linux. Docker installed.

## Dockerized Setup on Raspberry Pi

1. **Build the Docker image (on Pi)**  
   ```bash
   cd ~/smartparking/docker
   docker build -t ros2-humble-pi -f ros2-humble-pi.Dockerfile .

2. **Launch the container**
    ```bash
    docker rm -f smartparking_pi || true
    docker run -it --name smartparking_pi \
    -v ~/smartparking:/root/ros2_ws \
    -p 8080:8080 \
    -p 9090:9090 \
    ros2-humble-pi

3. **Inside the container: build & source**
    ```bash
    cd /root/ros2_ws
    colcon build --symlink-install
    source /opt/ros/humble/setup.bash
    source install/setup.bash

4. **Run the pipeline**
    ```bash
    ros2 run smartparking_ros video_publisher &
    ros2 run smartparking_ros video_detector &
    ros2 run rosbridge_server rosbridge_websocket &
    ros2 run web_video_server web_video_server

6. **View in your browser**
    * Annotated feed:
    ```bash
        http://<PI_IP>:8080/stream?topic=/camera/annotated 

7. * Raw feed:
    ```bash
        http://<PI_IP>:8080/stream?topic=/camera/image_raw

## Features:
* Looping video publisher for offline or recorded footage.

* YOLOv8-based real-time object detection.

* Parking-slot occupancy logic with bounding-box overlays and counts.

* Containerized for ARM64 (Raspberry Pi) for easy deployment.

* Browser-based live visualization via MJPEG stream.

* WebSocket bridge for remote ROS 2 clients.

## License: 

* This project is released under the MIT License.
