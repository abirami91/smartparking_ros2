#!/usr/bin/env python3
import os
import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# 1) Locate video file
#    We assume this script lives in training/src/, so project_dir is one up.
script_dir  = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
video_path  = os.path.join(project_dir, 'small_parking.mp4')
assert os.path.isfile(video_path), f"Video not found: {video_path}"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load a COCO-pretrained YOLOv8 model
#    This model already knows “car” (class id 2) without any extra training.
model = YOLO('yolov8n.pt')

# ─────────────────────────────────────────────────────────────────────────────
# 3) Open the video for frame-by-frame processing
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Cannot open video: {video_path}"
# Compute a waitKey delay so playback roughly matches source FPS
fps   = cap.get(cv2.CAP_PROP_FPS) or 25
delay = max(1, int(1000 / fps))

# ─────────────────────────────────────────────────────────────────────────────
# 4) Detection settings
CAR_CLASS_ID  = 2          # COCO index for “car”
CONF_THRESH   = 0.20       # only show detections ≥20% confidence
BOX_COLOR     = (0, 0, 255)# red bounding boxes
THICK_BOX     = 2          # box line thickness

# ─────────────────────────────────────────────────────────────────────────────
# 5) Define your lot’s total capacity
#    Used to compute “free” = TOTAL_SPOTS – occupied_count
TOTAL_SPOTS   = 60

# ─────────────────────────────────────────────────────────────────────────────
# 6) Text‐drawing parameters
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FSCALE        = 0.6        # font scale
OUTLINE_COLOR = (0, 0, 0)  # black outline for contrast
TEXT_COLOR    = (255,255,255) # white fill
LINE_TYPE     = cv2.LINE_AA

# ─────────────────────────────────────────────────────────────────────────────
# 7) Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # a) Run YOLO inference on the current frame
    #    conf=CONF_THRESH filters out low-confidence boxes
    results = model(frame, conf=CONF_THRESH)[0]

    # b) Iterate detections, draw red “car” boxes, and count occupied spots
    occupied = 0
    for box, conf, cls in zip(
            results.boxes.xyxy,   # [x1, y1, x2, y2]
            results.boxes.conf,   # confidence scores
            results.boxes.cls     # class indices
        ):
        if int(cls) != CAR_CLASS_ID:
            continue  # skip non‐cars

        occupied += 1
        x1, y1, x2, y2 = map(int, box.tolist())

        # Draw a red rectangle around each car
        cv2.rectangle(
            frame,
            (x1, y1), (x2, y2),
            BOX_COLOR,
            THICK_BOX
        )

        # Prepare the label text “car 0.82”
        label = f"car {float(conf):.2f}"
        text_pos = (x1, y1 - 6)  # slightly above the box

        # Draw a thick black outline for readability
        cv2.putText(
            frame, label, text_pos,
            FONT, FSCALE,
            OUTLINE_COLOR,
            thickness=2,
            lineType=LINE_TYPE
        )
        # Draw the white text on top
        cv2.putText(
            frame, label, text_pos,
            FONT, FSCALE,
            TEXT_COLOR,
            thickness=1,
            lineType=LINE_TYPE
        )

    # c) Compute how many free spots remain
    free = max(TOTAL_SPOTS - occupied, 0)

    # d) Overlay the “Occupied” and “Free” counters in the top-left
    cv2.putText(
        frame,
        f"Occupied: {occupied}",
        (10, 30),         # position
        FONT, 0.8,        # font + scale
        BOX_COLOR,        # red text
        2, LINE_TYPE      # thickness + line type
    )
    cv2.putText(
        frame,
        f"Free:     {free}",
        (10, 60),
        FONT, 0.8,
        (0,255,0),        # green text for free
        2, LINE_TYPE
    )

    # e) Show the annotated frame
    cv2.imshow("Parking Status", frame)

    # f) Exit on ESC
    if cv2.waitKey(delay) & 0xFF == 27:
        break

# ─────────────────────────────────────────────────────────────────────────────
# 8) Cleanup
cap.release()
cv2.destroyAllWindows()
