#!/usr/bin/env python3
import os, cv2
from ultralytics import YOLO

# ——————————————————————————————————————————————————————————————
# 1) Paths (as before)
script_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
model_path   = os.path.join(project_dir, 'runs', 'train', 'pklot50', 'weights', 'best.pt')
video_path   = os.path.join(project_dir, 'small_parking.mp4')

# 2) Sanity checks
assert os.path.isfile(model_path), f"Model not found: {model_path}"
assert os.path.isfile(video_path), f"Video not found: {video_path}"

# 3) Load
model = YOLO(model_path)
cap   = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Cannot open {video_path}"

fps   = cap.get(cv2.CAP_PROP_FPS) or 25
delay = max(1, int(1000 / fps))  # ms between frames

# ——————————————————————————————————————————————————————————————
# 4) Display params
CONF_THRESH   = 0.4
BOX_COLOR     = (0, 255, 0)    # green
TEXT_BG_COLOR = (0, 255, 0)    # same as box
TEXT_COLOR    = (255, 255, 255)
THICKNESS     = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # — a) Inference
    results = model(frame)[0]  # get the first (and only) Results object

    # — b) Extract boxes, confidences
    boxes = results.boxes.xyxy.cpu().numpy()   # shape: [N,4]
    confs = results.boxes.conf.cpu().numpy()   # shape: [N]

    # — c) Filter by confidence
    keep = confs >= CONF_THRESH
    boxes = boxes[keep]
    confs = confs[keep]

    # — d) Draw each box + label
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)

        # text label (e.g. “0.72”)
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        # draw filled rect behind text
        cv2.rectangle(frame,
                      (x1, y1 - th - 4),
                      (x1 + tw, y1),
                      TEXT_BG_COLOR,
                      cv2.FILLED)
        # draw text
        cv2.putText(frame,
                    label,
                    (x1, y1 - 2),
                    FONT,
                    FONT_SCALE,
                    TEXT_COLOR,
                    thickness=1)

    # — e) Overlay total count
    cv2.putText(frame,
                f"Cars: {len(boxes)}",
                (10, 30),
                FONT,
                1.0,
                TEXT_COLOR,
                thickness=2)

    # — f) Show
    cv2.imshow("YOLO Parking Detection", frame)
    key = cv2.waitKey(delay)
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
