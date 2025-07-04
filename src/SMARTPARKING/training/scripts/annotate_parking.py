#!/usr/bin/env python3
"""
Batch‐annotate a folder of parking‐lot images with your trained YOLOv8 model.

Draws each ROI in green (“free”), red (“occupied”) or yellow (“unknown”),
and writes the results to a matching filename in the output directory.

Usage:
    python annotate_parking_batch.py \
      --model /workspace/runs/train/pklot50/weights/best.pt \
      --input_dir /workspace/pklot_50/images \
      --output_dir /workspace/runs/improved_annotation
"""
import os
import argparse
import cv2
from ultralytics import YOLO

# Map class IDs → human‐readable labels
CLASS_NAMES = {1: "free", 2: "occupied", 0: "unknown"}

# BGR colors for each class box
COLORS = {
    1: (0, 255, 0),    # free  → green
    2: (0, 0, 255),    # occupied → red
    0: (0, 255, 255)   # unknown → yellow
}

def annotate_image(model, image_path, out_path):
    """
    Runs inference on one image and writes the color‐coded result to out_path.
    """
    img = cv2.imread(image_path)
    # Run YOLOv8 model, get the first (and only) results object
    res = model(img)[0]
    # Grab raw boxes: Nx6 array [x1,y1,x2,y2,conf,cls]
    boxes = res.boxes.data.cpu().numpy()

    # Draw each box + label
    for x1, y1, x2, y2, conf, cls in boxes.tolist():
        x1,y1,x2,y2 = map(int, (x1, y1, x2, y2))
        cls = int(cls)
        color = COLORS.get(cls, (255,255,255))
        label = CLASS_NAMES.get(cls, str(cls))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save the annotated image
    cv2.imwrite(out_path, img)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, help="Path to your .pt weights")
    p.add_argument("--input_dir",  required=True, help="Folder of images to annotate")
    p.add_argument("--output_dir", required=True, help="Where to save annotated images")
    args = p.parse_args()

    # Load your YOLOv8 model once
    model = YOLO(args.model)

    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process every JPG/PNG in the input folder
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        annotate_image(model, in_path, out_path)
        print(f"Annotated {fname}")

if __name__ == "__main__":
    main()
