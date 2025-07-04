#!/usr/bin/env python3
"""
Convert the pklot_50 COCO-style annotations to YOLO-txt files.

Usage:
    python scripts/convert_pklot50_to_yolo.py --root training/pklot_50
"""

import json, zipfile, argparse
from pathlib import Path

def convert(root: Path):
    # 1) make sure images are extracted
    img_zip = root / "images.zip"
    img_dir = root / "images"
    if img_zip.exists() and not img_dir.exists():
        print("Extracting images.zip …")
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(img_dir)

    # 2) load COCO json
    coco = json.loads((root / "annotations.json").read_text())

    # map image_id → (w, h, file_name)
    img_index = {im["id"]: (im["width"], im["height"], im["file_name"])
                 for im in coco["images"]}

    # create labels/ folder
    labels_dir = root / "labels"
    labels_dir.mkdir(exist_ok=True)

    # 3) walk through annotations
    for ann in coco["annotations"]:
        img_w, img_h, fname = img_index[ann["image_id"]]
        # COCO bbox = [x_min, y_min, w, h] in absolute pixels
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        bw = w / img_w
        bh = h / img_h
        class_id = ann["category_id"]      # 0 = empty, 1 = occupied for PKLot-50

        (labels_dir / f"{Path(fname).stem}.txt").open("a").write(
            f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )

    print("COCO annotations converted → YOLO txt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="training/pklot_50",
                   help="folder that holds annotations.json and images.zip")
    args = p.parse_args()
    convert(Path(args.root).resolve())
