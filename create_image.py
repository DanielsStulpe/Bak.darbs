import cv2
import numpy as np
import os

# =========================================
# CONFIG
# =========================================
rows = [
    ("Ground Truth", "images/ground_truth/gt_0.jpg", "images/ground_truth/gt_1.jpg"),
    ("YOLOv8", "images/yolo/yolov8_0.jpg", "images/yolo/yolov8_1.jpg"),
    ("YOLO11", "images/yolo/yolo11_0.jpg", "images/yolo/yolo11_1.jpg"),
    ("YOLO26", "images/yolo/yolo26_0.jpg", "images/yolo/yolo26_1.jpg"),
    ("SSD", "images/ssd/ssd_0.jpg", "images/ssd/ssd_1.jpg"),
    ("RetinaNet", "images/retinanet/retinanet_0.jpg", "images/retinanet/retinanet_1.jpg"),
    ("Faster R-CNN", "images/faster_rcnn/faster_rcnn_0.jpg", "images/faster_rcnn/faster_rcnn_1.jpg"),
]

output_path = "comparison_grid.jpg"

# =========================================
# LOAD & RESIZE IMAGES
# =========================================
loaded_rows = []

target_size = (400, 400)  # width, height

for label, img1_path, img2_path in rows:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)

    loaded_rows.append((label, img1, img2))

# =========================================
# CREATE GRID CANVAS
# =========================================
row_height = target_size[1]
col_width = target_size[0]

label_width = 200  # space for model names (left side)
header_height = 80  # space for top titles

total_height = header_height + len(rows) * row_height
total_width = label_width + 2 * col_width

canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

# =========================================
# ADD COLUMN TITLES
# =========================================
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(canvas, "Attēls 1",
            (label_width + 80, 50),
            font, 1, (0, 0, 0), 2)

cv2.putText(canvas, "Attēls 2",
            (label_width + col_width + 80, 50),
            font, 1, (0, 0, 0), 2)

# =========================================
# FILL GRID
# =========================================
for i, (label, img1, img2) in enumerate(loaded_rows):
    y_offset = header_height + i * row_height

    # Insert images
    canvas[y_offset:y_offset+row_height,
           label_width:label_width+col_width] = img1

    canvas[y_offset:y_offset+row_height,
           label_width+col_width:label_width+2*col_width] = img2

    # Add row label (model name)
    cv2.putText(canvas, label,
                (20, y_offset + row_height // 2),
                font, 0.7, (0, 0, 0), 2)

# =========================================
# SAVE RESULT
# =========================================
cv2.imwrite(output_path, canvas)
print(f"Saved comparison image: {output_path}")