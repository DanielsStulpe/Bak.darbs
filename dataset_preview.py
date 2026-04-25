import cv2
import numpy as np

# =========================================
# CONFIG
# =========================================
image_paths = [
    "dataset_preview/1.png",
    "dataset_preview/2.png",
    "dataset_preview/3.png",
    "dataset_preview/4.png"
]

output_path = "combined.png"

img_size = 640
padding = 20  # white space between images

# =========================================
# LOAD & RESIZE
# =========================================
images = []

for path in image_paths:
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Image not found: {path}")

    img = cv2.resize(img, (img_size, img_size))
    images.append(img)

# =========================================
# CREATE CANVAS
# =========================================
# 2x2 grid
canvas_height = 2 * img_size + padding
canvas_width = 2 * img_size + padding

canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # white

# =========================================
# PLACE IMAGES
# =========================================
# Top-left
canvas[0:img_size, 0:img_size] = images[0]

# Top-right
canvas[0:img_size, img_size + padding:img_size*2 + padding] = images[1]

# Bottom-left
canvas[img_size + padding:img_size*2 + padding, 0:img_size] = images[2]

# Bottom-right
canvas[
    img_size + padding:img_size*2 + padding,
    img_size + padding:img_size*2 + padding
] = images[3]

# =========================================
# SAVE
# =========================================
cv2.imwrite(output_path, canvas)
print(f"Saved: {output_path}")