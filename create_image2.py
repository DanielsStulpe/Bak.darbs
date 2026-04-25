import cv2
from PIL import Image, ImageDraw, ImageFont

# =========================================
# CONFIG
# =========================================
models = [
    ("Attēli", [
        "result_images/default/img-344.jpg",
        "result_images/default/img-415.jpg",
        "result_images/default/img-411.jpg"
    ]),
    ("YOLOv8", [
        "result_images/yolo/yolov8_0.jpg",
        "result_images/yolo/yolov8_1.jpg",
        "result_images/yolo/yolov8_2.jpg"
    ]),
    ("YOLO11", [
        "result_images/yolo/yolo11_0.jpg",
        "result_images/yolo/yolo11_1.jpg",
        "result_images/yolo/yolo11_2.jpg"
    ]),
    ("YOLO26", [
        "result_images/yolo/yolo26_0.jpg",
        "result_images/yolo/yolo26_1.jpg",
        "result_images/yolo/yolo26_2.jpg"
    ]),
    ("RetinaNet", [
        "result_images/retinanet/retinanet_0.jpg",
        "result_images/retinanet/retinanet_1.jpg",
        "result_images/retinanet/retinanet_2.jpg"
    ]),
    ("Faster R-CNN", [
        "result_images/faster_rcnn/faster_rcnn_0.jpg",
        "result_images/faster_rcnn/faster_rcnn_1.jpg",
        "result_images/faster_rcnn/faster_rcnn_2.jpg"
    ]),
]

output_path = "comparison_horizontal.jpg"

# =========================================
# SETTINGS
# =========================================
img_w, img_h = 350, 350
padding = 15
left_label_width = 160
footer_height = 80

font_path = "C:/Windows/Fonts/arial.ttf"
font_side = ImageFont.truetype(font_path, 28)
font_footer = ImageFont.truetype(font_path, 28)

rows = 3
cols = len(models)

# =========================================
# CANVAS
# =========================================
total_width = left_label_width + cols * img_w + (cols + 1) * padding
total_height = rows * img_h + (rows + 1) * padding + footer_height

canvas = Image.new("RGB", (total_width, total_height), "white")
draw = ImageDraw.Draw(canvas)

# =========================================
# INSERT IMAGES
# =========================================
for col, (model_name, image_list) in enumerate(models):

    for row in range(rows):
        img = cv2.imread(image_list[row])
        img = cv2.resize(img, (img_w, img_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img)

        x = left_label_width + padding + col * (img_w + padding)
        y = padding + row * (img_h + padding)

        canvas.paste(img_pil, (x, y))

# =========================================
# LEFT SIDE LABELS (Attēls 1/2/3)
# =========================================
row_labels = ["Attēls 1", "Attēls 2", "Attēls 3"]

for row in range(rows):
    bbox = draw.textbbox((0, 0), row_labels[row], font=font_side)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (left_label_width - text_w) // 2
    y = padding + row * (img_h + padding) + (img_h - text_h) // 2

    draw.text((x, y), row_labels[row], font=font_side, fill=(0, 0, 0))

# =========================================
# BOTTOM MODEL LABELS
# =========================================
for col, (model_name, _) in enumerate(models):

    bbox = draw.textbbox((0, 0), model_name, font=font_footer)
    text_w = bbox[2] - bbox[0]

    col_center = left_label_width + padding + col * (img_w + padding) + img_w // 2

    x = col_center - text_w // 2
    y = rows * (img_h + padding) + padding

    draw.text((x, y), model_name, font=font_footer, fill=(0, 0, 0))

# =========================================
# SAVE
# =========================================
canvas.save(output_path)
print(f"Saved: {output_path}")