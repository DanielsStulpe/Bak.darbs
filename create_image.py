import cv2
from PIL import Image, ImageDraw, ImageFont

# =========================================
# CONFIG
# =========================================
rows = [
    ("Attēli",
     "result_images/default/img-344.jpg",
     "result_images/default/img-415.jpg",
     "result_images/default/img-411.jpg"),

    ("YOLOv8",
     "result_images/yolo/yolov8_0.jpg",
     "result_images/yolo/yolov8_1.jpg",
     "result_images/yolo/yolov8_2.jpg"),

    ("YOLO11",
     "result_images/yolo/yolo11_0.jpg",
     "result_images/yolo/yolo11_1.jpg",
     "result_images/yolo/yolo11_2.jpg"),

    ("YOLO26",
     "result_images/yolo/yolo26_0.jpg",
     "result_images/yolo/yolo26_1.jpg",
     "result_images/yolo/yolo26_2.jpg"),

    ("RetinaNet",
     "result_images/retinanet/retinanet_0.jpg",
     "result_images/retinanet/retinanet_1.jpg",
     "result_images/retinanet/retinanet_2.jpg"),

    ("Faster R-CNN",
     "result_images/faster_rcnn/faster_rcnn_0.jpg",
     "result_images/faster_rcnn/faster_rcnn_1.jpg",
     "result_images/faster_rcnn/faster_rcnn_2.jpg"),
]

output_path = "comparison_grid.jpg"

# =========================================
# SETTINGS
# =========================================
img_w, img_h = 400, 400
label_width = 220
header_height = 100
padding = 15

font_path = "C:/Windows/Fonts/arial.ttf"

# Vienots stils
font_header = ImageFont.truetype(font_path, 36)
font_row = ImageFont.truetype(font_path, 30)

# =========================================
# CANVAS
# =========================================
rows_count = len(rows)
cols_count = 3

total_width = label_width + cols_count * img_w + (cols_count + 1) * padding
total_height = header_height + rows_count * img_h + (rows_count + 1) * padding

canvas = Image.new("RGB", (total_width, total_height), "white")
draw = ImageDraw.Draw(canvas)

# =========================================
# HEADER (precīzi centrēts)
# =========================================
headers = ["Attēls 1", "Attēls 2", "Attēls 3"]

for i, text in enumerate(headers):
    # Teksta izmērs
    bbox = draw.textbbox((0, 0), text, font=font_header)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Kolonnas centrs
    col_x_center = label_width + padding + i * (img_w + padding) + img_w // 2

    # Precīza centrēšana
    x = col_x_center - text_w // 2
    y = (header_height - text_h) // 2

    draw.text((x, y), text, font=font_header, fill=(0, 0, 0))

# =========================================
# ROWS (attēli + vertikāli centrēti labeli)
# =========================================
for i, (label, img1_path, img2_path, img3_path) in enumerate(rows):

    y_offset = header_height + padding + i * (img_h + padding)

    # ---- Attēli ----
    for j, path in enumerate([img1_path, img2_path, img3_path]):
        img = cv2.imread(path)
        img = cv2.resize(img, (img_w, img_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        x_offset = label_width + padding + j * (img_w + padding)
        canvas.paste(img_pil, (x_offset, y_offset))

    # ---- Label (VERTIKĀLI CENTRĒTS) ----
    bbox = draw.textbbox((0, 0), label, font=font_row)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Vertikāli centrēts pret attēlu
    text_y = y_offset + (img_h - text_h) // 2

    # Horizontāli (centrēt label laukā)
    text_x = (label_width - text_w) // 2

    draw.text((text_x, text_y), label, font=font_row, fill=(0, 0, 0))

# =========================================
# SAVE
# =========================================
canvas.save(output_path)
print(f"Saved: {output_path}")