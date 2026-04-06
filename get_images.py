# =========================================
# IMPORTS & CONFIG
# =========================================
from ultralytics import YOLO
import os
import torch
import cv2
import json
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

images = ['images/default/img-397.jpg', 'images/default/img-415.jpg']
annotations = ['images/default/img-397.json', 'images/default/img-415.json']

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
score_threshold = 0.5
num_classes = 2

# =========================================
# 1. GROUND TRUTH VISUALIZATION
# =========================================
def draw_coco_bboxes(image_path, json_path, output_path):
    image = cv2.imread(image_path)

    with open(json_path, 'r') as f:
        anns = json.load(f)

    for ann in anns:
        x, y, w, h = ann["bbox"]

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(ann["category_id"]), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"GT saved: {output_path}")


os.makedirs("images/ground_truth", exist_ok=True)

for i in range(len(images)):
    draw_coco_bboxes(
        images[i],
        annotations[i],
        f"images/ground_truth/gt_{i}.jpg"
    )


# =========================================
# 2. YOLO MODELS
# =========================================
os.makedirs("images/yolo", exist_ok=True)

# YOLOv8
yolov8_model = YOLO("best_yolov8m.pt")
for i, img in enumerate(images):
    result = yolov8_model(img)
    result[0].save(filename=f"images/yolo/yolov8_{i}.jpg")

# YOLO11
yolo11_model = YOLO("best_yolo11m.pt")
for i, img in enumerate(images):
    result = yolo11_model(img)
    result[0].save(filename=f"images/yolo/yolo11_{i}.jpg")

# YOLO26
yolo26_model = YOLO("best_yolo26m.pt")
for i, img in enumerate(images):
    result = yolo26_model(img)
    result[0].save(filename=f"images/yolo/yolo26_{i}.jpg")


# =========================================
# 3. TORCHVISION MODELS
# =========================================
transform = T.Compose([T.ToTensor()])


def run_torchvision_model(model, model_name, weights_path):
    print(f"\nRunning {model_name}...")

    output_dir = f"images/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    for i, img_path in enumerate(images):
        image = cv2.imread(img_path)
        orig = image.copy()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image_rgb).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        for box, score in zip(boxes, scores):
            if score < score_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(orig, f"{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(f"{output_dir}/{model_name}_{i}.jpg", orig)
        print(f"{model_name} saved: {output_dir}/{model_name}_{i}.jpg")


# Faster R-CNN
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

run_torchvision_model(faster_rcnn, "faster_rcnn", "best_faster_rcnn.pth")


# RetinaNet
retinanet = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)
run_torchvision_model(retinanet, "retinanet", "best_retinanet.pth")


# SSD
ssd = torchvision.models.detection.ssd300_vgg16(weights=None)
run_torchvision_model(ssd, "ssd", "best_ssd.pth")