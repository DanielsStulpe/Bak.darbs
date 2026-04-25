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

images = [
    'result_images/default/img-344.jpg',
    'result_images/default/img-415.jpg',
    'result_images/default/img-411.jpg'
]

annotations = [
    'result_images/default/img-344.json',
    'result_images/default/img-415.json',
    'result_images/default/img-411.json'
]

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
score_threshold = 0.5
num_classes = 2

# 🎨 Model colors (BGR format)
MODEL_COLORS = {
    "yolov8": (255, 0, 0),        # blue
    "yolo11": (0, 255, 255),      # yellow
    "yolo26": (255, 0, 255),      # purple
    "faster_rcnn": (0, 0, 255),   # red
    "retinanet": (0, 165, 255),   # orange
}

# =========================================
# 1. GROUND TRUTH
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

    cv2.imwrite(output_path, image)
    print(f"GT saved: {output_path}")


os.makedirs("result_images/ground_truth", exist_ok=True)

for i in range(len(images)):
    draw_coco_bboxes(
        images[i],
        annotations[i],
        f"result_images/ground_truth/gt_{i}.jpg"
    )

# =========================================
# 2. YOLO MODELS (CUSTOM COLORS)
# =========================================
os.makedirs("result_images/yolo", exist_ok=True)


def run_yolo_model(model, model_name):
    color = MODEL_COLORS[model_name]

    for i, img_path in enumerate(images):
        image = cv2.imread(img_path)

        results = model(img_path)[0]

        for box, score in zip(results.boxes.xyxy, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = f"result_images/yolo/{model_name}_{i}.jpg"
        cv2.imwrite(output_path, image)
        print(f"{model_name} saved: {output_path}")


# YOLOv8
yolov8_model = YOLO("best_yolov8m.pt")
run_yolo_model(yolov8_model, "yolov8")

# YOLO11
yolo11_model = YOLO("best_yolo11m.pt")
run_yolo_model(yolo11_model, "yolo11")

# YOLO26
yolo26_model = YOLO("best_yolo26m.pt")
run_yolo_model(yolo26_model, "yolo26")

# =========================================
# 3. TORCHVISION MODELS
# =========================================
transform = T.Compose([T.ToTensor()])


def run_torchvision_model(model, model_name, weights_path):
    print(f"\nRunning {model_name}...")

    output_dir = f"result_images/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    color = MODEL_COLORS[model_name]

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

            cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
            cv2.putText(orig, f"{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = f"{output_dir}/{model_name}_{i}.jpg"
        cv2.imwrite(output_path, orig)
        print(f"{model_name} saved: {output_path}")


# Faster R-CNN
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

run_torchvision_model(faster_rcnn, "faster_rcnn", "best_faster_rcnn.pth")

# RetinaNet
retinanet = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)
run_torchvision_model(retinanet, "retinanet", "best_retinanet.pth")