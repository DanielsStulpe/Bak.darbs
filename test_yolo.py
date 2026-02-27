from ultralytics import YOLO
import csv
import os
import torch


models = [
    "best_yolov8m.pt",
    "best_yolov9m.pt",
    "best_yolov10m.pt",
    "best_yolo11m.pt",
    "best_yolo12m.pt",
    "best_yolo26m.pt"
]

results_dir = "yolo_models_results"
os.makedirs(results_dir, exist_ok=True)

csv_file = os.path.join(results_dir, "test_results.csv")

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall"
    ])


for model in models:
    print(f"\nTesting {model}...")
    model_path = os.path.join(results_dir, model)
    yolo_model = YOLO(model_path)

    metrics = yolo_model.val(
        data="roboflow_dataset_yolo/data.yaml",
        split="test",
        imgsz=640,
        batch=1,
        conf=0.25,
        iou=0.7,
        max_det=100,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    map50 = metrics.box.map50
    map5095 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map5095:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model,
            f"{map50:.4f}",
            f"{map5095:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}"
        ])


print("\n All experiments completed.")
print(f" Results saved to: {csv_file}")