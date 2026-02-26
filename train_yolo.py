from ultralytics import YOLO
import time
import shutil
import csv
import os
import torch

# ==========================
# CONFIG
# ==========================

models_to_train = [
    "yolov8m.pt",
    "yolov9m.pt",
    "yolov10m.pt",
    "yolo11m.pt",
    "yolo12m.pt",
    "yolo26m.pt"
]

data_path = "roboflow_dataset_yolo/data.yaml"
imgsz = 640
epochs = 100
batch_size = 4
optimizer = "SGD"
lr0 = 0.0025
weight_decay = 0.0005

results_dir = "model_comparison_results"
os.makedirs(results_dir, exist_ok=True)

csv_file = os.path.join(results_dir, "results.csv")

# ==========================
# CSV HEADER
# ==========================

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model",
        "Parameters_M",
        "Model_size_MB",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "Training_time_sec"
    ])

# ==========================
# LOOP THROUGH MODELS
# ==========================

for model_name in models_to_train:

    print(f"\n========== Training {model_name} ==========")

    model = YOLO(model_name)
    
    # TRAINING TIME
    start_time = time.time()

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        optimizer=optimizer,
        lr0=lr0,
        weight_decay=weight_decay,
        workers=1,
        device=0,
        name=f"train_{model_name.replace('.pt','')}"
    )

    training_time = time.time() - start_time

    
    # VALIDATION METRICS
    metrics = model.val()

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    
    # SAVE BEST WEIGHTS
    best_weight_path = model.trainer.best
    new_weight_path = os.path.join(results_dir, f"best_{model_name}")
    shutil.copy(best_weight_path, new_weight_path)
    
    # MODEL SIZE (MB)
    model_size_mb = os.path.getsize(new_weight_path) / (1024 * 1024)
    
    # PARAMETERS
    params = sum(p.numel() for p in model.model.parameters()) / 1e6  # Millions

    
    # PRINT RESULTS
    print(f"\nResults for {model_name}")
    print(f"Parameters: {params:.2f}M")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"mAP50: {map50}")
    print(f"mAP50-95: {map50_95}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Training time: {training_time:.2f} sec")

    
    # SAVE TO CSV
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            round(params, 2),
            round(model_size_mb, 2),
            map50,
            map50_95,
            precision,
            recall,
            round(training_time, 2),
        ])

    # Free GPU memory (important for Jetson)
    torch.cuda.empty_cache()

print("\n All experiments completed.")
print(f" Results saved to: {csv_file}")