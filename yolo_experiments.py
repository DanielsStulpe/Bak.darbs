from ultralytics import YOLO
import time
import shutil
import csv
import os
import torch

# ==========================
# CONFIG
# ==========================

base_model = "yolo11m.pt"

optimizers = [
    "Adam",
    "Adamax",
    "AdamW",
    "NAdam",
    "RAdam",
    "RMSProp"
]

data_path = "pothole_dataset_yolo/data.yaml"
imgsz = 640
epochs = 300
batch_size = 4

results_dir = "yolo11_optimizer_results"
os.makedirs(results_dir, exist_ok=True)

train_csv = os.path.join(results_dir, "train_results.csv")
test_csv = os.path.join(results_dir, "test_results.csv")

device = [0, 1] if torch.cuda.is_available() else "cpu"

# ==========================
# CSV HEADERS
# ==========================
with open(train_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Optimizer",
        "Parameters_M",
        "Model_size_MB",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "Training_time_sec"
    ])

with open(test_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Optimizer",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall"
    ])

# ==========================
# TRAIN + TEST LOOP
# ==========================
for opt in optimizers:

    print(f"\n===================================")
    print(f" YOLO11 Training with {opt}")
    print(f"===================================")

    model = YOLO(base_model)

    lr0 = 0.01 if opt == "SGD" else 0.001

    start_time = time.time()

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        optimizer=opt,
        lr0=lr0,
        workers=1,
        seed=29,
        device=device,
        name=f"yolo11_{opt}"
    )

    training_time = time.time() - start_time

    # ==========================
    # VALIDATION (TRAIN METRICS)
    # ==========================
    metrics = model.val()

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    # ==========================
    # SAVE WEIGHTS
    # ==========================
    best_weight_path = model.trainer.best
    saved_model_path = os.path.join(results_dir, f"best_yolo11_{opt}.pt")

    shutil.copy(best_weight_path, saved_model_path)

    # ==========================
    # MODEL INFO
    # ==========================
    model_size_mb = os.path.getsize(saved_model_path) / (1024 * 1024)
    params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # ==========================
    # SAVE TRAIN RESULTS
    # ==========================
    with open(train_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            opt,
            round(params, 2),
            round(model_size_mb, 2),
            map50,
            map50_95,
            precision,
            recall,
            round(training_time, 2)
        ])

    print(f"\n TRAIN DONE: {opt}")

    # ==========================
    # TEST EVALUATION
    # ==========================
    print(f" Testing {opt}...")

    test_model = YOLO(saved_model_path)

    test_metrics = test_model.val(
        data=data_path,
        split="test",
        imgsz=640,
        batch=1,
        conf=0.01,
        max_det=100,
        device=device
    )

    test_map50 = test_metrics.box.map50
    test_map50_95 = test_metrics.box.map
    test_precision = test_metrics.box.mp
    test_recall = test_metrics.box.mr

    # ==========================
    # SAVE TEST RESULTS
    # ==========================
    with open(test_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            opt,
            round(test_map50, 4),
            round(test_map50_95, 4),
            round(test_precision, 4),
            round(test_recall, 4)
        ])

    print(f" TEST DONE: {opt}")

    torch.cuda.empty_cache()

print("\n ALL EXPERIMENTS COMPLETED")
print(f" Train results: {train_csv}")
print(f" Test results: {test_csv}")