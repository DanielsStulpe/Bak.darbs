from ultralytics import YOLO
import os
import csv
import torch
import logging

# ==========================
# CONFIG
# ==========================
model_name = "best_yolo26_1.pt"
results_dir = "yolo26_batch_sizes_results"

data_path = "pothole_dataset_yolo/data.yaml"

output_csv = os.path.join(results_dir, "single_test_result.csv")
log_file = os.path.join(results_dir, "single_test.log")

# Auto device selection (SAFE)
gpu_count = torch.cuda.device_count()
if gpu_count > 1:
    device = "0,1"
elif gpu_count == 1:
    device = "0"
else:
    device = "cpu"

# ==========================
# LOGGER
# ==========================
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info("=== STARTING SINGLE MODEL TEST ===")

# ==========================
# CHECK MODEL
# ==========================
model_path = os.path.join(results_dir, model_name)

if not os.path.exists(model_path):
    logging.error(f"Model not found: {model_path}")
    exit()

# ==========================
# CSV HEADER
# ==========================
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall"
    ])

# ==========================
# LOAD MODEL
# ==========================
logging.info(f"Loading model: {model_name}")
model = YOLO(model_path)

# ==========================
# TEST
# ==========================
logging.info("Running test evaluation...")

metrics = model.val(
    data=data_path,
    split="test",
    imgsz=640,
    batch=1,
    conf=0.01,
    max_det=100,
    device=device
)

# ==========================
# METRICS
# ==========================
map50 = metrics.box.map50
map50_95 = metrics.box.map
precision = metrics.box.mp
recall = metrics.box.mr

logging.info(f"mAP50: {map50:.4f}")
logging.info(f"mAP50-95: {map50_95:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")

# ==========================
# SAVE CSV
# ==========================
with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        model_name,
        round(map50, 4),
        round(map50_95, 4),
        round(precision, 4),
        round(recall, 4)
    ])

logging.info("=== TEST COMPLETED ===")
print(f"Results saved to: {output_csv}")
print(f"Logs saved to: {log_file}")