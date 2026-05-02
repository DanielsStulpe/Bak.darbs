from ultralytics import YOLO
import time
import shutil
import csv
import os
import torch
import logging

# ==========================
# CONFIG
# ==========================
base_model = "yolo26m.pt"

BATCH_SIZES = [2, 4, 8, 16, 32]

data_path = "pothole_dataset_yolo/data.yaml"
imgsz = 640
epochs = 300

results_dir = "yolo26_batch_sizes_results"
os.makedirs(results_dir, exist_ok=True)

train_csv = os.path.join(results_dir, "train_results.csv")
test_csv = os.path.join(results_dir, "test_results.csv")
log_file = os.path.join(results_dir, "experiment.log")

device = [0, 1] if torch.cuda.is_available() else "cpu"

# ==========================
# LOGGER SETUP
# ==========================
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info("=== STARTING YOLO26 BATCH SIZE EXPERIMENT ===")

# ==========================
# CSV HEADERS
# ==========================
with open(train_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Batch_size",
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
        "Batch_size",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall"
    ])

# ==========================
# LOOP
# ==========================
for batch_size in BATCH_SIZES:

    logging.info(f"\n--- Training with batch size {batch_size} ---")

    try:
        model = YOLO(base_model)

        start_time = time.time()

        model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            optimizer="NAdam",
            lr0=0.001,
            workers=4,
            seed=0,
            device=device,
            name=f"yolo26_bs{batch_size}"
        )

        training_time = time.time() - start_time

        # ==========================
        # VALIDATION
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

        if not best_weight_path or not os.path.exists(best_weight_path):
            logging.warning(f"No best weights for batch {batch_size}")
            continue

        saved_model_path = os.path.join(results_dir, f"best_yolo26_{batch_size}.pt")
        shutil.copy(best_weight_path, saved_model_path)

        # ==========================
        # MODEL INFO
        # ==========================
        model_size_mb = os.path.getsize(saved_model_path) / (1024 * 1024)
        params = sum(p.numel() for p in model.model.parameters()) / 1e6

        # ==========================
        # SAVE TRAIN CSV
        # ==========================
        with open(train_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                batch_size,
                round(params, 2),
                round(model_size_mb, 2),
                map50,
                map50_95,
                precision,
                recall,
                round(training_time, 2)
            ])

        logging.info(f"Train done for batch {batch_size}")

        # ==========================
        # TEST
        # ==========================
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
        # SAVE TEST CSV
        # ==========================
        with open(test_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                batch_size,
                round(test_map50, 4),
                round(test_map50_95, 4),
                round(test_precision, 4),
                round(test_recall, 4)
            ])

        logging.info(f"Test done for batch {batch_size}")

    except RuntimeError as e:
        logging.error(f"Runtime error at batch {batch_size}: {str(e)}")
        continue

    except Exception as e:
        logging.error(f"Unexpected error at batch {batch_size}: {str(e)}")
        continue

    finally:
        torch.cuda.empty_cache()

logging.info("=== ALL EXPERIMENTS COMPLETED ===")
print(f"Logs saved to: {log_file}")