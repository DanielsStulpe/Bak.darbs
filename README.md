# Pothole Detection Using Computer Vision

## Overview

This project focuses on developing a computer vision model for automatic pothole detection using computer vision object detecton models.

---

## Dataset

Dataset used for experiments:

**Roboflow Public Dataset:**
[https://public.roboflow.com/object-detection/pothole/1](https://public.roboflow.com/object-detection/pothole/1)

The dataset contains annotated images of road potholes prepared for object detection tasks.

All models were trained and evaluated under the same conditions to ensure a fair comparison.

---

## Models Evaluated

The following YOLO "medium" (m) variants were tested:

* YOLOv8m
* YOLOv9m
* YOLOv10m
* YOLOv11m
* YOLOv12m
* YOLO26m

---

## Evaluation Metrics

The models were evaluated using standard object detection metrics:

* **mAP@50** – Mean Average Precision at IoU threshold 0.5
* **mAP@50–95** – Mean Average Precision averaged across IoU thresholds 0.5 to 0.95
* **Precision** – Ratio of correct positive detections to total predicted positives
* **Recall** – Ratio of correct positive detections to total ground truth positives

---

## Experimental Results

| Model       | mAP@50     | mAP@50–95  | Precision | Recall |
| ----------- | ---------- | ---------- | --------- | ------ |
| YOLOv8m     | 0.7931     | 0.5198     | 0.8670    | 0.6667 |
| YOLOv9m     | 0.7746     | 0.4960     | 0.8244    | 0.6818 |
| YOLOv10m    | 0.7670     | 0.4962     | 0.7742    | 0.7000 |
| YOLOv11m    | 0.7785     | 0.5068     | 0.8146    | 0.6788 |
| YOLOv12m    | 0.7736     | 0.4792     | 0.8159    | 0.6715 |
| **YOLO26m** | **0.8043** | **0.5327** | 0.8539    | 0.7000 |

---

## Initial Observations

* **YOLO26m** achieved the highest performance in both mAP@50 and mAP@50–95.
* YOLOv8m demonstrated the highest precision, meaning fewer false positives.
* YOLOv10m and YOLO26m achieved the highest recall (0.70), meaning fewer missed potholes.
* Newer versions (v11, v12) did not significantly outperform earlier stable versions.

Based on these initial experiments, **YOLO26m** currently appears to be the strongest candidate for further optimization and real-world testing.

---

## Project Status

This repository currently contains baseline experimental results. Further experiments, architectural justification, and optimization steps will follow as part of the bachelor thesis development process.
