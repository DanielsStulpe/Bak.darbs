# Pothole Detection Using Computer Vision

## Overview

This project focuses on developing a computer vision model for automatic pothole detection using computer vision object detecton models.

---

## 1. Dataset Description

### 1.1 Dataset Source

Publicly available pothole detection dataset:
[https://public.roboflow.com/object-detection/pothole/1](https://public.roboflow.com/object-detection/pothole/1)

### 1.2 Dataset Characteristics

| Property          | Value                                 |
| ----------------- | ------------------------------------- |
| Total Images      | 665                                   |
| Number of Classes | 1 (Pothole)                           |
| Annotation Type   | Bounding boxes                        |
| Image Resolution  | Resized to 640 × 640                  |
| Data Split        | 70% Train / 20% Validation / 10% Test |

The dataset was randomly partitioned into training, validation, and test sets using a 70:20:10 ratio to ensure unbiased evaluation. All models were trained and evaluated on exactly the same splits to guarantee fair comparison.

---

## 2. Evaluated Models

### 2.1 YOLO (Ultralytics) Models

The following YOLO medium-scale variants were evaluated:

* YOLOv8m
* YOLOv9m
* YOLOv10m
* YOLOv11m
* YOLOv12m
* YOLO26m

All YOLO models were trained using the Ultralytics implementation with default optimization scheduling.

---

### 2.2 Torchvision Detection Models

The following object detection models from Torchvision were evaluated:

* Faster R-CNN
* SSD (Single Shot MultiBox Detector)
* FCOS (Fully Convolutional One-Stage Detector)
* RetinaNet

These models were trained using the official Torchvision detection training configuration, with learning rate adjusted proportionally to batch size.

---

## 3. Experimental Setup

All experiments were conducted under identical preprocessing and training conditions.

### 3.1 Global Training Configuration

| Parameter           | Value        |
| ------------------- | ------------ |
| Number of Epochs    | 100          |
| Input Image Size    | 640 × 640    |
| Optimizer           | SGD          |
| Evaluation Protocol | COCO Metrics |

---

## 4. Hyperparameter Configuration

For YOLO models was used standart learning rate 0.001 as Ultralytics implementaton adjusts learning rate for batch size automatically.

For Torchvision models, learning rate scaling followed the linear scaling rule relative to the reference configuration (LR = 0.02 for batch size 16):

LR_adjusted = 0.02 × (Batch Size / 16)

### 4.1 Training Parameters per Model

| Model        | Framework   | Batch Size | Initial LR | LR Strategy                     |
| ------------ | ----------- | ---------- | ---------- | ------------------------------- |
| YOLOv8m      | Ultralytics | 4          | 0.001      | Adaptive (automatic scheduling) |
| YOLOv9m      | Ultralytics | 4          | 0.001      | Adaptive                        |
| YOLOv10m     | Ultralytics | 4          | 0.001      | Adaptive                        |
| YOLOv11m     | Ultralytics | 4          | 0.001      | Adaptive                        |
| YOLOv12m     | Ultralytics | 4          | 0.001      | Adaptive                        |
| YOLO26m      | Ultralytics | 4          | 0.001      | Adaptive                        |
| Faster R-CNN | Torchvision | 4          | 0.005      | Scaled from 0.02                |
| SSD          | Torchvision | 4          | 0.005      | Scaled from 0.02                |
| FCOS         | Torchvision | 4          | 0.005      | Scaled from 0.02                |
| RetinaNet    | Torchvision | 2          | 0.0025     | Scaled from 0.02                |

---

## 5. Evaluation Metrics

Model performance was evaluated on the held-out test set using COCO-style metrics:

| Metric    | Description                                                   |
| --------- | ------------------------------------------------------------- |
| mAP@50    | Mean Average Precision at IoU = 0.5                           |
| mAP@50–95 | Mean Average Precision averaged over IoU thresholds 0.50–0.95 |
| AR        | Average Recall                                                |

COCO metrics provide a comprehensive assessment of both localization and classification performance across multiple IoU thresholds.

---

## 6. Experimental Results

### 6.1 YOLO Models

| Model    | mAP@50 | mAP@50–95 | AR     |
| -------- | ------ | --------- | ------ |
| YOLOv8m  | 0.7931 | 0.5198    | 0.6667 |
| YOLOv9m  | 0.7746 | 0.4960    | 0.6818 |
| YOLOv10m | 0.7670 | 0.4962    | 0.7000 |
| YOLOv11m | 0.7785 | 0.5068    | 0.6788 |
| YOLOv12m | 0.7736 | 0.4792    | 0.6715 |
| YOLO26m  | 0.8043 | 0.5327    | 0.7000 |

---

### 6.2 Torchvision Models

(To be updated after full evaluation)

| Model        | mAP@50 | mAP@50–95 | AR     |
| ------------ | ------ | --------- | ------ |
| Faster R-CNN | 0.7525 | 0.4715    | 0.5903 |
| SSD          | 0.7525 | 0.4715    | 0.5903 |
| FCOS         | 0.7525 | 0.4715    | 0.5903 |
| RetinaNet    | 0.7525 | 0.4715    | 0.5903 |

---

## 7. Preliminary Analysis

* YOLO26m achieved the highest mAP@50 and mAP@50–95 among evaluated YOLO models.
* YOLOv8m demonstrated the highest precision, indicating fewer false positives.
* YOLOv10m and YOLO26m achieved the highest recall (0.70), indicating fewer missed detections.
* Later YOLO versions (v11, v12) did not significantly outperform earlier stable variants under identical training conditions.

The inclusion of Faster R-CNN, SSD, FCOS, and RetinaNet enables architectural comparison between:

* Two-stage detectors (region proposal based)
* Anchor-based one-stage detectors
* Anchor-free one-stage detectors

---

## 8. Experimental Environment

All experiments were conducted on the following hardware and software configuration:

### 8.1 Hardware

| Component    | Specification           |
| ------------ | ----------------------- |
| GPU          | NVIDIA GeForce GTX 1080 |
| CUDA         | Version 12.2            |

### 8.2 Software

| Library         | Version |
| --------------- | ------- |
| Ultralytics     | 8.4.17  |
| PyTorch (torch) | 2.4.1   |
| Torchvision     | 0.19.1  |
| pycocotools     | 2.0.7   |

All models were trained using the same hardware to ensure consistent computational conditions. No distributed or multi-GPU training was used. Random dataset splits were fixed prior to training to maintain reproducibility across experiments.


## 9. Project Status

This repository currently contains baseline experimental results. Further experiments, architectural justification, and optimization steps will follow as part of the bachelor thesis development process.