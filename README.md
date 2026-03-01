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

For YOLO models was used standart learning rate 0.01 as Ultralytics implementaton adjusts learning rate for batch size automatically.

For Torchvision models, learning rate scaling followed the linear scaling rule relative to the reference configurations:

Faster_R-CNN_LR = 0.001 × (Batch Size / 1)

SSD_LR= 0.001 × (Batch Size / 16)

FCOS_LR = 0.01 × (Batch Size / 16)

RetinaNet_LR = 0.01 × (Batch Size / 16)

### 4.1 Training Parameters per Model

| Model        | Framework   | Batch Size |  LR        |
| ------------ | ----------- | ---------- | ---------- | 
| YOLOv8m      | Ultralytics | 4          | 0.01       | 
| YOLOv9m      | Ultralytics | 4          | 0.01       | 
| YOLOv10m     | Ultralytics | 4          | 0.01       |
| YOLOv11m     | Ultralytics | 4          | 0.01       |
| YOLOv12m     | Ultralytics | 4          | 0.01       | 
| YOLO26m      | Ultralytics | 4          | 0.01       | 
| Faster R-CNN | Torchvision | 4          | 0.004      |
| SSD          | Torchvision | 4          | 0.00025    |
| FCOS         | Torchvision | 4          | 0.0025     |
| RetinaNet    | Torchvision | 2          | 0.00125    |

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
| YOLOv8m  | 0.7760 | 0.5383    | 0.7321 |
| YOLOv9m  | 0.8290 | 0.5972    | 0.6818 |
| YOLOv10m | 0.7854 | 0.5419    | 0.7078 |
| YOLOv11m | 0.8130 | 0.5540    | 0.7690 |
| YOLOv12m | 0.8275 | 0.5829    | 0.7575 |
| YOLO26m  | 0.8453 | 0.5930    | 0.7532 |

---

### 6.2 Torchvision Models

(To be updated after full evaluation)

| Model        | mAP@50 | mAP@50–95 | AR     |
| ------------ | ------ | --------- | ------ |
| Faster R-CNN | 0.7580 | 0.4759    | 0.5610 |
| SSD          | 0.6316 | 0.3533    | 0.5104 |
| FCOS         | 0.7754 | 0.4955    | 0.6175 |
| RetinaNet    | 0.7404 | 0.4858    | 0.6156 |

---

## 7. Results Analysis

### 7.1 YOLO Models

Among the evaluated YOLO variants, YOLO26m achieved the highest overall performance, obtaining the best mAP@50 and mAP@50–95 scores. YOLOv9m and YOLOv12m also demonstrated strong results, particularly under stricter IoU thresholds.

Although newer YOLO versions showed incremental improvements over YOLOv8m, the performance differences between the variants remained moderate when trained under identical conditions.

### 7.2 Torchvision Models

Among the Torchvision-based detectors, FCOS achieved the highest overall performance. RetinaNet and Faster R-CNN showed competitive but slightly lower results, while SSD obtained the lowest scores across all evaluation metrics.

Overall, YOLO-based architectures outperformed Torchvision implementations within the current experimental setup.

### 7.3 Implication for Further Work

Based on these results, YOLO26m represents the most promising architecture for further investigation. In the second phase of this work, this model will be selected for systematic hyperparameter optimization to evaluate its maximum achievable performance under refined training configurations.

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