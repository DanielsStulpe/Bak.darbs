#!/bin/bash

echo "Starting Faster R-CNN training..."
python train_faster_rcnn.py

echo "Starting FCOS training..."
python train_fcos.py

echo "Starting SDD training..."
python train_sdd.py

echo "Starting RetinaNet model..."
python train_retinanet.py

echo "All trainings finished!"