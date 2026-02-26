#!/bin/bash
set -e

echo "Starting Faster R-CNN training..."
python train_faster_rcnn.py > faster_rcnn.log 2>&1

sleep 30

echo "Starting FCOS training..."
python train_fcos.py > fcos.log 2>&1

sleep 30

echo "Starting SsD training..."
python train_ssd.py > ssd.log 2>&1

sleep 30

echo "Starting RetinaNet model..."
python train_retinanet.py > retinanet.log 2>&1

echo "All trainings finished!"