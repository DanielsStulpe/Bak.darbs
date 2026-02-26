#!/bin/bash
set -e

echo "Starting Yolo models training..."
python train_yolo.py > yolo.log 2>&1

sleep 30

echo "Starting Yolo models testing..."
python test_yolo.py > yolo_test.log 2>&1

echo "All processes finished!"