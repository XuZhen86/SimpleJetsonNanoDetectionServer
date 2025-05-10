#!/bin/bash

# Print command traces before executing the command.
set -o xtrace

mkdir -p /app/data/yolo11/models
cd /app/data/yolo11/models

mkdir -p pytorch
mkdir -p onnx
mkdir -p tensorrt

# It appears Frigate always sends image of size 320x320.
# In case the image size mismatches, the YOLO library auto-converts the image size before the prediction.
imgsz=320

# It may trigger AutoUpdate and it may requires to re-run the command.
for model in yolo11n yolo11s yolo11m; do
  # It takes 6-7 mins for each export.
  yolo export model=pytorch/$model.pt format=engine half=true imgsz=$imgsz

  mv pytorch/$model.onnx   onnx/$model-$imgsz-fp16.onnx
  mv pytorch/$model.engine tensorrt/$model-$imgsz-fp16.engine
done
