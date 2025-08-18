import torch
from ultralytics import YOLO

model = YOLO('/home/sunil/pose-classifier-module/pose-classifier/src/yolo11n-pose.pt')  # or your correct .pt file
model.export(format='torchscript')