import torch
from ultralytics import YOLO

model = YOLO('/home/sunil/pose-classifier-module/pose-classifier/src/yolo11n-pose.pt')  # or your correct .pt file
scripted_model = torch.jit.script(model)
scripted_model.save('/home/sunil/pose-classifier-module/pose-classifier/src/model.pt')

model = torch.jit.load('model.pt')
print(type(model))