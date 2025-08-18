import torch
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')  # or your correct .pt file
torch.save(model.state_dict(), 'model.pt')
torch.load('model.pt') 