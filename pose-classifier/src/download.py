import torch
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')  # or your correct .pt file
scripted_model = torch.jit.script(model.model)
scripted_model.save('model.pt')