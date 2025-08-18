from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')  # or your custom .pt file
model.export(format='onnx', dynamic=False, imgsz=640)