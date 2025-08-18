import torch
import numpy as np

# Load the PyTorch model directly (not TorchScript)
model = torch.load('/home/sunil/pose-classifier-module/pose-classifier/src/yolo11n-pose.pt', map_location='cpu')
model.eval()

# Create a dummy input tensor (batch size 1, 3 channels, 640x640)
input_tensor = torch.from_numpy(np.zeros((1, 3, 640, 640), dtype=np.float32))

# Run inference and print the output shape
with torch.no_grad():
    output = model(input_tensor)
    print("Raw output shape from yolo11n-pose.pt:", output.shape)
    if len(output.shape) >= 2:
        print("Number of channels:", output.shape[1])
    else:
        print("Output tensor does not have expected shape.")
import torch
model = torch.jit.load('/home/sunil/pose-classifier-module/pose-classifier/src/model.pt')
model.eval()

import numpy as np
input_tensor = torch.from_numpy(np.zeros((1, 3, 640, 640), dtype=np.float32))
with torch.no_grad():
    output = model(input_tensor)
print("Output shape:", output.shape)