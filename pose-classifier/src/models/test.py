import torch
model = torch.jit.load('/home/sunil/pose-classifier-module/pose-classifier/src/model.pt')
model.eval()

import numpy as np
input_tensor = torch.from_numpy(np.zeros((1, 3, 640, 640), dtype=np.float32))
with torch.no_grad():
    output = model(input_tensor)
print("Output shape:", output.shape)