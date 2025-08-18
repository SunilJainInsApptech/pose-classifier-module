from ultralytics import YOLO
import numpy as np
import torch

# Load the model
model = YOLO('/home/sunil/pose-classifier-module/pose-classifier/src/model.pt')

# Print model keypoint names and config
print(model.model.names)  # Should show keypoint names if available
print(model.model.yaml)   # Shows model config, including number of keypoints

# Run inference on a sample image (use any image you have)
results = model('/home/sunil/pose-classifier-module/pose-classifier/camerasystemNVIDIA_training_camera_2025-07-06T20_53_12.274Z.jpg')

# Inspect the keypoints for the first detected person
for r in results:
    if r.keypoints is not None and r.keypoints.xy.numel() > 0:
        print("Keypoints shape:", r.keypoints.xy.shape)  # (num_persons, num_keypoints, 2)
        print("Keypoints for first person:", r.keypoints.xy[0])
        print("Number of keypoints:", r.keypoints.xy.shape[1])
        break

img = np.zeros((640, 640, 3), dtype=np.uint8)  # Dummy image
results = model(img)
output = results[0].keypoints.xy
print("Output shape:", output.shape)
    
    # Access the raw output tensor (logits)
    if hasattr(results, 'raw') and results.raw is not None:
        output = results.raw
        print("Raw output shape:", output.shape)
        if len(output.shape) >= 2:
            print("Number of channels:", output.shape[1])
        else:
            print("Output tensor does not have expected shape.")
    else:
        # Fallback: try to access the tensor from the model directly (for advanced users)
        print("Raw output not available in results. Check model export or use ONNX/TorchScript for direct tensor access.")