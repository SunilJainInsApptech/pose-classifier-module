from ultralytics import YOLO
import numpy as np

model = YOLO('/home/sunil/pose-classifier-module/pose-classifier/src/model.pt')

results = model('/home/sunil/pose-classifier-module/pose-classifier/camerasystemNVIDIA_training_camera_2025-07-06T20_53_12.274Z.jpg')

for r in results:
    if r.keypoints is not None and r.keypoints.xy.numel() > 0:
        print("Keypoints shape:", r.keypoints.xy.shape)
        print("Number of keypoints:", r.keypoints.xy.shape[1])
        break

img = np.zeros((640, 640, 3), dtype=np.uint8)
results = model(img)
output = results[0].keypoints.xy
print("Output shape:", output.shape)

if hasattr(results, 'raw') and results.raw is not None:
    output = results.raw
    print("Raw output shape:", output.shape)
    if len(output.shape) >= 2:
        print("Number of channels:", output.shape[1])
    else:
        print("Output tensor does not have expected shape.")
else:
    print("Raw output not available in results. Check model export or use ONNX/TorchScript for direct tensor access.")