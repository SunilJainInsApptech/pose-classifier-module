"""
Standalone script to test the full pose pipeline:
- Connects to Viam robot
- Captures image from camera
- Runs inference on Triton ML model (YOLOv8n-pose)
- Post-processes outputs to extract keypoints
- Runs pose classification using existing joblib model
"""

import asyncio
import joblib
import numpy as np
import logging
from viam.robot.client import RobotClient
from viam.services.mlmodel import MLModelClient
from viam.components.camera import Camera

# --- CONFIGURATION ---
ROBOT_ADDRESS = "camerasystemnvidia-main.niccosz288.viam.cloud"  # Replace with your robot address
ROBOT_API_KEY = "qj8qcg0093x28jtoi90cs5inashvjnd8"
ROBOT_API_KEY_ID = "db33ed99-42fe-46e4-a403-d9af6729dd2b"

TRITON_SERVICE_NAME = "pose-estimate"  # Replace with your ML model service name
POSE_CLASSIFIER_PATH = "../pose_classifier.joblib"  # Adjust path if needed

# Setup logging
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

# --- CONNECTION ---
async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY,
        api_key_id=ROBOT_API_KEY_ID
    )
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)

# --- PREPROCESSING ---
def preprocess_image(image):
    LOGGER.debug("Preprocessing image for model input...")
    # TODO: Implement preprocessing to match your YOLOv8n-pose model's input
    # For example, resize, normalize, convert to numpy array, etc.
    # Return a numpy array or dict as required by your Triton model
    raise NotImplementedError("Implement image preprocessing for your model")

# --- POSTPROCESSING ---
def process_yolo_pose_outputs(outputs, confidence_threshold=0.3):
    LOGGER.debug("Post-processing model outputs...")
    # TODO: Update 'output_tensor_name' to match your Triton model's output
    raw_output = outputs["output_tensor_name"]  # e.g., shape (N, 6+17*3)
    boxes = raw_output[:, :4]
    obj_confidence = raw_output[:, 4]
    keypoints = raw_output[:, 6:]
    detections = []
    keypoints_list = []
    for i, box in enumerate(boxes):
        LOGGER.debug(f"Detection {i}: confidence={obj_confidence[i]}")
        if obj_confidence[i] > confidence_threshold:
            detection = {
                "bbox": box.tolist(),
                "confidence": float(obj_confidence[i]),
                "keypoints": keypoints[i].reshape(-1, 3).tolist()
            }
            detections.append(detection)
            keypoints_list.append(keypoints[i].reshape(-1, 3).tolist())
    LOGGER.debug(f"Total detections above threshold: {len(detections)}")
    return detections, keypoints_list

# --- POSE CLASSIFICATION ---
def classify_pose(pose_classifier, keypoints):
    features = []
    for kp in keypoints:
        features.extend([kp[0], kp[1]])
    LOGGER.debug(f"Classifying pose with features: {features}")
    pose_probs = pose_classifier.predict_proba([features])[0]
    pose_classes = pose_classifier.classes_
    result = {class_name: float(prob) for class_name, prob in zip(pose_classes, pose_probs)}
    LOGGER.debug(f"Classification result: {result}")
    return result

# --- MAIN PIPELINE ---
async def main():
    LOGGER.info("Connecting to robot...")
    robot = await connect()
    LOGGER.info("Connected to robot.")

    LOGGER.info("Loading pose classifier...")
    pose_classifier = joblib.load(POSE_CLASSIFIER_PATH)
    LOGGER.info("Loaded pose classifier.")

    # Get ML model
    ml_model = MLModelClient.from_robot(robot, TRITON_SERVICE_NAME)

    # Dynamically get camera dependencies from robot
    LOGGER.info(f"All resources: {robot.resource_names}")
    camera_names = [res.name for res in robot.resource_names if getattr(res, "resource_type", "") == "camera"]
    LOGGER.info(f"Found camera dependencies: {camera_names}")
    if not camera_names:
        LOGGER.warning("No camera dependencies found in robot config.")
        await robot.close()
        return

    for camera_name in camera_names:
        LOGGER.info(f"Processing camera: {camera_name}")
        camera = Camera.from_robot(robot, camera_name)
        image = await camera.get_image()
        LOGGER.info(f"Captured image from camera: {camera_name}")

        try:
            input_tensors = {"input": preprocess_image(image)}
        except Exception as e:
            LOGGER.error(f"Error in preprocessing image for camera {camera_name}: {e}")
            continue

        try:
            output_tensors = await ml_model.infer(input_tensors)
            LOGGER.info("Inference complete.")
        except Exception as e:
            LOGGER.error(f"Error during inference for camera {camera_name}: {e}")
            continue

        try:
            detections, keypoints_list = process_yolo_pose_outputs(output_tensors)
            LOGGER.info(f"Detections: {len(detections)}")
        except Exception as e:
            LOGGER.error(f"Error in post-processing outputs for camera {camera_name}: {e}")
            continue

        for i, keypoints in enumerate(keypoints_list):
            try:
                pose_result = classify_pose(pose_classifier, keypoints)
                LOGGER.info(f"Camera {camera_name} - Detection {i}: {pose_result}")
            except Exception as e:
                LOGGER.error(f"Error classifying pose for camera {camera_name}, detection {i}: {e}")

    await robot.close()

if __name__ == "__main__":
    asyncio.run(main())
