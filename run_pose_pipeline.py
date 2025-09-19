"""
Standalone script to test the full pose pipeline:
- Connects to Viam robot
- Captures image from camera
- Runs inference on Triton ML model (YOLO11n-pose)
- Post-processes outputs to extract keypoints
- Runs pose classification using existing joblib model
"""

import asyncio
import joblib
import numpy as np
import logging
import cv2
import os
from viam.robot.client import RobotClient
from viam.services.mlmodel import MLModelClient
from viam.components.camera import Camera
from viam.media.video import ViamImage 
import io
from fall_detection_alerts import FallDetectionAlerts
from after_hours_alerts import AfterHoursAlerts
import json
import datetime
import time

# --- CONFIGURATION ---
ROBOT_ADDRESS = os.environ.get('ROBOT_ADDRESS', 'camerasystemnvidia-main.niccosz288.viam.cloud')  
ROBOT_API_KEY = os.environ.get('ROBOT_API_KEY', 'qj8qcg0093x28jtoi90cs5inashvjnd8')
ROBOT_API_KEY_ID = os.environ.get('ROBOT_API_KEY_ID', 'db33ed99-42fe-46e4-a403-d9af6729dd2b')

TRITON_SERVICE_NAME = "pose-estimate"  
POSE_CLASSIFIER_PATH = "/home/sunil/pose-classifier-module/pose_classifier_20250916.joblib"  

# Setup logging (configurable via LOG_LEVEL env var)
LOG_LEVEL = 'DEBUG'
numeric_level = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(level=numeric_level)
LOGGER = logging.getLogger(__name__)


async def start_health_server(host: str = '127.0.0.1', port: int = 8000):
    try:
        from aiohttp import web
    except Exception:
        LOGGER.warning("aiohttp not installed - health endpoint will not be available. Install with: pip install aiohttp")
        return None

    app = web.Application()
    start_time = time.time()

    async def _health(request):
        return web.json_response({
            'status': 'ok',
            'uptime_seconds': time.time() - start_time
        })

    app.router.add_get('/health', _health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    LOGGER.info(f"Health endpoint available at http://{host}:{port}/health")
    return runner

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
    # Convert ViamImage to numpy array (assuming image is ViamImage or bytes)
    if hasattr(image, 'data'):
        img_bytes = image.data
    else:
        img_bytes = image  # fallback if already bytes
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    # Resize to 640x640
    img = cv2.resize(img, (640, 640))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Change shape to [3, 640, 640]
    img = np.transpose(img, (2, 0, 1))
    # Add batch dimension [1, 3, 640, 640]
    img = np.expand_dims(img, axis=0)
    # Triton may expect a dict with the input name as key
    return img

# --- POSTPROCESSING ---
def process_yolo_pose_outputs(outputs, confidence_threshold=0.65, iou_threshold=0.45):
    LOGGER.debug("Post-processing model outputs with NMS...")
    raw_output = outputs["keypoints"]  # shape: (1, 57, 8400) expected for 17 keypoints
    LOGGER.debug(f"raw_output.shape: {raw_output.shape}")
    raw_output = np.squeeze(raw_output, axis=0)  # shape: (C, 8400)
    # LOGGER.debug(f"raw_output.shape after squeeze: {raw_output.shape}")
    # LOGGER.debug(f"First detection full channel vector: {raw_output[:,0]}")
    num_channels = raw_output.shape[0]

    # For single-class pose models with 17 keypoints and 56 channels:
    # 4 bbox + 1 obj_conf + 51 keypoints (17*3)
    if num_channels == 56:
        num_keypoints = 17
        LOGGER.info(f"Detected {num_keypoints} keypoints per detection (from output tensor, 56 channels, single-class model).")
        boxes = raw_output[0:4, :].T  # (N, 4)
        obj_conf = raw_output[4, :]
        scores = obj_conf  # No class_conf, single class
        keypoints = raw_output[5:5+num_keypoints*3, :].T  # (N, 51)
        LOGGER.debug(f"keypoints.shape: {keypoints.shape}")
        if keypoints.shape[1] > 0:
            # LOGGER.debug(f"First 20 keypoint values of first detection: {keypoints[0][:20]}")
            pass
    else:
        num_keypoint_channels = num_channels - 6
        LOGGER.info(f"Total channels: {num_channels}, keypoint channels: {num_keypoint_channels}")
        if num_keypoint_channels % 3 == 0:
            num_keypoints = num_keypoint_channels // 3
            LOGGER.info(f"Detected {num_keypoints} keypoints per detection (from output tensor).")
        else:
            # LOGGER.error(f"Keypoint channel count {num_keypoint_channels} is not divisible by 3! Model export or config is incorrect. Post-processing will not proceed.")
            return [], []
        # --- Keypoint extraction ---
        boxes = raw_output[0:4, :].T  # (N, 4)
        obj_conf = raw_output[4, :]
        class_conf = raw_output[5, :] if raw_output.shape[0] > 5 else np.ones_like(obj_conf)
        scores = obj_conf * class_conf
        keypoints = raw_output[6:6+num_keypoints*3, :].T  # (N, num_keypoints*3)
        LOGGER.debug(f"keypoints.shape: {keypoints.shape}")
        if keypoints.shape[1] > 0:
            LOGGER.debug(f"First 20 keypoint values of first detection: {keypoints[0][:20]}")

    # Filter by confidence threshold
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    keypoints = keypoints[mask]
    if boxes.shape[0] == 0:
        return [], []

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=scores.tolist(),
        score_threshold=confidence_threshold,
        nms_threshold=iou_threshold
    )
    if len(indices) == 0:
        return [], []
    indices = np.array(indices).flatten()

    detections = []
    keypoints_list = []
    for idx in indices:
        kps = keypoints[idx]
        if isinstance(kps, float) or isinstance(kps, np.floating):
            LOGGER.error(f"Detection {idx} has invalid keypoints: {kps}")
            continue
        # Always reshape to (num_keypoints, 3)
        if kps.ndim == 1 and kps.shape[0] == num_keypoints*3:
            kps = kps.reshape(num_keypoints, 3)
        elif kps.ndim == 2 and kps.shape == (num_keypoints, 3):
            pass  # already correct
        else:
            LOGGER.error(f"Detection {idx} has unexpected keypoints shape after slice: {kps.shape}")
            continue
        detection = {
            "bbox": boxes_xyxy[idx].tolist(),
            "confidence": float(scores[idx]),
            "keypoints": kps.tolist()
        }
        detections.append(detection)
        keypoints_list.append(detection["keypoints"])
    LOGGER.debug(f"Total detections after NMS: {len(detections)}")
    return detections, keypoints_list

# --- POSE CLASSIFICATION ---
def classify_pose(pose_classifier, keypoints, frame_width, frame_height):
    # keypoints: (N, 2) array, N should be 17
    if keypoints is None or len(keypoints) < 17:
        LOGGER.warning("Not enough keypoints for classification.")
        return "unknown"
    keypoints_np = np.array(keypoints[:17])[:, :2]  # Use only x, y for each keypoint
    normalized_keypoints = keypoints_np / np.array([frame_width, frame_height])
    features = normalized_keypoints.flatten()
    pose_label = pose_classifier.predict([features])[0]
    # Ensure label is a plain Python string for JSON serialization
    if hasattr(pose_label, 'item'):
        pose_label = pose_label.item()
    pose_label = str(pose_label)
    LOGGER.debug(f"Predicted pose label: {pose_label}")
    return {"label": pose_label}

# --- MAIN PIPELINE ---
async def main():
    LOGGER.info("Connecting to robot...")
    robot = await connect()
    LOGGER.info("Connected to robot.")

    LOGGER.info("Loading pose classifier...")
    # Allow overriding via environment variable, and try a few sensible fallbacks
    candidate = '/home/sunil/pose-classifier-module/pose_classifier_svc.joblib' or POSE_CLASSIFIER_PATH
    fallback_paths = [
        candidate,
        os.path.join(os.path.dirname(__file__), 'pose_classifier.joblib'),
        os.path.join(os.path.dirname(__file__), '..', 'pose_classifier.joblib'),
        '/home/sunil/pose-classifier-module/pose_classifier.joblib',
        '/home/sunil/pose-classifier-module/pose_classifier.joblib'
    ]
    pose_classifier = None
    for p in fallback_paths:
        try:
            if p and os.path.exists(p):
                LOGGER.info(f"Attempting to load pose classifier from: {p}")
                pose_classifier = joblib.load(p)
                LOGGER.info(f"Loaded pose classifier from: {p}")
                break
        except Exception as e:
            LOGGER.warning(f"Failed to load pose classifier from {p}: {e}")

    if pose_classifier is None:
        LOGGER.error("Could not find or load a pose classifier. Set POSE_CLASSIFIER_PATH env var or place 'pose_classifier.joblib' next to the script.")
        raise SystemExit(1)

    # Get ML model
    ml_model = MLModelClient.from_robot(robot, TRITON_SERVICE_NAME)


    # --- AUTOMATIC CAMERA DISCOVERY ---
    def get_camera_names(robot):
        camera_names = []
        for resource_name in robot.resource_names:
            if (resource_name.namespace == "rdk" and 
                resource_name.type == "component" and 
                resource_name.subtype == "camera"):
                camera_names.append(resource_name.name)
        return camera_names

    camera_names = get_camera_names(robot)
    LOGGER.info(f"Detected cameras: {camera_names}")

    # Initialize fall detection alert service
    fall_alerts = FallDetectionAlerts({
        # Fill with your config or load from file/env
        # 'twilio_account_sid': 'your_sid',
        # 'twilio_auth_token': 'your_token',
        # 'twilio_from_phone': '+1234567890',
        # 'twilio_to_phones': ['+1987654321'],
        'alert_cooldown_seconds': 1200,  # Check if this value is set here
    })

    # Initialize after-hours alert service (disabled by default)
    # after_hours_alerts = AfterHoursAlerts({
    #     # Fill with your config or load from file/env
    # })

    # --- TEST IMAGE MODE ---
    USE_TEST_IMAGE = False  # Set to False to use the Viam camera
    TEST_IMAGE_PATH = "camerasystemNVIDIA_training_camera_2025-07-06T20_53_12.274Z.jpg"  # Path to your test image

    # Polling loop configuration (seconds)
    poll_interval = 5

    output_detections = []

    # PID file handling
    pid_file = '/tmp/run_pose_pipeline.pid'
    try:
        with open(pid_file, 'w') as pf:
            pf.write(str(os.getpid()))
        LOGGER.info(f"Wrote PID file: {pid_file}")
    except Exception as e:
        LOGGER.warning(f"Could not write PID file {pid_file}: {e}")

    # Start health endpoint if requested
    health_runner = None
    if os.environ.get('ENABLE_HEALTH_ENDPOINT', '1') == '1':
        health_host = os.environ.get('HEALTH_HOST', '127.0.0.1')
        health_port = int(os.environ.get('HEALTH_PORT', '8000'))
        try:
            health_runner = await start_health_server(health_host, health_port)
        except Exception as e:
            LOGGER.warning(f"Failed to start health server: {e}")

    try:
        while True:
            # Re-discover cameras each loop to handle dynamic robots
            camera_names = get_camera_names(robot)
            LOGGER.debug(f"Polling cameras: {camera_names}")

            for camera_name in camera_names:
                LOGGER.info(f"Processing camera: {camera_name}")
                if USE_TEST_IMAGE:
                    # Load test image from disk
                    img = cv2.imread(TEST_IMAGE_PATH)
                    if img is None:
                        LOGGER.error(f"Failed to load test image: {TEST_IMAGE_PATH}")
                        continue
                    _, img_bytes = cv2.imencode('.jpg', img)
                    image = ViamImage(data=img_bytes.tobytes(), mime_type='image/jpeg')
                    LOGGER.info(f"Loaded test image from: {TEST_IMAGE_PATH}")
                else:
                    try:
                        camera = Camera.from_robot(robot, camera_name)
                        image = await camera.get_image()
                        LOGGER.info(f"Captured image from camera: {camera_name}")
                    except Exception as e:
                        LOGGER.error(f"Failed to get image from camera {camera_name}: {e}")
                        continue

                try:
                    input_tensors = {"input": preprocess_image(image)}
                except Exception as e:
                    LOGGER.error(f"Error in preprocessing image for camera {camera_name}: {e}")
                    continue

                try:
                    output_tensors = await ml_model.infer(input_tensors)
                    LOGGER.info("Inference complete.")
                    # --- Debugging: Log output tensor info ---
                    for k, v in output_tensors.items():
                        LOGGER.info(f"Output key: {k}, shape: {getattr(v, 'shape', type(v))}, dtype: {getattr(v, 'dtype', type(v))}")
                        if hasattr(v, 'shape') and len(v.shape) >= 2:
                            LOGGER.info(f"Number of channels in '{k}': {v.shape[1]}")
                    LOGGER.info(f"Tensor mappings: {list(output_tensors.keys())}")
                except Exception as e:
                    LOGGER.error(f"Error during inference for camera {camera_name}: {e}")
                    continue

                try:
                    detections, keypoints_list = process_yolo_pose_outputs(output_tensors)
                    LOGGER.info(f"Detections: {len(detections)}")
                except Exception as e:
                    LOGGER.error(f"Error in post-processing outputs for camera {camera_name}: {e}")
                    continue

                # Timestamp for this image
                image_timestamp = datetime.datetime.now().isoformat()

                for i, detection in enumerate(detections):
                    try:
                        keypoints = detection.get('keypoints')
                        # Get frame dimensions from the preprocessed image (should be 640x640)
                        frame_width, frame_height = 640, 640
                        pose_result = classify_pose(pose_classifier, keypoints, frame_width, frame_height)
                        LOGGER.info(f"Camera {camera_name} - Detection {i}: {pose_result}")

                        if isinstance(pose_result, dict):
                            pose_label = pose_result.get('label', '')
                            pose_confidence_raw = pose_result.get('confidence', 1.0 if pose_label == 'fallen' else 0.0)
                        else:
                            pose_label = pose_result
                            pose_confidence_raw = 1.0 if pose_label == 'fallen' else 0.0
                        try:
                            pose_confidence = float(pose_confidence_raw)
                        except Exception:
                            pose_confidence = 1.0 if pose_label == 'fallen' else 0.0
                        fall_confidence = pose_confidence

                        # Send fall alert if confidence is high (cooldown handled in fall_detection_alerts)
                        if fall_confidence > 0.7:
                            person_id = str(i)
                            LOGGER.info(f"Fall detected for {camera_name} with confidence {fall_confidence:.3f}")
                            """
                            alert_sent = await fall_alerts.send_fall_alert(
                                camera_name=camera_name,
                                alert_type="fall",
                                person_id=person_id,
                                confidence=pose_confidence,
                                image=image,
                                metadata={"probabilities": pose_result}
                            )
                            if alert_sent:
                                LOGGER.info(f"✅ Fall alert sent for detection {i} on camera {camera_name}")
                            else:
                                LOGGER.debug(f"⏳ Fall alert not sent (likely due to cooldown or low confidence)")
                            """
                        # Add detection to output list for objectfilter-camera, including pose classification label
                        output_detections.append({
                            "image_time": image_timestamp,
                            "camera_name": camera_name,
                            "detection_number": i,
                            "label": "person",
                            "pose_label": pose_label,
                            "confidence": pose_confidence,
                            "bbox": detection.get("bbox"),
                            "keypoints": keypoints
                        })
                    except Exception as e:
                        LOGGER.error(f"Error classifying pose for camera {camera_name}, detection {i}: {e}")

            # Print or persist detections (keeps behavior similar to single-shot)
            if output_detections:
                print(json.dumps(output_detections, indent=2))
                # Clear after printing to avoid duplicate outputs in next loop
                output_detections.clear()

            # Sleep until next poll
            await asyncio.sleep(poll_interval)

    except asyncio.CancelledError:
        LOGGER.info("Main loop cancelled, shutting down")
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user, shutting down")
    finally:
        # Cleanup health endpoint
        if health_runner:
            try:
                await health_runner.cleanup()
                LOGGER.info("Health endpoint stopped")
            except Exception:
                pass

        try:
            await robot.close()
        except Exception:
            pass
        # Remove PID file
        try:
            if os.path.exists(pid_file):
                os.remove(pid_file)
                LOGGER.info(f"Removed PID file: {pid_file}")
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
