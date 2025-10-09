"""
Standalone script to test the full pose pipeline using Roboflow model:
- Connects to Viam robot
- Captures image from camera
- Runs inference on Roboflow model (fall-detection-yg1ru/3)
- Extracts fall detection results directly from the model
"""

import asyncio
import os
import logging
from typing import List, Dict
from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.camera import Camera
from viam.media.video import ViamImage
from inference_sdk import InferenceHTTPClient
import numpy as np
import cv2
import requests
import subprocess
import tempfile
import ast
import shutil
import re
import json
from fall_detection_alerts import FallDetectionAlerts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

# Roboflow Configuration (require env var)
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise SystemExit("ROBOFLOW_API_KEY not set. export ROBOFLOW_API_KEY=<key> and retry.")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "fall-detection-yg1ru/3")
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:9001")  # Default Roboflow inference server

# Viam Configuration
VIAM_API_KEY = os.environ.get("ROBOT_API_KEY")
VIAM_API_KEY_ID = os.environ.get("ROBOT_API_KEY_ID")
VIAM_ROBOT_ADDRESS = os.environ.get("ROBOT_ADDRESS", "sunil-desktop-main.vw6zd12zux.local.viam.cloud:8080")

# Add alert configuration after Viam configuration
ALERT_CONFIG = {
    'twilio_account_sid': os.environ.get('TWILIO_ACCOUNT_SID'),
    'twilio_auth_token': os.environ.get('TWILIO_AUTH_TOKEN'),
    'twilio_from_phone': os.environ.get('TWILIO_FROM_PHONE'),
    'twilio_to_phones': os.environ.get('TWILIO_TO_PHONES', '+19738652226'),
    'fall_confidence_threshold': float(os.environ.get('FALL_CONFIDENCE_THRESHOLD', '0.7')),
    'alert_cooldown_seconds': int(os.environ.get('ALERT_COOLDOWN_SECONDS', '1200')),
    'twilio_notify_service_sid': os.environ.get('TWILIO_NOTIFY_SERVICE_SID'),
    'rigguardian_webhook_url': os.environ.get('RIGGUARDIAN_WEBHOOK_URL', 'https://building-sensor-platform-production.up.railway.app/webhook/fall-alert')
}

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url=INFERENCE_SERVER_URL,
    api_key=ROBOFLOW_API_KEY
)

# Initialize alert service (will be created in main)
alert_service = None

async def connect_to_robot() -> RobotClient:
    """Connect to the Viam robot."""
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID
    )
    return await RobotClient.at_address(VIAM_ROBOT_ADDRESS, opts)

async def get_camera_names(robot: RobotClient) -> List[str]:
    """Get list of available camera names from the robot."""
    camera_names = []
    for resource in robot.resource_names:
        if (resource.namespace == "rdk" and 
            resource.type == "component" and 
            resource.subtype == "camera"):
            camera_names.append(resource.name)
    return camera_names

def viam_image_to_numpy(viam_img: ViamImage) -> np.ndarray:
    """Convert ViamImage to numpy array for Roboflow inference."""
    img_bytes = viam_img.data
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

async def run_roboflow_inference(image: np.ndarray) -> Dict:
    """
    Run inference using local Roboflow inference server.
    Prefer CLI if available; otherwise fall back to HTTP POST.
    Robustly parse CLI output even if there are warnings/progress lines.
    """
    def _post_http(img: np.ndarray) -> Dict:
        success, buf = cv2.imencode('.jpg', img)
        if not success:
            raise RuntimeError("Failed to encode image to JPEG")
        files = {'file': ('image.jpg', buf.tobytes(), 'image/jpeg')}
        params = {'model_id': ROBOFLOW_MODEL_ID, 'api_key': ROBOFLOW_API_KEY}
        url = f"{INFERENCE_SERVER_URL.rstrip('/')}/model/infer"
        resp = requests.post(url, params=params, files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _run_cli(img: np.ndarray) -> Dict:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
            tmp_path = tf.name
        try:
            if not cv2.imwrite(tmp_path, img):
                raise RuntimeError("Failed to write temp image")

            cmd = [
                "inference", "infer",
                "-i", tmp_path,
                "-m", ROBOFLOW_MODEL_ID,
                "--api-key", ROBOFLOW_API_KEY,
                "--host", INFERENCE_SERVER_URL
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            LOGGER.debug("inference CLI rc=%s stdout_len=%d stderr_len=%d", proc.returncode, len(out), len(err))

            # Try parsing the last non-empty line from stdout (CLI prints status lines then the JSON/dict)
            if out:
                for line in reversed(out.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        return json.loads(line)
                    except Exception:
                        pass
                    try:
                        return ast.literal_eval(line)
                    except Exception:
                        pass

            # Try parsing stderr line-by-line
            if err:
                for line in reversed(err.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        return json.loads(line)
                    except Exception:
                        pass
                    try:
                        return ast.literal_eval(line)
                    except Exception:
                        pass

            # Robust fallback: extract the last {...} or [...] block from combined output
            combined = out + "\n" + err
            payload = None
            # try to find last {...}
            start = combined.rfind('{')
            end = combined.rfind('}')
            if start != -1 and end != -1 and start < end:
                payload = combined[start:end+1].strip()
            else:
                # try last [...]
                start = combined.rfind('[')
                end = combined.rfind(']')
                if start != -1 and end != -1 and start < end:
                    payload = combined[start:end+1].strip()

            if payload:
                try:
                    return json.loads(payload)
                except Exception:
                    pass
                try:
                    return ast.literal_eval(payload)
                except Exception:
                    pass

            # Nothing parseable found — raise with captured output for debugging
            raise RuntimeError(
                "inference CLI produced no parseable output. "
                f"rc={proc.returncode}, stdout_len={len(out)}, stderr_len={len(err)}. "
                f"stdout[:500]={out[:500]!r} stderr[:500]={err[:500]!r}"
            )
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    try:
        if shutil.which("inference"):
            result = await asyncio.to_thread(_run_cli, image)
            LOGGER.info(f"Roboflow inference via CLI complete. Found {len(result.get('predictions', []))} predictions")
            return result

        result = await asyncio.to_thread(_post_http, image)
        LOGGER.info(f"Roboflow inference via HTTP complete. Found {len(result.get('predictions', []))} predictions")
        return result

    except requests.exceptions.RequestException as e:
        LOGGER.error(f"HTTP error during Roboflow inference: {e}")
    except Exception as e:
        LOGGER.error(f"Error during Roboflow inference: {e}")
    return {"predictions": []}

def process_roboflow_results(results: Dict, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Process Roboflow detection results.
    
    Args:
        results: Raw results from Roboflow inference
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        List of processed detections with fall classification
    """
    detections = []
    
    for pred in results.get("predictions", []):
        confidence = pred.get("confidence", 0.0)
        
        if confidence < confidence_threshold:
            continue
            
        detection = {
            "class": pred.get("class", "unknown"),
            "confidence": confidence,
            "bbox": {
                "x": pred.get("x"),
                "y": pred.get("y"),
                "width": pred.get("width"),
                "height": pred.get("height")
            },
            "detection_id": pred.get("detection_id", ""),
            "is_fall": pred.get("class", "").lower() in ["fall", "fallen", "person_fallen"]
        }
        
        detections.append(detection)
        
        if detection["is_fall"]:
            LOGGER.warning(f"⚠️ FALL DETECTED with confidence {confidence:.2%}")
        else:
            LOGGER.info(f"Normal pose detected: {detection['class']} ({confidence:.2%})")
    
    return detections

async def process_camera(robot: RobotClient, camera_name: str):
    """Process a single camera feed."""
    try:
        camera = Camera.from_robot(robot, camera_name)
        LOGGER.info(f"Processing camera: {camera_name}")
        
        # Capture image
        viam_img = await camera.get_image()
        LOGGER.info(f"Captured image from camera: {camera_name}")
        
        # Convert to numpy array
        image = viam_image_to_numpy(viam_img)
        LOGGER.info(f"Image shape: {image.shape}")
        
        # Run Roboflow inference
        results = await run_roboflow_inference(image)
        
        # Process results
        detections = process_roboflow_results(results, confidence_threshold=0.6)
        
        # Log results
        LOGGER.info(f"Processed {len(detections)} detections from {camera_name}")
        for det in detections:
            LOGGER.info(f"  - {det['class']}: {det['confidence']:.2%} (Fall: {det['is_fall']})")
        
        # Send alerts for falls
        if alert_service:
            for det in detections:
                if det['is_fall']:
                    LOGGER.info(f"🚨 Triggering fall alert for camera {camera_name}")
                    try:
                        await alert_service.send_fall_alert(
                            camera_name=camera_name,
                            alert_type="fall",
                            person_id=det['detection_id'],
                            confidence=det['confidence'],
                            image=viam_img,
                            metadata={
                                'bbox': det['bbox'],
                                'class': det['class'],
                                'inference_id': results.get('inference_id'),
                                'model_id': ROBOFLOW_MODEL_ID
                            }
                        )
                    except Exception as e:
                        LOGGER.error(f"❌ Failed to send fall alert: {e}")
        else:
            LOGGER.warning("⚠️ Alert service not initialized - no alerts will be sent")
        
        return detections
        
    except Exception as e:
        LOGGER.error(f"Error processing camera {camera_name}: {e}")
        return []

async def main():
    """Main execution function."""
    global alert_service
    
    try:
        # Initialize alert service
        try:
            alert_service = FallDetectionAlerts(ALERT_CONFIG)
            LOGGER.info("✅ Fall detection alert service initialized")
        except Exception as e:
            LOGGER.error(f"⚠️ Failed to initialize alert service: {e}")
            LOGGER.warning("⚠️ Continuing without alerts - detections will still be logged")
        
        # Connect to robot
        LOGGER.info("Connecting to Viam robot...")
        robot = await connect_to_robot()
        LOGGER.info("Connected to robot successfully")
        
        # Get camera names
        camera_names = await get_camera_names(robot)
        LOGGER.info(f"Available cameras: {camera_names}")
        
        if not camera_names:
            LOGGER.error("No cameras found!")
            return
        
        # Process each camera
        all_detections = {}
        for camera_name in camera_names:
            detections = await process_camera(robot, camera_name)
            all_detections[camera_name] = detections
        
        # Summary
        LOGGER.info("\n" + "="*50)
        LOGGER.info("DETECTION SUMMARY")
        LOGGER.info("="*50)
        for camera_name, detections in all_detections.items():
            fall_count = sum(1 for d in detections if d["is_fall"])
            LOGGER.info(f"{camera_name}: {len(detections)} detections, {fall_count} falls")
        
        # Close connection
        await robot.close()
        LOGGER.info("Robot connection closed")
        
    except Exception as e:
        LOGGER.error(f"Error in main: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
