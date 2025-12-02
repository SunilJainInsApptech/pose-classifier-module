"""
Standalone script to test the full pose pipeline using Roboflow model:
- Connects to RTSP streams using OpenCV/GStreamer
- Captures image from a stream
- Runs inference on Roboflow model (fall-detection-yg1ru/3)
- Extracts fall detection results directly from the model
"""

import asyncio
import os
import logging
from typing import List, Dict, Optional
from viam.media.video import ViamImage
from inference_sdk import InferenceHTTPClient
import numpy as np
import cv2
import subprocess
import tempfile
import ast
import shutil
import re
import json
from fall_detection_alerts import FallDetectionAlerts
from after_hours_detection_alerts import AfterHoursDetectionAlerts
from datetime import datetime
import httpx # Add this import

# New imports for dynamic load of the copy module
import importlib.util

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

# NEW: Address of the capture service running on the Jetson
# Replace <JETSON_IP_ADDRESS> with the actual IP of your Jetson
CAPTURE_SERVICE_URL = os.environ.get("CAPTURE_SERVICE_URL", "http://localhost:8001")

# RTSP Stream Configuration (used to know which cameras to query)
RTSP_STREAMS = {
    'Lobby_Center_North': 'rtsp://192.168.1.200:554/chID=25&streamType=sub',
    'CPW_Awning_N_Facing': 'rtsp://192.168.1.200:554/chID=16&streamType=sub',
    'Roof_Front_East_Facing': 'rtsp://192.168.1.200:554/chID=01&streamType=sub',
    # Add other stream names and URLs here
}

# REMOVED: GSTREAMER_PIPELINE and _caps global cache

# Add alert configuration after Viam configuration
ALERT_CONFIG = {
    'twilio_account_sid': os.environ.get('TWILIO_ACCOUNT_SID'),
    'twilio_auth_token': os.environ.get('TWILIO_AUTH_TOKEN'),
    'twilio_from_phone': os.environ.get('TWILIO_FROM_PHONE'),
    'twilio_to_phones': os.environ.get('TWILIO_TO_PHONES', '+19738652226'),
    'fall_confidence_threshold': float(os.environ.get('FALL_CONFIDENCE_THRESHOLD', '0.85')),
    'alert_cooldown_seconds': int(os.environ.get('ALERT_COOLDOWN_SECONDS', '1200')),
    'twilio_notify_service_sid': os.environ.get('TWILIO_NOTIFY_SERVICE_SID'),
    'rigguardian_webhook_url': os.environ.get('RIGGUARDIAN_WEBHOOK_URL', 'https://building-sensor-platform-production.up.railway.app/webhook/fall-alert')
}

# New: cameras that should trigger a after hours alert when a "person" is detected
AFTER_HOURS_CAMERAS = {
    "Services_Area",
    "Service_Staircase_Top",
    "Elevator_South",
    "Courtyard_1",
    "Service_Staircase_Bottom",
    "Courtyard_2",
    "North_Elevator",
    "South_Elevator",
}

# Monitoring configuration (seconds between cycles, retry behavior)
MONITORING_INTERVAL_SECONDS = int(os.environ.get("MONITORING_INTERVAL_SECONDS", "15"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url=INFERENCE_SERVER_URL,
    api_key=ROBOFLOW_API_KEY
)

# Initialize alert service (will be created on first fall detection)
alert_service = None
_alert_service_lock = asyncio.Lock()
_alert_service_disabled = False

# New: after hours alert service globals (lazy init)
after_hours_alert_service = None
_after_hours_alert_service_lock = asyncio.Lock()
_after_hours_alert_service_disabled = False

def get_camera_names() -> List[str]:
    """Get list of available camera names from the RTSP_STREAMS config."""
    return list(RTSP_STREAMS.keys())

async def get_frame_from_capture_service(camera_name: str) -> Optional[np.ndarray]:
    """Fetches a frame from the remote capture service."""
    url = f"{CAPTURE_SERVICE_URL}/frame/{camera_name}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            
            # Convert JPEG bytes to numpy array
            image_bytes = await response.aread()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
            
    except httpx.RequestError as e:
        LOGGER.error(f"Could not connect to capture service for {camera_name}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        LOGGER.error(f"Capture service returned an error for {camera_name}: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred while fetching frame for {camera_name}: {e}")
        return None

def numpy_to_viam_image(image: np.ndarray) -> ViamImage:
    """Converts a NumPy array (OpenCV BGR) to a ViamImage with JPEG data."""
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        raise RuntimeError("Failed to encode numpy array to JPEG")
    return ViamImage(data=buffer.tobytes(), mime_type="image/jpeg")

def viam_image_to_numpy(viam_img: ViamImage) -> np.ndarray:
    """Convert ViamImage to numpy array for Roboflow inference."""
    img_bytes = viam_img.data
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

async def run_roboflow_inference(image: np.ndarray) -> Dict:
    """
    Run inference using the Roboflow HTTP client.
    Returns a dict with at least a 'predictions' key.
    """
    try:
        # The client is already initialized globally
        result = await asyncio.to_thread(
            client.infer, 
            image, 
            model_id=ROBOFLOW_MODEL_ID
        )
        LOGGER.info(f"Roboflow inference via HTTP client complete. Found {len(result.get('predictions', []))} predictions")
        return result
    except Exception as e:
        LOGGER.error(f"Error during Roboflow inference (HTTP): {e}")
        return {"predictions": []}

def process_roboflow_results(results: Dict, img_width: int, img_height: int, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Process Roboflow detection results with boundary exclusion.
    
    Args:
        results: Raw results from Roboflow inference
        img_width: Width of the source image
        img_height: Height of the source image
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        List of processed detections with fall classification
    """
    detections = []
    
    # Exclusion zone configuration (10%)
    MARGIN_PCT = 0.10
    left_boundary = img_width * MARGIN_PCT
    right_boundary = img_width * (1.0 - MARGIN_PCT)
    bottom_boundary = img_height * (1.0 - MARGIN_PCT)

    for pred in results.get("predictions", []):
        confidence = pred.get("confidence", 0.0)
        
        if confidence < confidence_threshold:
            continue

        # Get center coordinates
        x = pred.get("x", 0)
        y = pred.get("y", 0)

        # Check Exclusion Zones
        # If center of box is too far left, too far right, or too low
        if x < left_boundary:
            LOGGER.info(f"Skipping detection (Left Boundary): x={x:.1f} < {left_boundary:.1f}")
            continue
        if x > right_boundary:
            LOGGER.info(f"Skipping detection (Right Boundary): x={x:.1f} > {right_boundary:.1f}")
            continue
        if y > bottom_boundary:
            LOGGER.info(f"Skipping detection (Bottom Boundary): y={y:.1f} > {bottom_boundary:.1f}")
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

async def get_or_create_alert_service():
    """Lazy, atomic initialize alert service on first fall detection.
    Uses a lock to prevent concurrent initializations and a disabled flag
    to avoid repeated retries when required config is missing.
    """
    global alert_service, _alert_service_disabled

    if _alert_service_disabled:
        return None

    if alert_service is not None:
        return alert_service

    async with _alert_service_lock:
        # double-check after acquiring lock
        if _alert_service_disabled:
            return None
        if alert_service is not None:
            return alert_service

        try:
            LOGGER.info("🔔 Initializing fall detection alert service (first fall detected)...")

            # Basic quick validation to avoid repeated failed inits caused by missing config
            required = ("twilio_account_sid", "twilio_auth_token", "twilio_from_phone")
            missing = [k for k in required if not ALERT_CONFIG.get(k)]
            if missing:
                _alert_service_disabled = True
                LOGGER.error("❌ Missing required Twilio configuration: %s", ", ".join(missing))
                return None

            alert_service = FallDetectionAlerts(ALERT_CONFIG)
            LOGGER.info("✅ Fall detection alert service initialized")
            return alert_service

        except Exception as e:
            # If initialization fails for a recoverable reason, log once and disable further tries for now.
            LOGGER.error("❌ Failed to initialize alert service: %s", e)
            _alert_service_disabled = True
            return None

# New: lazy loader for the after hours alert service using the "fall_detection_alerts copy.py" file
async def get_or_create_after_hours_service():
    """Lazy initialize a AfterHoursDetectionAlerts service from the 'fall_detection_alerts copy.py' file."""
    global after_hours_alert_service, _after_hours_alert_service_disabled

    if _after_hours_alert_service_disabled:
        return None

    if after_hours_alert_service is not None:
        return after_hours_alert_service

    async with _after_hours_alert_service_lock:
        if _after_hours_alert_service_disabled:
            return None
        if after_hours_alert_service is not None:
            return after_hours_alert_service

        try:
            LOGGER.info("🔔 Initializing after hours alert service (first after hours detection detected)...")

            required = ("twilio_account_sid", "twilio_auth_token", "twilio_from_phone")
            missing = [k for k in required if not ALERT_CONFIG.get(k)]
            if missing:
                _after_hours_alert_service_disabled = True
                LOGGER.error("❌ Missing required Twilio configuration for after hours alerts: %s", ", ".join(missing))
                return None

            # Dynamically load the copy file (filename contains a space) so we don't rely on import name
            module_path = os.path.join(os.path.dirname(__file__), "after_hours_detection_alerts.py")
            if not os.path.exists(module_path):
                LOGGER.error("❌ After hours alerts module not found at %s", module_path)
                _after_hours_alert_service_disabled = True
                return None

            spec = importlib.util.spec_from_file_location("after_hours_detection_alerts", module_path)
            if spec is None or spec.loader is None:
                LOGGER.error("❌ Could not create module spec or loader for %s", module_path)
                _after_hours_alert_service_disabled = True
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            AfterHoursDetectionAlerts = getattr(mod, "AfterHoursDetectionAlerts", None)
            if AfterHoursDetectionAlerts is None:
                LOGGER.error("❌ AfterHoursDetectionAlerts class not found in %s", module_path)
                _after_hours_alert_service_disabled = True
                return None

            after_hours_alert_service = AfterHoursDetectionAlerts(ALERT_CONFIG)
            LOGGER.info("✅ After hours alert service initialized")
            return after_hours_alert_service

        except Exception as e:
            LOGGER.error("❌ Failed to initialize after hours alert service: %s", e)
            _after_hours_alert_service_disabled = True
            return None

async def process_camera(camera_name: str):
    """Process a single camera feed by fetching it from the capture service."""
    try:
        LOGGER.info(f"Processing camera: {camera_name}")
        
        # Capture image from the remote service
        image = await get_frame_from_capture_service(camera_name)
        if image is None:
            LOGGER.error(f"Could not get image from camera: {camera_name}")
            return []
        
        LOGGER.info(f"Captured image from camera: {camera_name} | Shape: {image.shape}")
        
        # Get dimensions for boundary checking
        height, width = image.shape[:2]

        # Convert to ViamImage for alert functions
        viam_img = numpy_to_viam_image(image)

        # Run Roboflow inference
        results = await run_roboflow_inference(image)
        
        # Process results (Pass width and height now)
        detections = process_roboflow_results(results, width, height, confidence_threshold=0.6)
        
        # Log results
        LOGGER.info(f"Processed {len(detections)} detections from {camera_name}")
        for det in detections:
            LOGGER.info(f"  - {det['class']}: {det['confidence']:.2%} (Fall: {det['is_fall']})")
        
        # Send alerts for falls (initialize service on first fall)
        for det in detections:
            if det['is_fall']:
                LOGGER.info(f"🚨 Fall detected on camera {camera_name}")
                
                # Lazy initialize alert service
                service = await get_or_create_alert_service()
                
                if service:
                    LOGGER.info(f"📤 Sending fall alert for camera {camera_name}")
                    
                    # --- DEBUGGING BLOCK START ---
                    # Access internal state of the service to see what it actually has
                    LOGGER.info(f"DEBUG: Alert Service State:")
                    LOGGER.info(f"  - Account SID: {'***' if service.account_sid else 'MISSING'}")
                    LOGGER.info(f"  - From Phone: {service.from_phone}")
                    LOGGER.info(f"  - To Phones (Active): {service.to_phones}")
                    # --- DEBUGGING BLOCK END ---

                    try:
                        await service.send_fall_alert(
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
                        LOGGER.info(f"✅ Fall alert sent successfully")
                    except Exception as e:
                        LOGGER.error(f"❌ Failed to send fall alert: {e}")
                else:
                    LOGGER.error("⚠️ Alert service failed to initialize - alert not sent")

        # NEW: Send after hours alerts for "person" detections coming from whitelisted cameras
        #      ONLY between 23:00 and 06:00.
        current_hour = datetime.now().hour
        is_after_hours = current_hour >= 23 or current_hour < 6

        if is_after_hours:
            for det in detections:
                try:
                    if det.get("class", "").lower() == "person" and camera_name in AFTER_HOURS_CAMERAS:
                        LOGGER.info(f"🔔 After hours (person) detected on whitelisted camera {camera_name}")
                        pservice = await get_or_create_after_hours_service()
                        if pservice:
                            try:
                                await pservice.send_after_hours_alert(
                                    camera_name=camera_name,
                                    alert_type="after_hours",
                                    person_id=det.get("detection_id", ""),
                                    confidence=det.get("confidence", 0.0),
                                    image=viam_img,
                                    metadata={
                                        'bbox': det['bbox'],
                                        'class': det['class'],
                                        'inference_id': results.get('inference_id'),
                                        'model_id': ROBOFLOW_MODEL_ID
                                    }
                                )
                                LOGGER.info("✅ After hours detection alert sent successfully")
                            except Exception as e:
                                LOGGER.error(f"❌ Failed to send after hours alert: {e}")
                        else:
                            LOGGER.error("⚠️ After hours alert service failed to initialize - alert not sent")
                except Exception as e:
                    LOGGER.error(f"Error while handling after hours alert logic: {e}")
        else:
            # Optional: Log that a person was detected but it's not after hours
            if any(d.get("class", "").lower() == "person" and camera_name in AFTER_HOURS_CAMERAS for d in detections):
                LOGGER.info(f"Person detected on {camera_name}, but it is not after hours. Skipping alert.")

        return detections
        
    except Exception as e:
        LOGGER.error(f"Error processing camera {camera_name}: {e}")
        return []

async def main():
    """Continuous monitoring main loop: initialize once, then run cycles."""
    iteration = 0
    try:
        # Get camera names from config
        camera_names = get_camera_names()
        LOGGER.info(f"Available cameras: {camera_names}")

        if not camera_names:
            LOGGER.error("No cameras found in RTSP_STREAMS configuration!")
            return

        LOGGER.info(f"Starting continuous monitoring (interval={MONITORING_INTERVAL_SECONDS}s)")

        while True:
            iteration += 1
            LOGGER.info("\n" + "="*60)
            LOGGER.info(f"MONITORING CYCLE #{iteration}")
            LOGGER.info("="*60)

            all_detections = {}
            for camera_name in camera_names:
                # The stream_url is no longer needed here
                detections = await process_camera(camera_name)
                all_detections[camera_name] = detections

            # Summary for this cycle
            LOGGER.info("\n" + "="*50)
            LOGGER.info(f"CYCLE #{iteration} SUMMARY")
            LOGGER.info("="*50)
            for camera_name, detections in all_detections.items():
                fall_count = sum(1 for d in detections if d.get("is_fall"))
                LOGGER.info(f"{camera_name}: {len(detections)} detections, {fall_count} falls")

            LOGGER.info(f"Waiting {MONITORING_INTERVAL_SECONDS}s until next cycle...")
            await asyncio.sleep(MONITORING_INTERVAL_SECONDS)

    except asyncio.CancelledError:
        LOGGER.info("Shutdown requested")
    except Exception as e:
        LOGGER.error(f"Fatal error in main: {e}", exc_info=True)
        raise
    finally:
        # No longer need to release cv2 captures here
        LOGGER.info("Monitoring stopped.")

if __name__ == "__main__":
    asyncio.run(main())
