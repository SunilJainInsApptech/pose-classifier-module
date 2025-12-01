import asyncio
import logging
from typing import Dict, Optional, AsyncGenerator
import atexit
from contextlib import asynccontextmanager
import os
from datetime import datetime
import threading  # <--- ADDED THIS

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# --- Configuration (Should match your main script) ---
# Use the literal '&' character - GStreamer/OpenCV handle it correctly when it's in the URL string
RTSP_STREAMS = {
    'Lobby_Center_North': 'rtsp://192.168.1.200:554/chID=25&streamType=sub',
    'CPW_Awning_N_Facing': 'rtsp://192.168.1.200:554/chID=16&streamType=sub',
    'Roof_Front_East_Facing': 'rtsp://192.168.1.200:554/chID=01&streamType=sub',
    # Add other stream names and URLs here
}

CAMERAS_AVAILABLE_TO_STREAM = {
    'CPW_Awning_N_Facing': 'rtsp://192.168.1.200:554/chID=16&streamType=sub',
	'CPW_South_Cam': 'rtsp://192.168.1.200:554/chID=7&streamType=sub',
	'66th_St_East_Facing': 'rtsp://192.168.1.200:554/chID=28&streamType=sub',
	'Service_Staircase_Top': 'rtsp://192.168.1.200:554/chID=15&streamType=sub',
	'Service_Staircase_Bottom': 'rtsp://192.168.1.200:554/chID=26&streamType=sub',
	'Main_Entrance': 'rtsp://192.168.1.200:554/chID=23&streamType=sub',
	'Concierge': 'rtsp://192.168.1.200:554/chID=13&streamType=sub',
	'Package_Room': 'rtsp://192.168.1.200:554/chID=8&streamType=sub',
	'Bike_Room': 'rtsp://192.168.1.200:554/chID=11&streamType=sub',
	'Lobby_Center_North': 'rtsp://192.168.1.200:554/chID=25&streamType=sub',
	'Lobby_Center_South': 'rtsp://192.168.1.200:554/chID=21&streamType=sub',
	'Lobby_ABCD_Side': 'rtsp://192.168.1.200:554/chID=19&streamType=sub',
	'Lobby_EFG_Side': 'rtsp://192.168.1.200:554/chID=18&streamType=sub',
	'North_Elevator': 'rtsp://192.168.1.200:554/chID=35&streamType=sub',
	'South_Elevator': 'rtsp://192.168.1.200:554/chID=36&streamType=sub',
	'Services_Area': 'rtsp://192.168.1.200:554/chID=9&streamType=sub',
	'Basement_Hallway': 'rtsp://192.168.1.200:554/chID=31&streamType=sub',
	'ABCD_Basement_Hallway': 'rtsp://192.168.1.200:554/chID=32&streamType=sub',
	'Basement_EFG_Side': 'rtsp://192.168.1.200:554/chID=24&streamType=sub',
	'EFG_Basement_Hallway': 'rtsp://192.168.1.200:554/chID=27&streamType=sub',
	'Roof_Front_East_Facing': 'rtsp://192.168.1.200:554/chID=1&streamType=sub',
	'Roof_Front_North_Facing': 'rtsp://192.168.1.200:554/chID=2&streamType=sub',
	'Roof_South_Side': 'rtsp://192.168.1.200:554/chID=4&streamType=sub',
	'Roof_Center_East_Facing': 'rtsp://192.168.1.200:554/chID=12&streamType=sub',
	'Courtyard_1': 'rtsp://192.168.1.200:554/chID=22&streamType=sub',
	'Courtyard_2': 'rtsp://192.168.1.200:554/chID=30&streamType=sub',
}

# --- UPDATED GSTREAMER PIPELINE LOGIC ---
def _gst_candidate_pipelines(rtsp_url: str):
    """
    Returns a list of pipelines to try.
    1. Specific Hardware H.265 (Matches your working gst-launch command)
    2. Generic Decodebin (Fallback for H.264 or other formats)
    """
    # Matches: gst-launch-1.0 rtspsrc ... ! rtpjitterbuffer ! ... H265 ... ! nvv4l2decoder ...
    # Note: Removed drop-on-latency=true to match your working command.
    hw_h265 = (
        f'rtspsrc location="{rtsp_url}" latency=500 protocols=tcp name=src '
        'src. ! rtpjitterbuffer ! application/x-rtp,media=video,encoding-name=H265 ! '
        'rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! '
        'video/x-raw,format=BGRx,width=704,height=480 ! videoconvert ! '
        'video/x-raw,format=BGR ! appsink drop=1 sync=false'
    )
    
    # Fallback: Generic decodebin (useful if a camera is H.264)
    generic = (
        f'rtspsrc location="{rtsp_url}" latency=500 protocols=tcp drop-on-latency=true ! '
        'decodebin ! nvvidconv ! '
        'video/x-raw,format=BGRx,width=704,height=480 ! videoconvert ! '
        'video/x-raw,format=BGR ! appsink drop=1 sync=false'
    )
    
    return [hw_h265, generic]

# --- OpenCV Capture Logic ---
_caps: Dict[str, cv2.VideoCapture] = {}
_init_lock = threading.Lock()  # <--- ADDED GLOBAL LOCK

# --- Add this shutdown function ---
def _shutdown_captures():
    """
    This function is registered to run at script exit.
    It ensures all OpenCV VideoCapture objects are properly released.
    """
    LOGGER.info("Shutdown signal received. Releasing all video captures...")
    for cam_name, cap_obj in _caps.items():
        if cap_obj and cap_obj.isOpened():
            LOGGER.info(f"Releasing capture for {cam_name}")
            cap_obj.release()
    LOGGER.info("All captures released.")

# Register the shutdown function to be called on exit
atexit.register(_shutdown_captures)

def capture_rtsp_frame_sync(stream_name: str) -> Optional[np.ndarray]:
    """
    Synchronously captures a frame from an RTSP stream using a persistent
    cv2.VideoCapture object.
    """
    # --- MORE LOGGING ---
    LOGGER.info(f"[Capture] Attempting to get frame for {stream_name}")
    if stream_name not in RTSP_STREAMS:
        LOGGER.error(f"[Capture] Unknown stream name: {stream_name}")
        return None

    stream_url = RTSP_STREAMS[stream_name]
    
    # Use a lock during initialization to prevent crashing the NVDEC driver
    # by trying to open multiple hardware decoders at the exact same millisecond.
    with _init_lock:
        cap = _caps.get(stream_name)

        if cap is None or not cap.isOpened():
            LOGGER.warning(f"[Capture] No existing capture for {stream_name} or it's closed. Opening new one.")
            
            # Iterate through candidate pipelines
            for pipeline in _gst_candidate_pipelines(stream_url):
                LOGGER.info(f"[Capture] Trying pipeline: {pipeline[:80]}...")
                try:
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    if cap.isOpened():
                        LOGGER.info(f"[Capture] Pipeline opened successfully.")
                        break
                except Exception as e:
                    LOGGER.warning(f"[Capture] Pipeline failed: {e}")
            
            if cap is None or not cap.isOpened():
                LOGGER.error(f"[Capture] FATAL: All GStreamer pipelines failed for {stream_name}.")
                if stream_name in _caps:
                    del _caps[stream_name]
                return None
            
            # WARMUP LOOP with timeout protection
            # Some cameras take 5-10 seconds to send the first keyframe.
            # We try to read frames but don't block forever.
            LOGGER.info(f"[Capture] Warming up pipeline for {stream_name}...")
            import time
            warmup_start = time.time()
            warmup_timeout = 15  # seconds
            warmup_frames = 0
            
            while time.time() - warmup_start < warmup_timeout and warmup_frames < 5:
                try:
                    ret, _ = cap.read()
                    if ret:
                        warmup_frames += 1
                        LOGGER.info(f"[Capture] Warmup frame {warmup_frames}/5 received for {stream_name}")
                    else:
                        # Frame not ready yet, wait a bit
                        time.sleep(0.2)
                except Exception as e:
                    LOGGER.warning(f"[Capture] Warmup error for {stream_name}: {e}")
                    break
            
            if warmup_frames == 0:
                LOGGER.error(f"[Capture] Warmup failed for {stream_name} - no frames received after {warmup_timeout}s")
                cap.release()
                if stream_name in _caps:
                    del _caps[stream_name]
                return None
                
            _caps[stream_name] = cap
            LOGGER.info(f"[Capture] Pipeline initialized for {stream_name} ({warmup_frames} warmup frames)")

    # Now read the actual frame
    # We use the cap object we just retrieved or created
    cap = _caps.get(stream_name)
    if cap is None:
        return None

    ret, frame = cap.read()
    if not ret:
        LOGGER.warning(f"[Capture] cap.read() failed for {stream_name}. Releasing capture object.")
        cap.release()
        if stream_name in _caps:
            del _caps[stream_name]
        return None
    
    # --- MORE LOGGING ---
    LOGGER.info(f"[Capture] Successfully read frame for {stream_name} with shape {frame.shape}")
    return frame

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    # Startup logic can go here if needed
    yield
    # --- Shutdown Logic ---
    LOGGER.info("Shutdown event received. Releasing all stream captures...")
    global _caps
    for cam_name, cap in _caps.items():
        if cap and cap.isOpened():
            LOGGER.info(f"Releasing capture for {cam_name}")
            cap.release()
    cv2.destroyAllWindows()
    LOGGER.info("All stream captures released.")

app = FastAPI(lifespan=lifespan)

# --- CORS Middleware Configuration ---
# This allows your web app (on localhost or rigguardian.com) to make requests to this API
origins = [
    "http://localhost:3000",      # Local Next.js development environment
    "https://rigguardian.com",    # Production domain
    "http://rigguardian.com",     # Production domain (non-HTTPS)
    "http://www.rigguardian.com", # ADD THIS LINE for the 'www' subdomain
    "https://www.rigguardian.com", # ADD THIS LINE for the 'www' subdomain
    "http://building-sensor-platform.vercel.app",
    "https://building-sensor-platform.vercel.app",
    "https://uncusped-principled-leon.ngrok-free.dev", # Your ngrok URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

@app.get("/cameras")
async def get_cameras():
    """Returns a JSON object of available camera names and their RTSP URLs."""
    return CAMERAS_AVAILABLE_TO_STREAM

@app.get("/frame/{camera_name}",
         responses={
             200: {"content": {"image/jpeg": {}}},
             404: {"description": "Camera not found or frame could not be captured"}
         })
async def get_frame(camera_name: str):
    """Endpoint to capture a frame from a specific camera."""
    LOGGER.info(f"Request received for camera: {camera_name}")
    
    # Run the synchronous capture