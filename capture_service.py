import asyncio
import logging
from typing import Dict, Optional, AsyncGenerator
import atexit
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# --- Configuration (Should match your main script) ---
RTSP_STREAMS = {
    'Lobby_Center_North': 'rtsp://192.168.1.200:554/chID=25&streamType=sub',
    'CPW_Awning_N_Facing': 'rtsp://192.168.1.200:554/chID=16&streamType=sub',
    # Add other stream names and URLs here
}

GSTREAMER_PIPELINE = (
    # Add protocols=tcp to rtspsrc to force a more reliable connection
    "rtspsrc location={rtsp_url} latency=0 ! "
    "rtph265depay ! h265parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1"
)

# --- OpenCV Capture Logic ---
_caps: Dict[str, cv2.VideoCapture] = {}

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
    cap = _caps.get(stream_name)

    if cap is None or not cap.isOpened():
        LOGGER.warning(f"[Capture] No existing capture for {stream_name} or it's closed. Opening new one.")
        pipeline = GSTREAMER_PIPELINE.format(rtsp_url=stream_url)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            LOGGER.error(f"[Capture] FATAL: GStreamer pipeline failed to open for {stream_name}.")
            if stream_name in _caps:
                del _caps[stream_name]
            return None
        _caps[stream_name] = cap

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
    return RTSP_STREAMS

async def stream_generator(camera_name: str) -> AsyncGenerator[bytes, None]:
    """
    A generator function that captures frames from the RTSP stream,
    encodes them as JPEG, and yields them for streaming.
    """
    while True:
        # Run the synchronous capture function in a thread to avoid blocking
        frame = await asyncio.to_thread(capture_rtsp_frame_sync, camera_name)
        
        if frame is None:
            LOGGER.warning(f"Skipping frame for {camera_name} due to capture error.")
            # Wait a moment before retrying to avoid a tight loop on failure
            await asyncio.sleep(0.5)
            continue

        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            LOGGER.warning(f"Skipping frame for {camera_name} due to encoding error.")
            continue

        # Yield the frame in the multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Control the frame rate (e.g., ~30fps). Adjust as needed.
        await asyncio.sleep(0.03)


@app.get("/stream/{camera_name}")
async def stream_video(camera_name: str):
    """
    Endpoint to stream video from a specific camera as an MJPEG stream.
    View this in a browser via an <img> tag.
    """
    if camera_name not in RTSP_STREAMS:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return StreamingResponse(stream_generator(camera_name),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/frame/{camera_name}",
         responses={
             200: {"content": {"image/jpeg": {}}},
             404: {"description": "Camera not found or frame could not be captured"}
         })
async def get_frame(camera_name: str):
    """Endpoint to capture a frame from a specific camera."""
    LOGGER.info(f"Request received for camera: {camera_name}")
    
    # Run the synchronous capture function in a thread to avoid blocking
    frame = await asyncio.to_thread(capture_rtsp_frame_sync, camera_name)
    
    if frame is None:
        raise HTTPException(status_code=404, detail=f"Could not capture frame from {camera_name}")

    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode frame to JPEG")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    # Run on all available network interfaces on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)