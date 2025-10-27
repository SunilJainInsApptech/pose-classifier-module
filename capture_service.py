import asyncio
import logging
import uuid
from typing import Dict, Optional, AsyncGenerator, Set
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# New imports for WebRTC
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaStreamTrack

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# --- Configuration (Should match your main script) ---
RTSP_STREAMS = {
    'Lobby_Center_North': 'rtsp://70.19.68.121:554/chID=25&streamType=sub',
    'CPW_Awning_N_Facing': 'rtsp://70.19.68.121:554/chID=16&streamType=sub',
    # Add other stream names and URLs here
}

# OPTIMIZED GStreamer pipeline using Jetson's hardware decoder
GSTREAMER_PIPELINE = (
    "rtspsrc location={rtsp_url} latency=0 ! "
    "rtph265depay ! h265parse ! "
    "nvv4l2decoder ! "  # Use the NVIDIA hardware decoder
    "nvvidconv ! "      # Use NVIDIA hardware converter
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "   # Convert to BGR for OpenCV
    "video/x-raw, format=BGR ! "
    "appsink drop=1"
)

# --- OpenCV Capture Logic ---
_caps: Dict[str, cv2.VideoCapture] = {}

def capture_rtsp_frame_sync(stream_name: str) -> Optional[np.ndarray]:
    """
    Synchronously captures a frame from an RTSP stream using a persistent
    cv2.VideoCapture object.
    """
    if stream_name not in RTSP_STREAMS:
        LOGGER.error(f"Unknown stream name: {stream_name}")
        return None

    stream_url = RTSP_STREAMS[stream_name]
    cap = _caps.get(stream_name)

    if cap is None or not cap.isOpened():
        LOGGER.info(f"Opening new stream for {stream_name}...")
        pipeline = GSTREAMER_PIPELINE.format(rtsp_url=stream_url)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            LOGGER.error(f"Failed to open stream: {stream_name}")
            if stream_name in _caps:
                del _caps[stream_name]
            return None
        _caps[stream_name] = cap

    ret, frame = cap.read()
    if not ret:
        LOGGER.warning(f"Failed to read frame from {stream_name}. Releasing capture object.")
        cap.release()
        if stream_name in _caps:
            del _caps[stream_name]
        return None
    
    return frame

# --- WebRTC Implementation ---

# This class bridges OpenCV frames to the WebRTC video track
class CameraVideoStreamTrack(MediaStreamTrack):
    """
    A video stream track that gets frames from a camera.
    """
    kind = "video"

    def __init__(self, camera_name: str):
        super().__init__()
        self.camera_name = camera_name

    async def recv(self):
        # This is called by aiortc to get the next frame
        frame = await asyncio.to_thread(capture_rtsp_frame_sync, self.camera_name)
        
        if frame is None:
            # If frame capture fails, we must still provide a frame.
            # A black frame is a good fallback.
            frame = np.zeros((480, 640, 3), np.uint8)

        # Convert the frame to a WebRTC VideoFrame
        from av import VideoFrame
        pts, time_base = await self.next_timestamp()
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

# Keep track of active peer connections
pcs: Set[RTCPeerConnection] = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    # Startup logic can go here if needed
    yield
    # --- Updated Shutdown Logic ---
    LOGGER.info("Shutdown event received. Releasing all stream captures and closing peer connections...")
    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    # Shutdown logic
    LOGGER.info("Releasing all stream captures...")
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


# --- New WebRTC Signaling Endpoint ---

# Define STUN and TURN servers for NAT traversal.
ice_servers = [
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
    # Adding a free, public TURN server for testing as a fallback.
    # In a production environment, you should use your own or a paid TURN service.
    RTCIceServer(
        urls="turn:openrelay.metered.ca:80",
        username="openrelayproject",
        credential="openrelayproject",
    ),
    RTCIceServer(
        urls="turn:openrelay.metered.ca:443",
        username="openrelayproject",
        credential="openrelayproject",
    ),
]
config = RTCConfiguration(iceServers=ice_servers)

@app.post("/offer")
async def offer(params: dict = Body(...)):
    """
    The browser sends an SDP offer to start a WebRTC connection.
    The server responds with an SDP answer.
    """
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    camera_name = params.get("cameraName")

    if not camera_name or camera_name not in RTSP_STREAMS:
        raise HTTPException(status_code=400, detail="Invalid camera name provided")

    # Create PeerConnection with the STUN server configuration
    pc = RTCPeerConnection(config)
    pc_id = f"PeerConnection({uuid.uuid4()})"
    pcs.add(pc)

    LOGGER.info(f"[{pc_id}] Creating peer connection for camera: {camera_name}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        LOGGER.info(f"[{pc_id}] Connection state is {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
            LOGGER.info(f"[{pc_id}] Peer connection closed and removed.")

    # Create and add the video track
    video_track = CameraVideoStreamTrack(camera_name)
    pc.addTrack(video_track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


if __name__ == "__main__":
    import uvicorn
    # Run on all available network interfaces on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)