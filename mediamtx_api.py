"""
Flask API for MediaMTX Camera Streaming
Simplified API that provides camera info and WebRTC endpoints
MediaMTX handles all stream lifecycle management
Deploy this to /opt/mediamtx_api.py
"""

from flask import Flask, jsonify, request
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Configuration ---

# MediaMTX server configuration
MEDIAMTX_DOMAIN = 'stream.rigguardian.com'
MEDIAMTX_WEBRTC_PORT = 8889  # WebRTC port from mediamtx.yml
MEDIAMTX_HLS_PORT = 8888     # HLS port (fallback) from mediamtx.yml
MEDIAMTX_API_PORT = 9997     # MediaMTX API port

# Camera registry - matches path names in mediamtx.yml
CAMERAS = {
    'CPW_Awning_N_Facing': {
        'name': 'CPW Awning N Facing'
    },
    'CPW_South_Cam': {
        'name': 'CPW South Cam'
    },
    '66th_St_East_Facing': {
        'name': '66th St East Facing'
    },
    'Service_Staircase_Top': {
        'name': 'Service Staircase Top'
    },
    'Service_Staircase_Bottom': {
        'name': 'Service Staircase Bottom'
    },
    'Main_Entrance': {
        'name': 'Main Entrance'
    },
    'Concierge': {
        'name': 'Concierge'
    },
    'Package_Room': {
        'name': 'Package Room'
    },
    'Bike_Room': {
        'name': 'Bike Room'
    },
    'Lobby_Center_North': {
        'name': 'Lobby Center North'
    },
    'Lobby_Center_South': {
        'name': 'Lobby Center South'
    },
    'Lobby_ABCD_Side': {
        'name': 'Lobby ABCD Side'
    },
    'Lobby_EFG_Side': {
        'name': 'Lobby EFG Side'
    },
    'North_Elevator': {
        'name': 'North Elevator'
    },
    'South_Elevator': {
        'name': 'South Elevator'
    },
    'Services_Area': {
        'name': 'Services Area'
    },
    'Basement_Hallway': {
        'name': 'Basement Hallway'
    },
    'ABCD_Basement_Hallway': {
        'name': 'ABCD Basement Hallway'
    },
    'Basement_EFG_Side': {
        'name': 'Basement EFG Side'
    },
    'EFG_Basement_Hallway': {
        'name': 'EFG Basement Hallway'
    },
    'Roof_Front_East_Facing': {
        'name': 'Roof Front East Facing'
    },
    'Roof_Front_North_Facing': {
        'name': 'Roof Front North Facing'
    },
    'Roof_South_Side': {
        'name': 'Roof South Side'
    },
    'Roof_Center_East_Facing': {
        'name': 'Roof Center East Facing'
    },
    'Courtyard_1': {
        'name': 'Courtyard 1'
    },
    'Courtyard_2': {
        'name': 'Courtyard 2'
    },
}

# --- Helper Functions ---

def get_camera_urls(camera_id: str) -> dict:
    """Generate streaming URLs for a camera.

    Nginx proxies all requests on port 443, so we don't include port numbers
    in the HTTPS URLs.
    """
    return {
        'whep': f'https://{MEDIAMTX_DOMAIN}/{camera_id}/whep',
        'hls': f'https://{MEDIAMTX_DOMAIN}/{camera_id}/index.m3u8',
        'rtsp': f'rtsp://{MEDIAMTX_DOMAIN}:8554/{camera_id}'
    }

# --- Middleware for request logging ---

@app.before_request
def log_request_info():
    """Log detailed information about incoming requests."""
    logger.info(f"===== INCOMING REQUEST =====")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info(f"Method: {request.method}")
    logger.info(f"Path: {request.path}")
    logger.info(f"Remote IP: {request.remote_addr}")
    logger.info(f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}")
    logger.info(f"Referrer: {request.headers.get('Referer', 'None')}")
    if request.args:
        logger.info(f"Query Params: {dict(request.args)}")

@app.after_request
def log_response_info(response):
    """Log response information."""
    logger.info(f"Response Status: {response.status_code}")
    logger.info(f"===== REQUEST COMPLETE =====")
    return response

# --- API Endpoints ---

@app.route('/api/cameras', methods=['GET'])
def list_cameras():
    """
    List all available cameras with their streaming URLs.

    Returns:
        {
            "cameras": [
                {
                    "id": "CPW_Awning_N_Facing",
                    "name": "CPW Awning N Facing",
                    "urls": {
                        "whep": "https://stream.rigguardian.com/CPW_Awning_N_Facing/whep",
                        "hls": "https://stream.rigguardian.com/CPW_Awning_N_Facing/index.m3u8",
                        "rtsp": "rtsp://stream.rigguardian.com:8554/CPW_Awning_N_Facing"
                    }
                },
                ...
            ]
        }
    """
    start_time = time.time()
    logger.info(f"Building camera list for {len(CAMERAS)} cameras")
    
    cameras_list = []

    for camera_id, camera_info in CAMERAS.items():
        camera_data = {
            'id': camera_id,
            'name': camera_info['name'],
            'urls': get_camera_urls(camera_id)
        }
        cameras_list.append(camera_data)
        logger.debug(f"Added camera: {camera_id} -> WHEP: {camera_data['urls']['whep']}")

    duration = (time.time() - start_time) * 1000
    logger.info(f"Camera list built in {duration:.1f}ms")

    return jsonify({
        'status': 'success',
        'cameras': cameras_list,
        'count': len(cameras_list)
    }), 200

@app.route('/api/cameras/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Get info for a specific camera with detailed logging."""
    logger.info(f"Fetching info for camera: {camera_id}")
    
    if camera_id not in CAMERAS:
        logger.warning(f"Camera not found: {camera_id}")
        logger.info(f"Available cameras: {list(CAMERAS.keys())}")
        return jsonify({
            'status': 'error',
            'message': f'Camera {camera_id} not found',
            'available_cameras': list(CAMERAS.keys())
        }), 404
    
    camera_info = CAMERAS[camera_id]
    urls = get_camera_urls(camera_id)
    
    logger.info(f"✓ Camera found: {camera_info['name']}")
    logger.info(f"  WHEP URL: {urls['whep']}")
    logger.info(f"  HLS URL: {urls['hls']}")
    logger.info(f"  RTSP URL: {urls['rtsp']}")
    
    return jsonify({
        'status': 'success',
        'camera': {
            'id': camera_id,
            'name': camera_info['name'],
            'urls': urls
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system info."""
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cameras_registered': len(CAMERAS),
        'mediamtx_domain': MEDIAMTX_DOMAIN
    }), 200

# --- Startup ---

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info(f"Starting MediaMTX API server")
    logger.info(f"MediaMTX domain: {MEDIAMTX_DOMAIN}")
    logger.info(f"Number of cameras: {len(CAMERAS)}")
    logger.info(f"API running on http://127.0.0.1:5001 (localhost only - Nginx proxies public traffic)")
    logger.info("=" * 60)

    # ✅ Bind to localhost only - Nginx handles all public traffic
    app.run(host='127.0.0.1', port=5001, debug=False)
