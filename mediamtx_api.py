"""
Flask API for MediaMTX Camera Streaming
Simplified API that provides camera info and WebRTC endpoints
MediaMTX handles all stream lifecycle management
Deploy this to /opt/mediamtx_api.py
"""

from flask import Flask, jsonify

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
                    "location": "CPW Awning",
                    "description": "North facing camera at CPW awning",
                    "urls": {
                        "whep": "https://stream.rigguardian.com:8889/CPW_Awning_N_Facing/whep",
                        "hls": "https://stream.rigguardian.com:8888/CPW_Awning_N_Facing/index.m3u8",
                        "rtsp": "rtsp://stream.rigguardian.com:8554/CPW_Awning_N_Facing"
                    }
                },
                ...
            ]
        }
    """
    cameras_list = []

    for camera_id, camera_info in CAMERAS.items():
        cameras_list.append({
            'id': camera_id,
            'name': camera_info['name'],
            'urls': get_camera_urls(camera_id)
        })

    return jsonify({
        'status': 'success',
        'cameras': cameras_list,
        'count': len(cameras_list)
    }), 200

# --- Startup ---

if __name__ == '__main__':
    print(f"Starting MediaMTX API server")
    print(f"MediaMTX domain: {MEDIAMTX_DOMAIN}")
    print(f"Number of cameras: {len(CAMERAS)}")
    print(f"API running on http://127.0.0.1:5001 (localhost only - Nginx proxies public traffic)")

    # ✅ Bind to localhost only - Nginx handles all public traffic
    app.run(host='127.0.0.1', port=5001, debug=False)
