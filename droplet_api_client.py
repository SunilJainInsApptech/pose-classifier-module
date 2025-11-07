# Flask API for Direct RTSP-to-HLS Streaming on Droplet
# No Jetson dependency - streams cameras directly from the Droplet

from flask import Flask, jsonify
import subprocess
import threading
import time
import os
import shutil
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuration ---

# Camera RTSP URLs
CAMERA_URLS = {
    'Lobby_Center_North': 'rtsp://70.19.68.121:554/chID=25&streamType=sub',
    'CPW_Awning_N_Facing': 'rtsp://70.19.68.121:554/chID=16&streamType=sub',
    'Roof_Front_East_Facing': 'rtsp://70.19.68.121:554/chID=01&streamType=sub',
}

# HLS Output Configuration
DROPLET_IP = '104.236.30.246'
DROPLET_HLS_PORT = 8000
HLS_OUTPUT_DIR = '/var/www/html/hls'  # Served by nginx on port 8000

# Stream Management
STREAM_IDLE_TIMEOUT = 60  # Stop stream after 60 seconds with no requests
STREAM_INIT_WAIT_TIME = 3  # Wait for initial segments

# Active FFmpeg processes {camera_id: subprocess.Popen}
active_streams = {}
last_request_time = {}
stream_locks = {}

# --- FFmpeg Management ---

def start_ffmpeg_stream(camera_id: str, rtsp_url: str):
    """Start FFmpeg process to convert RTSP to HLS."""
    
    if camera_id in active_streams:
        app.logger.info(f"Stream {camera_id} already running.")
        return True
    
    # Create output directory
    output_dir = os.path.join(HLS_OUTPUT_DIR, camera_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear old segments
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        try:
            os.remove(file_path)
        except Exception as e:
            app.logger.warning(f"Could not delete {file_path}: {e}")
    
    playlist_path = os.path.join(output_dir, 'playlist.m3u8')
    segment_pattern = os.path.join(output_dir, 'segment_%03d.ts')
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-c:v', 'copy',  # Copy video codec (no re-encoding)
        '-c:a', 'aac',   # Encode audio to AAC
        '-f', 'hls',
        '-hls_time', '2',  # 2-second segments
        '-hls_list_size', '5',  # Keep last 5 segments in playlist
        '-hls_flags', 'delete_segments',  # Auto-delete old segments
        '-hls_segment_filename', segment_pattern,
        playlist_path
    ]
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        active_streams[camera_id] = process
        stream_locks[camera_id] = threading.Lock()
        
        app.logger.info(f"Started FFmpeg for {camera_id} (PID: {process.pid})")
        return True
        
    except Exception as e:
        app.logger.error(f"Failed to start FFmpeg for {camera_id}: {e}")
        return False

def stop_ffmpeg_stream(camera_id: str):
    """Stop FFmpeg process for a camera."""
    
    if camera_id not in active_streams:
        return {'status': 'ok', 'message': f'Stream {camera_id} not running'}
    
    try:
        process = active_streams[camera_id]
        process.terminate()
        process.wait(timeout=5)
        
        del active_streams[camera_id]
        if camera_id in stream_locks:
            del stream_locks[camera_id]
        
        # Clean up HLS files
        output_dir = os.path.join(HLS_OUTPUT_DIR, camera_id)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        app.logger.info(f"Stopped FFmpeg for {camera_id}")
        return {'status': 'success', 'message': f'Stream {camera_id} stopped'}
        
    except Exception as e:
        app.logger.error(f"Error stopping {camera_id}: {e}")
        return {'status': 'error', 'message': str(e)}

# --- Idle Stream Checker ---

def idle_checker():
    """Background thread that stops idle streams."""
    while True:
        time.sleep(10)
        current_time = time.time()
        
        for camera_id, last_time in list(last_request_time.items()):
            if current_time - last_time > STREAM_IDLE_TIMEOUT:
                app.logger.info(f"Stream {camera_id} idle for {STREAM_IDLE_TIMEOUT}s. Stopping...")
                try:
                    stop_ffmpeg_stream(camera_id)
                    del last_request_time[camera_id]
                except Exception as e:
                    app.logger.error(f"Failed to stop idle stream {camera_id}: {e}")

# --- API Endpoints ---

@app.route('/stream/start', methods=['POST'])
def start_stream():
    """Start streaming a camera."""
    from flask import request
    
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if not camera_id:
        return jsonify({'status': 'error', 'message': 'Missing camera_id'}), 400
    
    if camera_id not in CAMERA_URLS:
        return jsonify({'status': 'error', 'message': f'Unknown camera: {camera_id}'}), 404
    
    rtsp_url = CAMERA_URLS[camera_id]
    last_request_time[camera_id] = time.time()
    
    # Start FFmpeg
    success = start_ffmpeg_stream(camera_id, rtsp_url)
    
    if success:
        time.sleep(STREAM_INIT_WAIT_TIME)  # Wait for initial segments
        
        hls_url = f"http://{DROPLET_IP}:{DROPLET_HLS_PORT}/{camera_id}/playlist.m3u8"
        
        return jsonify({
            'status': 'success',
            'message': 'Stream started',
            'hls_url': hls_url
        }), 200
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start FFmpeg'}), 500

@app.route('/start_stream/<camera_id>', methods=['GET'])
def start_stream_get(camera_id: str):
    """GET endpoint to start a stream."""
    
    if camera_id not in CAMERA_URLS:
        return jsonify({'status': 'error', 'message': f'Unknown camera: {camera_id}'}), 404
    
    rtsp_url = CAMERA_URLS[camera_id]
    last_request_time[camera_id] = time.time()
    
    success = start_ffmpeg_stream(camera_id, rtsp_url)
    
    if success:
        time.sleep(STREAM_INIT_WAIT_TIME)
        
        hls_url = f"http://{DROPLET_IP}:{DROPLET_HLS_PORT}/{camera_id}/playlist.m3u8"
        
        return jsonify({
            'status': 'success',
            'message': 'Stream started',
            'hls_url': hls_url,
            'camera_id': camera_id
        }), 200
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start FFmpeg'}), 500

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop streaming a camera."""
    from flask import request
    
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if not camera_id:
        return jsonify({'status': 'error', 'message': 'Missing camera_id'}), 400
    
    result = stop_ffmpeg_stream(camera_id)
    
    if camera_id in last_request_time:
        del last_request_time[camera_id]
    
    return jsonify(result)

@app.route('/stream/keepalive', methods=['POST'])
def keepalive_stream():
    """Update last access time for a camera."""
    from flask import request
    
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if camera_id:
        last_request_time[camera_id] = time.time()
    
    return jsonify({'status': 'ok'}), 200

@app.route('/stream/status', methods=['GET'])
def stream_status():
    """Get status of all active streams."""
    status = {
        'active_streams': list(active_streams.keys()),
        'last_access': {k: time.time() - v for k, v in last_request_time.items()}
    }
    return jsonify(status)

# --- Startup ---

if __name__ == '__main__':
    # Start idle checker thread
    checker_thread = threading.Thread(target=idle_checker, daemon=True)
    checker_thread.start()
    print("Idle stream checker started")
    
    # Create HLS output directory
    os.makedirs(HLS_OUTPUT_DIR, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5001, debug=False)
