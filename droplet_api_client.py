# Flask API for the Droplet (Public Server). 
# This script receives public requests and forwards commands securely to the Jetson API.

from flask import Flask, request, jsonify, abort, Response
import requests
import time
import threading
import os
from flask_cors import CORS 

app = Flask(__name__)
# This line tells Flask to allow cross-origin requests from any domain (*)
CORS(app) 

# --- Configuration ---

# 1. Jetson API URL - **FIXED TO USE REVERSE SSH TUNNEL ENDPOINT**
# The reverse SSH tunnel forwards traffic from Droplet's 50000 port
# to the Jetson's 5000 port.
JETSON_API_URL = "http://127.0.0.1:50000"

# 2. Security Secret (MUST MATCH the one set in jetson_api_service.py)
API_SECRET = 'your-strong-secret'  # <-- Replace with your actual secret

# 3. Droplet Public URL Configuration
# This is used to build the final HLS stream URL for the client.
DROPLET_IP = '104.236.30.246' 
DROPLET_HLS_PORT = 8000

# Timeout for Jetson API calls (seconds)
JETSON_API_TIMEOUT = 10 
# Time to wait for FFmpeg to generate initial HLS segments before returning stream URL
STREAM_INIT_WAIT_TIME = 5

# --- Add timeout configuration ---
STREAM_IDLE_TIMEOUT = 60  # Stop stream after 60 seconds with no requests

# Track last request time per camera
last_request_time = {}
idle_check_thread = None

def _idle_checker():
    """Background thread that stops streams if no requests for STREAM_IDLE_TIMEOUT seconds."""
    global last_request_time
    while True:
        time.sleep(10)  # Check every 10 seconds
        current_time = time.time()
        
        for camera_id, last_time in list(last_request_time.items()):
            if current_time - last_time > STREAM_IDLE_TIMEOUT:
                app.logger.info(f"Stream {camera_id} idle for {STREAM_IDLE_TIMEOUT}s. Stopping...")
                try:
                    stop_stream_for_camera(camera_id)
                    del last_request_time[camera_id]
                except Exception as e:
                    app.logger.error(f"Failed to stop idle stream {camera_id}: {e}")

@app.before_first_request
def startup_event():
    """Start the idle checker thread on service startup."""
    global idle_check_thread
    idle_check_thread = threading.Thread(target=_idle_checker, daemon=True)
    idle_check_thread.start()
    app.logger.info("Idle stream checker started.")

# --- Utility Functions ---

def send_jetson_command(endpoint, json_data=None):
    """Sends an authenticated POST request to the Jetson API."""
    try:
        # The endpoint argument includes the leading slash, e.g., '/stream/start'
        headers = {'X-API-Secret': API_SECRET}
        response = requests.post(
            f"{JETSON_API_URL}{endpoint}",
            json=json_data,
            headers=headers,
            timeout=JETSON_API_TIMEOUT
        )
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': 'Jetson API command timed out.'}
    except requests.exceptions.RequestException as e:
        # Log the detailed connection error message
        app.logger.error(f"Error connecting to Jetson API via tunnel: {e}")
        return {'status': 'error', 'message': f'Failed to connect to Jetson API via tunnel: {e}'}
    except Exception as e:
        return {'status': 'error', 'message': f'An unexpected error occurred: {e}'}

# --- Droplet API Endpoints ---

@app.route('/stream/start', methods=['POST'])
def start_stream():
    """
    Receives a request to start a stream, forwards the command to the Jetson,
    waits for initialization, and returns the HLS URL.
    """
    data = request.get_json()
    camera_id = data.get('camera_id')

    if not camera_id:
        return jsonify({'status': 'error', 'message': 'Missing camera_id'}), 400

    app.logger.info(f"Received request to start stream for {camera_id}.")
    
    # 1. Send command to Jetson
    jetson_response = send_jetson_command('/stream/start', {'camera_id': camera_id})

    # --- FIX: Check for the 'ok': true status returned by the Jetson API ---
    if jetson_response.get('ok') is True: # Primary success check
    # We still check for 'status' == 'success' for other endpoints, but 'ok' is primary for start
    
        # 2. Wait for FFmpeg to generate initial segments
        app.logger.info(f"Jetson acknowledged. Waiting {STREAM_INIT_WAIT_TIME} seconds for HLS initialization...")
        time.sleep(STREAM_INIT_WAIT_TIME)
        
        # 3. Construct the public HLS URL
        hls_url = f"http://{DROPLET_IP}:{DROPLET_HLS_PORT}/{camera_id}/playlist.m3u8"
        
        return jsonify({
            'status': 'success',
            'message': f'Stream successfully initialized and syncing. URL provided.',
            'hls_url': hls_url
        }), 200 # <-- Explicitly returning 200 OK

    else:
        # 4. Handle errors from Jetson
        # The Jetson API uses the key 'error' and 'ok', so we adapt here
        error_message = jetson_response.get('error') or jetson_response.get('message')
        app.logger.error(f"Error starting stream on Jetson: {error_message}")
        # Note: We return the response JSON from the Jetson, but the HTTP status should reflect the error.
        # Since the Jetson API returned 200 and the failure is internal to the Droplet's logic 
        # (which is now fixed), this 'else' should now only catch actual Jetson errors.
        return jsonify(jetson_response), 500

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """
    Receives a request to stop the current stream and forwards the command to the Jetson.
    """
    app.logger.info("Received request to stop stream.")
    
    # Send command to Jetson
    jetson_response = send_jetson_command('/stream/stop')
    
    if jetson_response.get('status') == 'ok' or jetson_response.get('status') == 'success':
        return jsonify({
            'status': 'success',
            'message': 'Stream successfully stopped on Jetson.'
        })
    else:
        error_message = jetson_response.get('error') or jetson_response.get('message')
        app.logger.error(f"Error stopping stream on Jetson: {error_message}")
        return jsonify(jetson_response), 500

@app.route("/start_stream/<camera_id>", methods=['GET'])
def start_stream_get_endpoint(camera_id: str):
    """
    GET endpoint to start a stream (alternative to POST for simpler browser testing).
    Updates last_request_time to keep stream alive.
    """
    global last_request_time
    
    # Update last request timestamp (keeps stream alive)
    last_request_time[camera_id] = time.time()
    
    app.logger.info(f"GET request to start stream for {camera_id}.")
    
    # Forward to Jetson API
    jetson_response = send_jetson_command('/stream/start', {'camera_id': camera_id})
    
    if jetson_response.get('ok') is True:
        app.logger.info(f"Jetson acknowledged. Waiting {STREAM_INIT_WAIT_TIME} seconds for HLS initialization...")
        time.sleep(STREAM_INIT_WAIT_TIME)
        
        hls_url = f"http://{DROPLET_IP}:{DROPLET_HLS_PORT}/{camera_id}/playlist.m3u8"
        
        return jsonify({
            'status': 'success',
            'message': f'Stream successfully initialized and syncing.',
            'hls_url': hls_url,
            'camera_id': camera_id
        }), 200
    else:
        error_message = jetson_response.get('error') or jetson_response.get('message')
        app.logger.error(f"Error starting stream on Jetson: {error_message}")
        return jsonify(jetson_response), 500

def stop_stream_for_camera(camera_id: str):
    """Stops a specific camera's stream by forwarding to Jetson API."""
    app.logger.info(f"Stopping stream for {camera_id}")
    jetson_response = send_jetson_command('/stream/stop', {'camera_id': camera_id})
    return jetson_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
