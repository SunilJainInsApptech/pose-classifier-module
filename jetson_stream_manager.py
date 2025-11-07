# python script to run on the Jetson Nano to start/stop the required FFmpeg stream
# based on a requested camera ID.

import subprocess
import time
import os
import signal
import sys
import threading
import logging 

# Set up logging for better output in supervisor/jetson_api_service.err.log
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- Configuration ---

# Directory where HLS segments for all cameras will be stored temporarily
HLS_OUTPUT_BASE_DIR = '/home/sunil/streams/hls_output'

# Placeholder map of Camera ID (used by web app) to its RTSP URL
# You MUST replace these with your actual RTSP URLs
CAMERA_URLS = {
    'Lobby_Center_North': 'rtsp://70.19.68.121:554/chID=25&streamType=sub',
    'CPW_Awning_N_Facing': 'rtsp://70.19.68.121:554/chID=16&streamType=sub',
    'Roof_Front_East_Facing': 'rtsp://70.19.68.121:554/chID=01&streamType=sub',
}

# --- Droplet Configuration (REQUIRED FOR RSYNC) ---
DROPLET_USER = 'root'  
DROPLET_IP = '104.236.30.246'
DROPLET_HLS_SERVE_DIR = '/var/www/hls' 

# RSYNC SSH Key Path - Use ABSOLUTE path for reliability in subprocesses
RSYNC_SSH_KEY_PATH = os.path.expanduser('~/.ssh/id_ed25519_hls')

# FFmpeg HLS Parameters 
FFMPEG_PARAMS = [
    '-loglevel', 'error',       
    '-c:v', 'copy',             
    '-an',                      
    '-f', 'hls',                
    '-hls_time', '4',
    '-hls_list_size', '6',
    '-hls_flags', 'delete_segments+split_by_time',
    '-hls_delete_threshold', '2',
    '-hls_base_url', '',        
    '-map', '0:v:0'             
]

# Global dictionaries to hold multiple processes (one per camera)
ffmpeg_processes = {}    # camera_id -> process
rsync_processes = {}     # camera_id -> thread
watchdog_threads = {}    # camera_id -> thread
stream_locks = {}        # camera_id -> lock

# --- Watchdog Globals ---
keep_running = True


# --- WATCHDOG IMPLEMENTATION ---

def _monitor_ffmpeg(camera_id, playlist_path):
    """
    Monitors the FFmpeg process for a specific camera and restarts it if it crashes.
    """
    global ffmpeg_processes, stream_locks, keep_running
    
    RESTART_CHECK_INTERVAL = 5 
    logging.info(f"Watchdog started for camera {camera_id}.")

    while keep_running:
        time.sleep(RESTART_CHECK_INTERVAL)

        with stream_locks.get(camera_id, threading.Lock()):
            if camera_id in ffmpeg_processes:
                proc = ffmpeg_processes[camera_id]
                exit_code = proc.poll()
                
                if exit_code is not None:
                    logging.warning(f"FFmpeg for {camera_id} died (Exit: {exit_code}). Restarting...")
                    
                    try:
                        proc.wait()
                    except Exception:
                        pass
                    
                    del ffmpeg_processes[camera_id]
                    _start_ffmpeg_for_camera(camera_id, playlist_path)
                    logging.info(f"FFmpeg restarted for {camera_id}.")


def _start_ffmpeg_for_camera(camera_id, playlist_path):
    """Internal function to launch FFmpeg for a specific camera."""
    global ffmpeg_processes 
    
    ffmpeg_command = get_ffmpeg_command(camera_id, playlist_path)
    if not ffmpeg_command:
        raise ValueError(f"Could not construct FFmpeg command for {camera_id}")

    log_file_path = f"/tmp/ffmpeg_{camera_id}.log"
    log_file = open(log_file_path, 'w')
    logging.info(f"FFmpeg stderr logged to: {log_file_path}")

    ffmpeg_processes[camera_id] = subprocess.Popen(
        ffmpeg_command,
        stdout=subprocess.DEVNULL, 
        stderr=log_file,
        stdin=subprocess.PIPE,
        start_new_session=True
    )
    logging.info(f"FFmpeg started for {camera_id} with PID: {ffmpeg_processes[camera_id].pid}")


# --- EXISTING FUNCTIONS (Modified/Relocated for Watchdog) ---


def get_ffmpeg_command(camera_id, output_path):
    """Constructs the full FFmpeg command."""
    if camera_id not in CAMERA_URLS:
        logging.error(f"Error: Camera ID '{camera_id}' not found in configuration.")
        return None

    rtsp_url = CAMERA_URLS[camera_id]
    
    # Prepend the input URL and append the output path
    command = [
        'ffmpeg',
        '-i', rtsp_url,
        *FFMPEG_PARAMS,
        output_path  # Final output path (e.g., /path/to/CAM_01/playlist.m3u8)
    ]
    
    return command

def start_sync_job(camera_id, source_dir):
    """
    Starts a continuous rsync job in a subprocess to push files 
    from the Jetson to the Droplet. (Retained original logic).
    """
    
    # Destination directory on the Droplet
    dest_path = f"{DROPLET_USER}@{DROPLET_IP}:{DROPLET_HLS_SERVE_DIR}/{camera_id}"
    
    # Enhanced rsync command with better error handling
    rsync_command = [
        'rsync',
        '-avz',  # Archive mode, verbose, compress
        '--timeout=30',  # Add timeout for stuck transfers
        '--delete',
        '--delay-updates',
        '--partial',  # Keep partially transferred files
        '--partial-dir=.rsync-partial',  # Store partial files in hidden dir
        # Ensure trailing slash on source to sync contents, not the folder itself
        f"{source_dir}/", 
        dest_path
    ]
    
    def run_rsync_loop(cmd):
        logging.info(f"Starting continuous rsync job for {camera_id} to {DROPLET_IP}")
        
        # Enhanced SSH options
        ssh_opts = (
            f'ssh -o StrictHostKeyChecking=no '
            f'-o ServerAliveInterval=15 '
            f'-o ServerAliveCountMax=3 '
            f'-o ConnectTimeout=10 '
            f'-i {RSYNC_SSH_KEY_PATH}'
        )
        cmd_with_key = cmd + ['-e', ssh_opts]
        
        # Add the check flag for the rsync thread
        current_thread = threading.current_thread()
        if not hasattr(current_thread, 'stop_event'):
            current_thread.stop_event = threading.Event()
        
        consecutive_failures = 0
        max_consecutive_failures = 5
            
        while True:
            # Check if we've been signaled to stop
            if current_thread.stop_event.is_set():
                logging.info(f"Rsync loop for {camera_id} received stop signal. Exiting.")
                break
                
            try:
                # Use subprocess.run for a single execution, wait for completion
                result = subprocess.run(
                    cmd_with_key, 
                    check=True, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, 
                    timeout=60,  # Increased timeout for larger transfers
                    text=True
                )
                
                # Reset failure counter on success
                if consecutive_failures > 0:
                    logging.info(f"Rsync recovered for {camera_id}")
                    consecutive_failures = 0
                
            except subprocess.TimeoutExpired:
                consecutive_failures += 1
                logging.warning(f"Rsync timeout for {camera_id} ({consecutive_failures}/{max_consecutive_failures})")
                
            except subprocess.CalledProcessError as e:
                consecutive_failures += 1
                # Log stderr for debugging
                stderr_output = e.stderr if hasattr(e, 'stderr') else 'No stderr'
                logging.error(
                    f"Rsync failed for {camera_id} (exit {e.returncode}, "
                    f"{consecutive_failures}/{max_consecutive_failures}): {stderr_output[:200]}"
                )
                
            except FileNotFoundError:
                logging.error("Rsync Error: rsync command not found. Is it installed?")
                break  # Fatal error, stop trying
                
            except Exception as e:
                consecutive_failures += 1
                logging.error(f"Rsync unexpected error for {camera_id}: {e}")
            
            # If too many failures, pause longer before retry
            if consecutive_failures >= max_consecutive_failures:
                logging.error(
                    f"Rsync for {camera_id} failed {max_consecutive_failures} times. "
                    f"Pausing for 30 seconds..."
                )
                time.sleep(30)
                consecutive_failures = 0  # Reset counter
            else:
                time.sleep(2)  # Normal retry delay

    # Create and start the thread
    rsync_thread = threading.Thread(target=run_rsync_loop, args=(rsync_command,), daemon=True)
    rsync_thread.stop_event = threading.Event()
    rsync_processes[camera_id] = rsync_thread
    rsync_thread.start()
    logging.info(f"Rsync thread started for {camera_id}")


def start_stream(camera_id):
    """Starts the FFmpeg subprocess for the given camera ID."""
    global keep_running
    
    if camera_id not in stream_locks:
        stream_locks[camera_id] = threading.Lock()
    
    with stream_locks[camera_id]:
        if camera_id in ffmpeg_processes and ffmpeg_processes[camera_id].poll() is None:
            logging.info(f"Stream for {camera_id} is already running.")
            return os.path.join(HLS_OUTPUT_BASE_DIR, camera_id)
        
        # Setup paths
        output_dir = os.path.join(HLS_OUTPUT_BASE_DIR, camera_id)
        os.makedirs(output_dir, exist_ok=True)
        playlist_path = os.path.join(output_dir, 'playlist.m3u8')
        
        try:
            # Start FFmpeg
            _start_ffmpeg_for_camera(camera_id, playlist_path)
            
            # Start rsync
            start_sync_job(camera_id, output_dir)
            
            # Start watchdog
            keep_running = True
            watchdog_thread = threading.Thread(
                target=_monitor_ffmpeg, 
                args=(camera_id, playlist_path),
                daemon=True
            )
            watchdog_threads[camera_id] = watchdog_thread
            watchdog_thread.start()
            logging.info(f"Watchdog thread started for {camera_id}")
            
            return output_dir
            
        except Exception as e:
            logging.error(f"Failed to start stream for {camera_id}: {e}")
            stop_stream(camera_id)
            return None

def stop_stream(camera_id):
    """Stops the stream for a specific camera."""
    global keep_running
    
    with stream_locks.get(camera_id, threading.Lock()):
        # Stop watchdog
        if camera_id in watchdog_threads:
            keep_running = False
            watchdog_threads[camera_id].join(timeout=2)
            del watchdog_threads[camera_id]
            logging.info(f"Stopped watchdog for {camera_id}")
        
        # Stop rsync
        if camera_id in rsync_processes:
            rsync_processes[camera_id].stop_event.set()
            rsync_processes[camera_id].join(timeout=5)
            del rsync_processes[camera_id]
            logging.info(f"Stopped rsync for {camera_id}")
        
        # Stop FFmpeg
        if camera_id in ffmpeg_processes:
            proc = ffmpeg_processes[camera_id]
            if proc.poll() is None:
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            del ffmpeg_processes[camera_id]
            logging.info(f"Stopped FFmpeg for {camera_id}")

def main_loop(camera_id):
    """Simple loop for demonstration, in a real scenario this would be an API listener."""
    print("--- Stream Manager Demo ---")
    
    # 1. Start the stream and the sync job
    output_location = start_stream(camera_id)
    if not output_location:
        return
        
    print(f"\nSuccessfully started stream and sync. HLS files being written to: {output_location}")
    print(f"Streaming and syncing for 30 seconds (simulating user viewing)...")
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nReceived interrupt. Stopping stream...")
    finally:
        # 2. Stop the stream and the sync job
        stop_stream(camera_id)
        print("Demo finished.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python jetson_stream_manager.py <camera_id>")
        print("Example: python jetson_stream_manager.py Roof_Front_East_Facing")
        sys.exit(1)
        
    requested_camera = sys.argv[1]
    
    # Set up signal handler for Ctrl+C to ensure clean shutdown
    def signal_handler(sig, frame):
        print('\nCaught interrupt signal. Shutting down gracefully...')
        stop_stream(requested_camera)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    main_loop(requested_camera)
