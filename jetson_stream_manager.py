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
    '-hls_time', '2',
    '-hls_list_size', '5',
    '-hls_flags', 'delete_segments+split_by_time',
    '-hls_delete_threshold', '1',
    '-hls_base_url', '',        
    '-map', '0:v:0'             
]

# Global variables to hold the running processes
ffmpeg_process = None
rsync_process = None
active_camera_id = None
stream_lock = threading.Lock() # Added lock for thread safety

# --- Watchdog Globals ---
watchdog_thread = None
keep_running = False


# --- WATCHDOG IMPLEMENTATION ---

def _monitor_ffmpeg(camera_id, playlist_path):
    """
    Monitors the FFmpeg process status and automatically restarts it if it crashes.
    Runs as a separate daemon thread.
    """
    global ffmpeg_process, stream_lock, keep_running
    
    # Check interval (seconds) - check every 5 seconds
    RESTART_CHECK_INTERVAL = 5 
    
    logging.info(f"Watchdog started for camera {camera_id}.")

    while keep_running:
        time.sleep(RESTART_CHECK_INTERVAL)

        with stream_lock:
            # Check if FFmpeg process exists and if it has terminated
            if ffmpeg_process:
                # poll() returns None if running, or the exit code if terminated
                exit_code = ffmpeg_process.poll()
                
                if exit_code is not None:
                    # FFmpeg has crashed/terminated (like the defunct process you saw)
                    logging.warning(f"FFmpeg process detected dead (Exit Code: {exit_code}). Attempting restart for {camera_id}...")
                    
                    # 1. Ensure the old dead process is cleaned up
                    try:
                        ffmpeg_process.wait() # Clean up the zombie process entry
                    except Exception:
                        pass
                    
                    # 2. Relaunch FFmpeg using the internal start function
                    try:
                        # Before restarting, clear the old reference
                        ffmpeg_process = None 
                        _start_ffmpeg(camera_id, playlist_path)
                        logging.info(f"FFmpeg process successfully restarted.")
                    except Exception as e:
                        logging.error(f"Failed to restart FFmpeg: {e}")
                        # If restart fails, we let the loop continue and try again later
            
            # If ffmpeg_process is None, the stream must have been explicitly stopped
            elif active_camera_id is not None:
                # This case indicates an error where active_camera_id is set, but the process reference is lost.
                logging.error("Watchdog found stream active but FFmpeg process handle missing. Exiting watchdog.")
                break


def _start_ffmpeg(camera_id, playlist_path):
    """Internal function to launch the FFmpeg process."""
    global ffmpeg_process
    
    ffmpeg_command = get_ffmpeg_command(camera_id, playlist_path)
    if not ffmpeg_command:
        raise ValueError(f"Could not construct FFmpeg command for {camera_id}")

    # Start the process using start_new_session=True for clean process group termination
    # This is the portable replacement for preexec_fn=os.setsid
    ffmpeg_process = subprocess.Popen(
        ffmpeg_command,
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL, 
        stdin=subprocess.PIPE,
        start_new_session=True  # Creates a new process group for clean shutdown
    )
    logging.info(f"FFmpeg process started for {camera_id} with PID: {ffmpeg_process.pid}")


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
    global rsync_process
    
    # Destination directory on the Droplet
    dest_path = f"{DROPLET_USER}@{DROPLET_IP}:{DROPLET_HLS_SERVE_DIR}/{camera_id}"
    
    rsync_command = [
        'rsync',
        '-azW',
        '--delete',
        '--delay-updates',
        # Ensure trailing slash on source to sync contents, not the folder itself
        f"{source_dir}/", 
        dest_path
    ]
    
    def run_rsync_loop(cmd):
        logging.info(f"Starting continuous rsync job for {camera_id} to {DROPLET_IP}")
        
        # Add the SSH key argument
        cmd_with_key = cmd + ['-e', f'ssh -o StrictHostKeyChecking=no -i {RSYNC_SSH_KEY_PATH}']
        
        # Add the check flag for the rsync thread
        current_thread = threading.current_thread()
        if not hasattr(current_thread, 'stop_event'):
            current_thread.stop_event = threading.Event()
            
        while True:
            # Check if we've been signaled to stop
            if current_thread.stop_event.is_set():
                logging.info(f"Rsync loop for {camera_id} received stop signal. Exiting.")
                break
                
            try:
                # Use subprocess.run for a single execution, wait for completion
                subprocess.run(
                    cmd_with_key, 
                    check=True, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    timeout=10 # Increased timeout slightly for reliable network operation
                )
            except subprocess.TimeoutExpired:
                logging.warning("Rsync Warning: Command timed out. Retrying...")
            except subprocess.CalledProcessError as e:
                logging.error(f"Rsync Error (retry in 1s): Command failed: return code {e.returncode}")
            except FileNotFoundError:
                logging.error("Rsync Error: rsync command not found. Is it installed?")
            except Exception as e:
                logging.error(f"Rsync Error: Unexpected exception: {e}")
            
            time.sleep(1) # Short delay before the next sync attempt

    # Create and start the thread
    rsync_thread = threading.Thread(target=run_rsync_loop, args=(rsync_command,))
    rsync_thread.stop_event = threading.Event()
    rsync_thread.daemon = True
    rsync_process = rsync_thread
    rsync_thread.start()


def stop_sync_job():
    """Stops the running rsync thread."""
    global rsync_process
    
    if rsync_process and rsync_process.is_alive():
        logging.info("Attempting to stop rsync synchronization job...")
        # Signal the thread to stop its loop
        rsync_process.stop_event.set()
        # Wait for the thread to finish
        rsync_process.join(timeout=5)
        
        if rsync_process and rsync_process.is_alive():
            logging.warning("Rsync thread did not terminate gracefully.")
        else:
            logging.info("Rsync synchronization stopped.")

    rsync_process = None


def start_stream(camera_id):
    """Starts the FFmpeg subprocess for the given camera ID and the watchdog."""
    global ffmpeg_process, active_camera_id, watchdog_thread, keep_running
    
    with stream_lock:
        if active_camera_id:
            if active_camera_id == camera_id:
                logging.info(f"Stream for {camera_id} is already running.")
                return os.path.join(HLS_OUTPUT_BASE_DIR, camera_id)
                
            logging.warning(f"Stopping active stream for {active_camera_id} before starting {camera_id}.")
            stop_stream()
            
        # 1. Setup paths
        output_dir = os.path.join(HLS_OUTPUT_BASE_DIR, camera_id)
        os.makedirs(output_dir, exist_ok=True)
        playlist_path = os.path.join(output_dir, 'playlist.m3u8')
        
        try:
            keep_running = True # Set global flag for watchdog and rsync loops
            
            # 2. Start FFmpeg
            _start_ffmpeg(camera_id, playlist_path)
            
            # 3. Start Rsync sync job
            start_sync_job(camera_id, output_dir)
            
            # 4. Start the watchdog monitor thread
            watchdog_thread = threading.Thread(
                target=_monitor_ffmpeg, 
                args=(camera_id, playlist_path,), 
                daemon=True
            )
            watchdog_thread.start()
            
            active_camera_id = camera_id
            return output_dir

        except Exception as e:
            logging.error(f"Failed to start stream components: {e}")
            # Ensure proper cleanup on failure
            stop_stream() 
            return None


def stop_stream():
    """Stops the currently running FFmpeg subprocess, the rsync job, and the watchdog."""
    global ffmpeg_process, active_camera_id, watchdog_thread, keep_running
    
    with stream_lock:
        # 1. Stop background loops
        keep_running = False
        
        # 2. Stop rsync first
        stop_sync_job()
        
        # 3. Stop FFmpeg
        if ffmpeg_process and ffmpeg_process.poll() is None:
            pid = ffmpeg_process.pid
            logging.info(f"Attempting to stop FFmpeg PID {pid} for {active_camera_id}...")
            
            try:
                # Use os.kill on the PID. If start_new_session=True was used, 
                # this PID is the session leader and kills the whole group.
                os.kill(pid, signal.SIGTERM) 
                
                # Wait for process termination and clean up the process table entry (no zombie)
                ffmpeg_process.wait(timeout=5)
                logging.info("FFmpeg process stopped gracefully.")
            except Exception as e:
                logging.warning(f"Could not stop gracefully, forcing kill: {e}")
                try:
                    ffmpeg_process.kill()
                    ffmpeg_process.wait()
                except Exception:
                    pass
            
            ffmpeg_process = None
            active_camera_id = None
        else:
            logging.info("No active FFmpeg process was found to stop.")

        # 4. Clear watchdog thread reference (it will exit due to keep_running=False)
        watchdog_thread = None
        logging.info("Stream and associated processes successfully stopped.")


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
        stop_stream()
        print("Demo finished.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python jetson_stream_manager.py <camera_id>")
        print("Example: python jetson_stream_manager.py ch01")
        sys.exit(1)
        
    requested_camera = sys.argv[1]
    
    # Set up signal handler for Ctrl+C to ensure clean shutdown
    def signal_handler(sig, frame):
        print('\nCaught interrupt signal. Shutting down gracefully...')
        stop_stream()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    main_loop(requested_camera)
