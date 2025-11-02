# python script to run on the Jetson Nano to start/stop the required FFmpeg stream
# based on a requested camera ID.

import subprocess
import time
import os
import signal
import sys
import threading

# --- Configuration ---

# Directory where HLS segments for all cameras will be stored temporarily
sudo mkdir -p /home/sunil/streams/hls_output
sudo chown sunil: /home/sunil/streams/hls_output

# Placeholder map of Camera ID (used by web app) to its RTSP URL
# You MUST replace these with your actual RTSP URLs
CAMERA_URLS = {
    'ch01': 'rtsp://192.168.1.200:554/chID=01&streamType=sub',
}

# --- Droplet Configuration (REQUIRED FOR RSYNC) ---
# NOTE: This user MUST have passwordless SSH access set up (Step 2)
DROPLET_USER = 'root'  # CHANGED TO 'root'
DROPLET_IP = '104.236.30.246'
DROPLET_HLS_SERVE_DIR = '/var/www/hls' # Matches directory in droplet_hls_server.py

# FFmpeg HLS Parameters (using stream copy for efficiency)
# -hls_time 2: segment duration of 2 seconds
# -hls_list_size 5: keep 5 segments in the playlist (10 seconds of video total)
# -hls_delete_threshold 1: delete old segments as new ones are created
FFMPEG_PARAMS = [
    '-loglevel', 'error',       # Only show errors
    '-c:v', 'copy',             # Stream copy (highly efficient)
    '-an',                      # No audio
    '-f', 'hls',                # HLS format
    '-hls_time', '2',
    '-hls_list_size', '5',
    '-hls_flags', 'delete_segments+split_by_time',
    '-hls_delete_threshold', '1',
    '-hls_base_url', '',        # Relative URL for segments
    '-map', '0:v:0'             # Map only the first video stream
]

# Global variables to hold the running processes
ffmpeg_process = None
rsync_process = None
active_camera_id = None

def get_ffmpeg_command(camera_id, output_path):
    """Constructs the full FFmpeg command."""
    if camera_id not in CAMERA_URLS:
        print(f"Error: Camera ID '{camera_id}' not found in configuration.")
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
    from the Jetson to the Droplet.
    """
    global rsync_process
    
    # Destination directory on the Droplet
    dest_path = f"{DROPLET_USER}@{DROPLET_IP}:{DROPLET_HLS_SERVE_DIR}/{camera_id}"
    
    # rsync command details:
    # -a: archive mode (preserves attributes)
    # -z: compression
    # --delete: CRUCIAL for HLS - removes old .ts files on the destination
    # -W: copy files block-by-block (faster for smaller segment files)
    # --exclude: ignores the log file if we were logging here
    # --delay-updates: writes files to a temp dir first, useful for HLS playback reliability
    rsync_command = [
        'rsync',
        '-azW',
        '--delete',
        '--delay-updates',
        # Ensure trailing slash on source to sync contents, not the folder itself
        f"{source_dir}/", 
        dest_path
    ]
    
    # We will run this command repeatedly in a thread, sleeping between pushes
    
    def run_rsync_loop(cmd):
        print(f"Starting continuous rsync job for {camera_id} to {DROPLET_IP}")
        # Add the SSH key identity file argument here for the running thread
        # This is essential when running inside Python/a service, 
        # as it doesn't automatically load the default keys.
        cmd_with_key = cmd + ['-e', 'ssh -i ~/.ssh/id_ed25519_hls']

        while True:
            # Check if we've been signaled to stop (handled by stop_sync_job)
            if threading.current_thread().stop_event.is_set():
                print(f"Rsync loop for {camera_id} received stop signal. Exiting.")
                break
                
            try:
                # Use subprocess.run for a single execution, wait for completion
                subprocess.run(
                    cmd_with_key, 
                    check=True, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError as e:
                # If rsync fails (e.g., connection drop), it will be caught here and retried
                print(f"Rsync Error (retry in 1s): Command failed: {e}")
            except FileNotFoundError:
                print("Rsync Error: rsync command not found. Is it installed?")
            except Exception as e:
                print(f"Rsync Error: Unexpected exception: {e}")
            
            # The heart of the low-latency sync: rsync every 1 second
            time.sleep(1) 

    # Create and start the thread
    rsync_thread = threading.Thread(target=run_rsync_loop, args=(rsync_command,))
    # Use a custom event to signal the thread to stop
    rsync_thread.stop_event = threading.Event()
    rsync_thread.daemon = True
    rsync_process = rsync_thread
    rsync_thread.start()


def stop_sync_job():
    """Stops the running rsync thread."""
    global rsync_process
    if rsync_process and rsync_process.is_alive():
        print("Attempting to stop rsync synchronization job...")
        # Signal the thread loop to exit gracefully
        rsync_process.stop_event.set()
        rsync_process.join(timeout=5)
        
        if rsync_process.is_alive():
            print("Rsync thread did not terminate gracefully.")
        else:
            print("Rsync synchronization stopped.")

    rsync_process = None


def start_stream(camera_id):
    """Starts the FFmpeg subprocess for the given camera ID."""
    global ffmpeg_process, active_camera_id

    if ffmpeg_process and ffmpeg_process.poll() is None:
        if active_camera_id == camera_id:
            print(f"Stream for {camera_id} is already running.")
            return

        print(f"Stopping active stream for {active_camera_id} before starting {camera_id}.")
        stop_stream()

    # Create output directory for the specific camera
    output_dir = os.path.join(HLS_OUTPUT_BASE_DIR, camera_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # The output playlist file name
    playlist_path = os.path.join(output_dir, 'playlist.m3u8')
    
    ffmpeg_command = get_ffmpeg_command(camera_id, playlist_path)
    if not ffmpeg_command:
        return

    print(f"Starting stream for {camera_id}...")
    try:
        # Start the subprocess without waiting
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            # Use 'detached' mode for long-running background process management
            preexec_fn=os.setsid 
        )
        active_camera_id = camera_id
        print(f"FFmpeg process started for {camera_id} with PID: {ffmpeg_process.pid}")
        
        # *** NEW: Start the rsync sync job right after FFmpeg starts ***
        start_sync_job(camera_id, output_dir)
        
        # Return the expected location of the files for syncing
        return output_dir

    except FileNotFoundError:
        print("Error: FFmpeg command not found. Ensure FFmpeg is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        stop_sync_job() # Clean up rsync if FFmpeg fails to start

def stop_stream():
    """Stops the currently running FFmpeg subprocess and the rsync job gracefully."""
    global ffmpeg_process, active_camera_id
    
    # 1. Stop rsync first to prevent syncing stale files
    stop_sync_job()
    
    # 2. Stop FFmpeg
    if ffmpeg_process and ffmpeg_process.poll() is None:
        print(f"Attempting to stop FFmpeg PID {ffmpeg_process.pid} for {active_camera_id}...")
        
        # Send SIGTERM (graceful kill) to the process group
        try:
            os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGTERM)
            ffmpeg_process.wait(timeout=5)
            print("FFmpeg process stopped gracefully.")
        except Exception as e:
            print(f"Could not stop gracefully, forcing kill: {e}")
            ffmpeg_process.kill()
            ffmpeg_process.wait()
        
        ffmpeg_process = None
        active_camera_id = None
    else:
        print("No FFmpeg process is currently running.")

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
        print("Example: python jetson_stream_manager.py CAM_01")
        sys.exit(1)
        
    requested_camera = sys.argv[1].upper()
    
    # Set up signal handler for Ctrl+C to ensure clean shutdown
    def signal_handler(sig, frame):
        print('\nCaught interrupt signal. Shutting down gracefully...')
        stop_stream()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    main_loop(requested_camera)
