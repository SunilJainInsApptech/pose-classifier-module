# Python HTTP server to run on the Droplet for serving HLS files.
# It uses the built-in http.server module for simplicity, as Nginx is restricted.
# NOTE: For production, this server should be managed by systemd or screen/tmux 
# to keep it running after you disconnect.

import http.server
import socketserver
import os

# --- Configuration ---

# This directory MUST match the destination directory used by rsync from the Jetson
HLS_SERVE_DIR = '/var/www/hls' 
PORT = 8000

# Ensure the base directory exists
os.makedirs(HLS_SERVE_DIR, exist_ok=True)

class HLSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom request handler that serves files from the HLS_SERVE_DIR
    and ensures the correct MIME types are sent for HLS to work.
    """
    
    def __init__(self, *args, **kwargs):
        # Set the server directory to serve from
        super().__init__(*args, directory=HLS_SERVE_DIR, **kwargs)

    def guess_type(self, path):
        """Overrides the default type guesser for HLS files."""
        if path.endswith('.m3u8'):
            return 'application/vnd.apple.mpegurl'
        if path.endswith('.ts'):
            return 'video/mp2t'
        # Default to base class guessing for other files
        return super().guess_type(path)

    def end_headers(self):
        # Add required CORS headers so your web app can access the stream 
        # from a different domain (the Droplet).
        self.send_header('Access-Control-Allow-Origin', '*') 
        super().end_headers()

print(f"Starting HLS web server on port {PORT}")
print(f"Serving content from directory: {HLS_SERVE_DIR}")
print(f"Access stream at: http://104.236.30.246:{PORT}/<CAMERA_ID>/playlist.m3u8")

try:
    with socketserver.TCPServer(("", PORT), HLSRequestHandler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
