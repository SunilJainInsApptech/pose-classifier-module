import cv2
import time

# --- IMPORTANT ---
# 1. REPLACE THIS URL with your camera's actual RTSP address
#    (Using the one from the working gst-launch command for reference)
RTSP_URL = "rtsp://192.168.1.200:554/chID=16&streamType=sub"

# 2. Define the GStreamer pipeline based on your working command
#    We replace the 'udpsink' part with the 'videoconvert' and 'appsink'
#    required by OpenCV.
GST_PIPELINE = (
    f'rtspsrc location="{RTSP_URL}" protocols=tcp latency=100 ! ' # Source the stream (must be first)
    'rtph265depay ! '          # Depayload H.265 from RTP
    'h265parse ! '             # Parse the H.265 elementary stream
    'queue ! '                 # Add a queue for better streaming stability
    'avdec_h265 ! '            # Decode the H.265 stream
    'videoconvert ! '          # Convert decoded video to an OpenCV-compatible format
    'video/x-raw, format=BGR ! ' # The final format OpenCV (cv2.imshow) expects
    'appsink drop=1'           # Sink to the OpenCV application (must be last)
)

print(f"Pipeline: {GST_PIPELINE}")
print("Attempting to open RTSP stream...")

# Use the GStreamer backend explicitly
cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print(f"Error: Failed to open RTSP stream with GStreamer backend.")
    print(f"Please check your GStreamer installation, plugins, and network.")
    exit()

print("RTSP stream opened successfully. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Stream read error or end of stream. Reattempting read.")
        time.sleep(0.1)
        continue

    # Display the frame
    cv2.imshow('RTSP Stream Test (GStreamer/H265)', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()