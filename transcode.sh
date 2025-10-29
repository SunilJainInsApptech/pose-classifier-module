#!/bin/bash

# This script transcodes an H.265 RTSP stream to H.264 using GStreamer
# with hardware acceleration from a specific Python virtual environment.

# --- Configuration ---
# IMPORTANT: Set this to the correct path of your virtual environment's activate script
VENV_ACTIVATE="/home/sunil/opencv/build/opencv_env/bin/activate" 

# The base URL of the MediaMTX RTSP server
RTSP_BASE_URL="rtsp://localhost:8554"

# --- Script Logic ---

# Check if a camera path was provided as an argument
if [ -z "$1" ]; then
  echo "Error: No camera path provided. Usage: $0 <camera_path>"
  exit 1
fi

CAMERA_PATH="$1"
INPUT_URL="${RTSP_BASE_URL}/${CAMERA_PATH}"
OUTPUT_URL="${RTSP_BASE_URL}/${CAMERA_PATH}_h264"

echo "Activating virtual environment: ${VENV_ACTIVATE}"
source "${VENV_ACTIVATE}"

echo "Starting GStreamer transcoder for ${CAMERA_PATH}"
echo "  Input: ${INPUT_URL}"
echo "  Output: ${OUTPUT_URL}"

# Add a 2-second delay to wait for the MediaMTX RTSP server to be ready
sleep 1

# Adapted GStreamer pipeline for hardware-accelerated transcoding with scaling
gst-launch-1.0 -v \
  rtspsrc location="${INPUT_URL}" latency=0 protocols=tcp ! \
  rtph265depay ! h265parse ! nvv4l2decoder ! \
  nvvidconv ! \
  videorate ! \
  'video/x-raw, format=I420, width=640, height=480, framerate=15/1' ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=3000 ! h264parse ! \
  rtph264pay pt=96 ! \
  rtspsink location="${OUTPUT_URL}"