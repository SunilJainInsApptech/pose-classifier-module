#!/bin/bash

# --- Configuration ---
RTSP_BASE_URL="rtsp://localhost:8554"
SYSTEM_GST_PLUGINS="/usr/lib/aarch64-linux-gnu/gstreamer-1.0"

# --- Script Logic ---
if [ -z "$1" ]; then
  echo "Error: No camera path provided."
  exit 1
fi

CAMERA_PATH="$1"
INPUT_URL="${RTSP_BASE_URL}/${CAMERA_PATH}"
OUTPUT_URL="${RTSP_BASE_URL}/${CAMERA_PATH}_h24" # Note: Changed to _h24 to avoid path conflict

# This bypasses any environment conflicts from the virtualenv.
echo "Forcing GStreamer plugin path to: ${SYSTEM_GST_PLUGINS}"
export GST_PLUGIN_PATH="${SYSTEM_GST_PLUGINS}"

echo "Starting GStreamer transcoder for ${CAMERA_PATH}"
echo "  Input: ${INPUT_URL}"
echo "  Output: ${OUTPUT_URL}"

sleep 2

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