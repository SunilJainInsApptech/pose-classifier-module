#!/bin/bash

# --- Configuration ---
RTSP_BASE_URL="rtsp://localhost:8554"
# MediaMTX listens for RTMP on port 1935 by default
RTMP_OUTPUT_URL="rtmp://localhost:1935"

# Optional: Set GStreamer path for stability when run from a virtual environment
SYSTEM_GST_PLUGINS="/usr/lib/aarch64-linux-gnu/gstreamer-1.0"
export GST_PLUGIN_PATH="${SYSTEM_GST_PLUGINS}"

# --- Script Logic ---
if [ -z "$1" ]; then
    echo "Error: No camera path provided."
    echo "Usage: $0 <camera_path_name>"
    exit 1
fi

CAMERA_PATH="$1"
# FIX: Changed RTSP_INPUT_URL to the defined variable: RTSP_BASE_URL
INPUT_URL="${RTSP_BASE_URL}/${CAMERA_PATH}" 
# The output path will be the stream name for RTMP
OUTPUT_PATH="${CAMERA_PATH}_h264"
OUTPUT_URL="${RTMP_OUTPUT_URL}/${OUTPUT_PATH}"

echo "Starting GStreamer RTMP transcoder for ${CAMERA_PATH}"
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
  flvmux ! rtmp2sink location="${OUTPUT_URL}"