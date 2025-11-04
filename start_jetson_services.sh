#!/bin/bash

# --- Rig Guardian Service Startup Script ---
# This script ensures both the API Service and the SSH Tunnel start 
# correctly and persist using nohup.

# --- Configuration ---
# Set your API Secret using single quotes to protect the '!' from shell history expansion
export API_SECRET='Rig_Guardian10023!'
JETSON_API_SCRIPT="jetson_api_service.py"
SSH_TUNNEL_COMMAND="ssh -N -R 50000:localhost:5000 root@104.236.30.246 -i ~/.ssh/id_ed25519_hls"

# --- Cleanup ---
echo "1. Cleaning up old processes..."
ps aux | grep "${JETSON_API_SCRIPT}" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
ps aux | grep "ssh -N -R 50000" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
ps aux | grep -E 'ffmpeg|rsync' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
sleep 2 # Give kernel time to free ports

# --- Start API Service (Runs with the exported API_SECRET) ---
echo "2. Starting Jetson API Service (python3 ${JETSON_API_SCRIPT})..."
# nohup redirects all output to jetson_api.log and runs in background
nohup python3 "${JETSON_API_SCRIPT}" > jetson_api.log 2>&1 &
echo "   - API Service PID: $!"

# --- Start SSH Tunnel ---
echo "3. Starting SSH Reverse Tunnel..."
# nohup redirects all output to ssh_tunnel.log and runs in background
nohup ${SSH_TUNNEL_COMMAND} > ssh_tunnel.log 2>&1 &
echo "   - Tunnel PID: $!"

echo "--- All services started successfully. ---"
