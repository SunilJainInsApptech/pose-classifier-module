"""
Fall Detection Alert Service using Twilio
Sends SMS alerts when a fall is detected with image and metadata
"""

import os
import asyncio
import traceback
import time
from time import sleep
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import base64
import tempfile
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from viam.media.video import ViamImage
from annotate_image import main as annotate_main  # Import the main function

LOGGER = logging.getLogger(__name__)

class FallDetectionAlerts:
    """Service for sending fall detection alerts via Twilio"""
    
    def __init__(self, config: dict):
        """Initialize Twilio client and alert configuration"""
        # Try to load from environment variables first, then config
        self.account_sid = (
            os.environ.get('TWILIO_ACCOUNT_SID') or 
            config.get('twilio_account_sid')
        )
        self.auth_token = (
            os.environ.get('TWILIO_AUTH_TOKEN') or 
            config.get('twilio_auth_token')
        )
        self.from_phone = (
            os.environ.get('TWILIO_FROM_PHONE') or 
            config.get('twilio_from_phone')
        )
        
        # Handle phone numbers from environment (comma-separated) or config (list)
        env_phones = os.environ.get('TWILIO_TO_PHONES') or config.get('twilio_to_phones')
        LOGGER.debug(f"Raw TWILIO_TO_PHONES value: '{env_phones}'")
        
        if env_phones and env_phones.strip():
            # Split by comma and filter out empty strings
            raw_phones = env_phones.split(',')
            LOGGER.debug(f"Split phones: {raw_phones}")
            self.to_phones = [phone.strip() for phone in raw_phones if phone.strip()]
            LOGGER.debug(f"Filtered phones: {self.to_phones}")
        else:
            # Fallback to default
            self.to_phones = ['+19738652226']
            LOGGER.info("Using fallback phone number")
        
        # Validate phone number formats
        self._validate_phone_numbers()
        
        self.webhook_url = (
            os.environ.get('TWILIO_WEBHOOK_URL') or 
            config.get('webhook_url')
        )
        
        # Alert settings (these can stay in config since they're not sensitive)
        self.min_confidence = config.get('fall_confidence_threshold', 0.7)
        self.cooldown_seconds = config.get('alert_cooldown_seconds', 120)
        self.last_alert_time = {}  # Track last alert time per camera
        
        # Push notification settings
        self.notify_service_sid = (
            os.environ.get('TWILIO_NOTIFY_SERVICE_SID') or
            config.get('twilio_notify_service_sid')
        )
        self.push_notification_url = (
            os.environ.get('RIGGUARDIAN_WEBHOOK_URL') or
            config.get('rigguardian_webhook_url', 'https://building-sensor-platform-production.up.railway.app/webhook/fall-alert')
        )
        
        # Log what source we're using (without exposing credentials)
        if os.environ.get('TWILIO_ACCOUNT_SID'):
            LOGGER.info("✅ Using Twilio credentials from environment variables")
        else:
            LOGGER.info("⚠️ Using Twilio credentials from robot configuration")
        
        # Validate required config
        if not all([self.account_sid, self.auth_token, self.from_phone]):
            raise ValueError("Missing required Twilio configuration: account_sid, auth_token, from_phone")
        
        if not self.to_phones:
            raise ValueError("No alert phone numbers configured")
        
        # Initialize Twilio client
        try:
            self.client = Client(self.account_sid, self.auth_token)
            LOGGER.info("✅ Twilio client initialized successfully")
        except Exception as e:
            LOGGER.error(f"❌ Failed to initialize Twilio client: {e}")
            raise
    
    def _validate_phone_numbers(self):
        """Validate phone number formats and log warnings for invalid numbers"""
        import re
        
        # E.164 format: + followed by 1-15 digits
        e164_pattern = re.compile(r'^\+[1-9]\d{1,14}$')
        
        # Check from_phone
        if self.from_phone and not e164_pattern.match(self.from_phone):
            LOGGER.warning(f"⚠️ From phone number '{self.from_phone}' may not be valid E.164 format")
        
        # Check to_phones and filter out invalid ones
        valid_phones = []
        for phone in self.to_phones:
            if e164_pattern.match(phone):
                valid_phones.append(phone)
                LOGGER.info(f"✅ Valid recipient phone: {phone}")
            else:
                LOGGER.error(f"❌ Invalid recipient phone number: '{phone}' - must be E.164 format (+1234567890)")
        
        if not valid_phones:
            raise ValueError("No valid recipient phone numbers found. Use E.164 format: +1234567890")
        
        # Update to_phones with only valid numbers
        self.to_phones = valid_phones
        LOGGER.info(f"📱 Configured {len(self.to_phones)} valid recipient phone number(s)")
    
    def should_send_alert(self, camera_name: str, confidence: float) -> bool:
        """Check if we should send an alert based on confidence and cooldown (per camera)"""
        # Check confidence threshold
        if confidence < self.min_confidence:
            LOGGER.debug(f"Fall confidence {confidence:.3f} below threshold {self.min_confidence}")
            return False
        
        # Check cooldown period per camera (not per person)
        now = datetime.now()
        cooldown_key = camera_name  # Use camera_name as the cooldown key
        if cooldown_key in self.last_alert_time:
            time_since_last = (now - self.last_alert_time[cooldown_key]).total_seconds()
            if time_since_last < self.cooldown_seconds:
                LOGGER.info(f"⏳ Alert cooldown active for camera {camera_name} ({time_since_last:.1f}s < {self.cooldown_seconds}s) - no alert sent")
                return False
            else:
                LOGGER.info(f"✅ Cooldown period expired for camera {camera_name} ({time_since_last:.1f}s >= {self.cooldown_seconds}s) - alert will be sent")
        else:
            LOGGER.info(f"🆕 First alert for camera {camera_name} - alert will be sent")
        
        return True
    
    async def save_image_locally(self, image: ViamImage, person_id: str) -> str:
        """Save image to local temporary file and return path"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_detection_{person_id}_{timestamp}.jpg"
            
            # Save to temporary directory
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, filename)
            
            # Convert ViamImage to bytes and save
            with open(image_path, 'wb') as f:
                f.write(image.data)
            
            LOGGER.info(f"📸 Fall detection image saved: {image_path}")
            return image_path
            
        except Exception as e:
            LOGGER.error(f"❌ Failed to save image: {e}")
            return ""
    
    def format_alert_message(self, 
                           camera_name: str, 
                           alert_type: str,
                           person_id: str, 
                           confidence: float,
                           timestamp: datetime,
                           image_path: str = "",
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format the alert message for SMS"""
        
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"🚨 FALL DETECTED 🚨\n"
        message += f"Camera: {camera_name}\n"
        # message += f"Person: {person_id}\n"
        message += f"Fall Confidence: {confidence:.1%}\n"
        message += f"Time: {timestamp_str}\n"
        
        if metadata:
            if 'probabilities' in metadata:
                probs = metadata['probabilities']
                # message += f"Pose Probs: "
                # message += f"Fall:{probs.get('fallen', 0):.1%}\n"
        #        message += f"Stand:{probs.get('standing', 0):.1%} "
        #       message += f"Sit:{probs.get('sitting', 0):.1%}\n"
        
        if image_path:
            message += f"Image: {os.path.basename(image_path)}\n"
        message += "\nPlease check the location immediately."
        
        return message
    
    async def send_fall_alert(self, 
                            camera_name: str,
                            alert_type: str,
                            person_id: str, 
                            confidence: float,
                            image: ViamImage,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send fall detection alert via Twilio SMS (file-fallback only)."""
        
        try:
            # Check if we should send alert (camera-based cooldown)
            if not self.should_send_alert(camera_name, confidence):
                return False
            
            # Record alert time for this camera
            timestamp = datetime.now()
            self.last_alert_time[camera_name] = timestamp
            LOGGER.info(f"🕐 Recording alert time for camera {camera_name}: {timestamp}")
            
            # Save image using simple file-based fallback (data_manager/vision_service removed)
            keypoints = None
            if metadata and isinstance(metadata, dict):
                keypoints = metadata.get('keypoints')
            await self.save_fall_image(camera_name, person_id, confidence, image, detection_info=None, keypoints=keypoints)
            
            # Save image locally for SMS reference
            image_path = await self.save_image_locally(image, person_id)
            
            # Format alert message
            message = self.format_alert_message(
                camera_name=camera_name,
                alert_type="fall",  
                person_id=person_id,
                confidence=confidence,
                timestamp=timestamp,
                image_path=image_path,
                metadata=metadata
            )
            
            # Send SMS to all configured phone numbers
            success_count = 0
            for phone_number in self.to_phones:
                try:
                    # Log the exact numbers being used
                    LOGGER.info(f"📱 Sending SMS from {self.from_phone} to {phone_number}")
                    
                    # Send SMS
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    
                    LOGGER.info(f"📱 Fall alert sent to {phone_number}, SID: {message_obj.sid}")
                    success_count += 1
                    
                except Exception as e:
                    # Surface Twilio API error details when available (400 responses etc.)
                    if hasattr(e, 'code') and hasattr(e, 'msg'):  # Twilio exception
                        LOGGER.error(
                            f"❌ Failed to send SMS to {phone_number}: Twilio error {getattr(e, 'code', None)} - {getattr(e, 'msg', None)} (status={getattr(e, 'status', None)})"
                        )
                        # Add specific guidance for common errors
                        if getattr(e, 'code', None) == 21211:
                            LOGGER.error(f"💡 Error 21211 means invalid 'To' number. Check that {phone_number} is in correct E.164 format (+1234567890)")
                        elif getattr(e, 'code', None) == 21606:
                            LOGGER.error(f"💡 Error 21606 means 'From' number {self.from_phone} is not verified/purchased in your Twilio account")
                    else:
                        LOGGER.error(f"❌ Failed to send SMS to {phone_number}: {e}")
            
            # Send push notification to rigguardian.com app
            push_success = await self.send_push_notification(
                camera_name=camera_name,
                alert_type="fall",
                person_id=person_id,
                confidence=confidence,
                timestamp=timestamp,
                metadata=metadata,
                image=image
            )
            
            if push_success:
                LOGGER.info("📱 Push notification sent to rigguardian.com successfully")
            else:
                LOGGER.warning("⚠️ Push notification failed - SMS alert still sent")
            
            if success_count > 0:
                LOGGER.info(f"✅ Fall alert sent successfully to {success_count}/{len(self.to_phones)} recipients")
                return True
            else:
                LOGGER.error("❌ Failed to send fall alert to any recipients")
                return False
                
        except Exception as e:
            LOGGER.error(f"❌ Error sending fall alert: {e}")
            return False
    
    async def send_push_notification(self, camera_name: str, alert_type: str, person_id: str, confidence: float, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None, image: Optional[ViamImage] = None) -> bool:
        """Send push notification to rigguardian.com web app (forward alert_type)."""
        try:
            # Forward alert_type to webhook notification flow so payloads use the correct type
            return await self.send_webhook_notification(camera_name=camera_name, person_id=person_id, confidence=confidence, timestamp=timestamp, metadata=metadata, image=image, alert_type=alert_type)
        except Exception as e:
            LOGGER.error(f"❌ Error sending push notification: {e}")
            return False
    
    async def send_webhook_notification(self, camera_name: str, person_id: str, confidence: float, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None, image: Optional[ViamImage] = None, alert_type: str = "fall") -> bool:
        """Send notification via webhook to configured push URL(s).

        This consolidates attempts so that failures to the primary format don't
        prevent trying alternative payload shapes. Uses _try_webhook_endpoint
        for the actual HTTP POST and logs each attempt.
        """
        try:
            import aiohttp
            import json
            import base64

            # Build primary payload (Railway-style)
            primary_payload = {
                "alert_type": alert_type,
                "camera_name": camera_name,
                "person_id": str(person_id),
                "location": f"Camera {camera_name}",
                "confidence": confidence,
                "severity": "critical",
                "title": "Fall Alert Detected",
                "message": f"Fall detected on {camera_name} with {confidence:.1%} confidence",
                "requires_immediate_attention": True,
                "notification_type": "fall_detection",
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {},
                "actions": [
                    {"action": "view_camera", "title": "View Camera"},
                    {"action": "acknowledge", "title": "Acknowledge"}
                ]
            }

            # Include image data when available (base64)
            if image:
                try:
                    primary_payload["image"] = base64.b64encode(image.data).decode("utf-8")
                    primary_payload["image_filename"] = f"fall_alert_{camera_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    LOGGER.info(f"📷 Added image to webhook payload ({len(image.data)} bytes)")
                except Exception as e:
                    LOGGER.warning(f"⚠️ Failed to attach image to payload: {e}")

            # Keep a list of payloads to try in order (primary first, then alternatives)
            payloads_to_try = [
                (primary_payload, "primary"),
            ]

            # Add rigguardian-compatible compact payload as a secondary attempt
            rigguardian_payload = {
                "alert_type": alert_type,
                "timestamp": timestamp.isoformat(),
                "camera_name": camera_name,
                "person_id": str(person_id),
                "confidence": confidence,
                "severity": "critical",
                "location": camera_name,
                "title": "🚨 Fall Alert - Immediate Action Required",
                "message": f"Fall detected on {camera_name} with {confidence:.1%} confidence",
                "requires_immediate_attention": True,
                "notification_type": "web_push"
            }
            payloads_to_try.append((rigguardian_payload, "rigguardian"))

            # Minimal payload as a last resort
            minimal_payload = {
                "alert_type": alert_type,
                "timestamp": timestamp.isoformat(),
                "camera_name": camera_name,
                "person_id": str(person_id),
                "confidence": confidence,
                "severity": "critical",
                "location": camera_name,
                "title": "Fall Alert",
                "message": "Fall detected",
                "requires_immediate_attention": True,
                "notification_type": "web_push"
            }
            payloads_to_try.append((minimal_payload, "minimal"))

            # Try each payload in order using the helper
            for payload, name in payloads_to_try:
                LOGGER.info(f"🔄 Attempting webhook ({name}) to {self.push_notification_url}")
                try:
                    sent = await self._try_webhook_endpoint(payload, self.push_notification_url, name)
                    if sent:
                        LOGGER.info(f"✅ Webhook ({name}) delivered successfully")
                        return True
                    else:
                        LOGGER.warning(f"⚠️ Webhook ({name}) attempt failed; trying next payload if available")
                except Exception as e:
                    LOGGER.error(f"❌ Exception during webhook ({name}) attempt: {e}")

            LOGGER.error("❌ All webhook attempts failed")
            return False

        except ImportError:
            LOGGER.error("❌ aiohttp not installed - install with: pip install aiohttp")
            return False
        except Exception as e:
            LOGGER.error(f"❌ Webhook error: {e}")
            return False
    
    async def _try_webhook_endpoint(self, payload: dict, url: str, attempt_name: str) -> bool:
        """Try sending webhook to a specific endpoint with detailed logging"""
        try:
            import aiohttp
            import json
            
            LOGGER.info(f"🔄 Trying {attempt_name} approach to {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'FallDetectionSystem/1.0',
                        'X-Alert-Type': 'fall',
                        'X-Severity': 'critical',
                        'Accept': 'application/json'
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_text = await response.text()
                    response_headers = dict(response.headers)
                    
                    LOGGER.info(f"📡 {attempt_name} response status: {response.status}")
                    LOGGER.info(f"📋 {attempt_name} response headers: {response_headers}")
                    LOGGER.info(f"� {attempt_name} response body: {response_text}")
                    
                    if response.status == 200:
                        LOGGER.info(f"✅ {attempt_name} webhook sent successfully")
                        return True
                    else:
                        LOGGER.error(f"❌ {attempt_name} webhook failed with status {response.status}")
                        
                        # Try to parse response as JSON for error details
                        try:
                            response_json = json.loads(response_text)
                            LOGGER.error(f"🔍 {attempt_name} parsed error: {json.dumps(response_json, indent=2)}")
                            if "error" in response_json:
                                LOGGER.error(f"� {attempt_name} server error: {response_json['error']}")
                            if "expected" in response_json:
                                LOGGER.error(f"💡 {attempt_name} expected format: {response_json['expected']}")
                            if "details" in response_json:
                                LOGGER.error(f"📝 {attempt_name} error details: {response_json['details']}")
                        except json.JSONDecodeError:
                            LOGGER.error(f"📄 {attempt_name} response is not valid JSON")
                        
                        return False
                        
        except aiohttp.ClientTimeout:
            LOGGER.error(f"❌ {attempt_name} webhook request timed out")
            return False
        except Exception as e:
            LOGGER.error(f"❌ {attempt_name} webhook error: {e}")
            return False
    
    async def save_fall_image(self, camera_name: str, person_id: str, confidence: float, image: ViamImage, detection_info=None, keypoints=None):
        """Save fall detection image using the file-based fallback only."""

        try:
            LOGGER.info(f"🔄 Saving fall image (file fallback) for camera: {camera_name}")
            LOGGER.info(f"📷 Image size: {len(image.data)} bytes, Person: {person_id}, Confidence: {confidence:.3f}")

            # Save using the file-based fallback
            file_result = await self._save_fall_image_to_file(camera_name, person_id, confidence, image, keypoints=keypoints)
            
            # Wait for 1 minute to ensure _save_fall_image_to_file has completed
            sleep(60)

            # If the file was saved successfully, run the annotate_image script
            if isinstance(file_result, dict) and 'filename' in file_result:
                filename = file_result['filename']
                LOGGER.info(f"📂 File saved: {filename}")

                # Run the annotate_image script with the filename and bounding boxes
                if detection_info:
                    os.environ['FALL_IMAGE_FILENAME'] = filename  # Pass the filename via environment variable
                    await annotate_main(detection_info)

            return file_result
        
        except Exception as e:
            LOGGER.error(f"❌ Error in save_fall_image: {e}")
            LOGGER.error(traceback.format_exc())
            return {"status": "error", "method": "save_fall_image", "error": str(e)}
    
    async def _save_fall_image_to_file(self, camera_name: str, person_id: str, confidence: float, image: ViamImage, keypoints=None):
        """Fallback method to save image directly to data manager's capture directory"""
        try:
            from datetime import datetime
            import os

            # Use the data manager's capture directory
            capture_dir = "/home/sunil/Documents/viam_captured_images"
            timestamp = datetime.utcnow()

            # Create filename with proper Viam naming convention for data manager to recognize
            # Format: [timestamp]_[component_name]_[method_name].[extension]
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            filename = f"{timestamp_str}_{camera_name}_ReadImage.jpg"
            filepath = os.path.join(capture_dir, filename)

            # Ensure directory exists
            os.makedirs(capture_dir, exist_ok=True)

            # Save the image
            with open(filepath, 'wb') as f:
                f.write(image.data)

            # Create metadata file with Fall tag for data manager to process
            metadata_filename = f"{timestamp_str}_{camera_name}_ReadImage.json"
            metadata_filepath = os.path.join(capture_dir, metadata_filename)

            import json
            metadata_content = {
                "component_name": camera_name,
                "method_name": "ReadImage",
                "tags": ["Fall"],
                "timestamp": timestamp.isoformat(),
                "additional_metadata": {
                    "person_id": person_id,
                    "confidence": f"{confidence:.3f}",
                    "event_type": "fall",
                    "vision_service": "yolo11n-pose",
                    "keypoints": keypoints
                }
            }

            with open(metadata_filepath, 'w') as meta_f:
                json.dump(metadata_content, meta_f, indent=2)

            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                LOGGER.info(f"✅ Fall image saved: {filename} ({file_size} bytes)")
                LOGGER.info(f"📋 Metadata saved: {metadata_filename}")
                LOGGER.info(f"🎯 Component: {camera_name}, Tags: ['Fall']")
                LOGGER.info("🔄 Files will sync to Viam within 1 minute")

                return {"status": "success", "method": "file_fallback", "filename": filename, "path": filepath}
            else:
                LOGGER.error(f"❌ Failed to save: {filepath}")
                return {"status": "error", "method": "file_fallback", "error": "File not saved"}

        except Exception as e:
            LOGGER.error(f"❌ Error in file fallback: {e}")
            return {"status": "error", "method": "file_fallback", "error": str(e)}
