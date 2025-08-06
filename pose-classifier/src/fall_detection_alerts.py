"""
Fall Detection Alert Service using Twilio
Sends SMS alerts when a fall is detected with image and metadata
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import base64
import tempfile
from twilio.rest import Client
from viam.media.video import ViamImage

# Try to import Viam DataManager service
try:
    from viam.services.data_manager import DataManager
    VIAM_DATA_AVAILABLE = True
except ImportError:
    VIAM_DATA_AVAILABLE = False

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
        env_phones = os.environ.get('TWILIO_TO_PHONES')
        if env_phones:
            self.to_phones = [phone.strip() for phone in env_phones.split(',')]
        else:
            self.to_phones = config.get('twilio_to_phones', [])
        
        self.webhook_url = (
            os.environ.get('TWILIO_WEBHOOK_URL') or 
            config.get('webhook_url')
        )
        
        # Alert settings (these can stay in config since they're not sensitive)
        self.min_confidence = config.get('fall_confidence_threshold', 0.7)
        self.cooldown_seconds = config.get('alert_cooldown_seconds', 300)
        self.last_alert_time = {}  # Track last alert time per person
        
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
    
    def should_send_alert(self, person_id: str, confidence: float) -> bool:
        """Check if we should send an alert based on confidence and cooldown"""
        # Check confidence threshold
        if confidence < self.min_confidence:
            LOGGER.debug(f"Fall confidence {confidence:.3f} below threshold {self.min_confidence}")
            return False
        
        # Check cooldown period
        now = datetime.now()
        if person_id in self.last_alert_time:
            time_since_last = (now - self.last_alert_time[person_id]).total_seconds()
            if time_since_last < self.cooldown_seconds:
                LOGGER.debug(f"Alert cooldown active for person {person_id} ({time_since_last:.1f}s < {self.cooldown_seconds}s)")
                return False
        
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
                            person_id: str, 
                            confidence: float,
                            image: ViamImage,
                            metadata: Optional[Dict[str, Any]] = None,
                            data_manager=None,
                            vision_service=None) -> bool:
        """Send fall detection alert via Twilio SMS"""
        
        try:
            # Check if we should send alert
            if not self.should_send_alert(person_id, confidence):
                return False
            
            # Record alert time
            timestamp = datetime.now()
            self.last_alert_time[person_id] = timestamp
            
            # Save image to Viam-monitored directory for automatic sync
            await self.save_fall_image(camera_name, person_id, confidence, image, data_manager, vision_service)
            
            # Save image locally for SMS reference
            image_path = await self.save_image_locally(image, person_id)
            
            # Format alert message
            message = self.format_alert_message(
                camera_name=camera_name,
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
                    # Send SMS
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    
                    LOGGER.info(f"📱 Fall alert sent to {phone_number}, SID: {message_obj.sid}")
                    success_count += 1
                    
                except Exception as e:
                    LOGGER.error(f"❌ Failed to send SMS to {phone_number}: {e}")
            
            # Send push notification to rigguardian.com app
            push_success = await self.send_push_notification(
                camera_name=camera_name,
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
    
    async def send_test_alert(self, camera_name: str = "test_camera") -> bool:
        """Send a test alert to verify Twilio configuration"""
        try:
            message = f"🧪 TEST ALERT 🧪\n"
            message += f"Fall detection system is active\n"
            message += f"Camera: {camera_name}\n"
            message += f"Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n"
            message += f"System is monitoring for falls."
            
            success_count = 0
            for phone_number in self.to_phones:
                try:
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    
                    LOGGER.info(f"📱 Test alert sent to {phone_number}, SID: {message_obj.sid}")
                    success_count += 1
                    
                except Exception as e:
                    LOGGER.error(f"❌ Failed to send test SMS to {phone_number}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            LOGGER.error(f"❌ Error sending test alert: {e}")
            return False
    
    async def send_push_notification(self, camera_name: str, person_id: str, confidence: float, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None, image: Optional[ViamImage] = None) -> bool:
        """Send push notification to rigguardian.com web app"""
        try:
            # For web-based apps, use webhook approach (not Twilio Notify)
            # Twilio Notify is for mobile apps only
            return await self.send_webhook_notification(camera_name, person_id, confidence, timestamp, metadata, image)
                
        except Exception as e:
            LOGGER.error(f"❌ Error sending push notification: {e}")
            return False
    
    async def send_webhook_notification(self, camera_name: str, person_id: str, confidence: float, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None, image: Optional[ViamImage] = None) -> bool:
        """Send notification via webhook to Railway server"""
        try:
            import aiohttp
            import json
            import base64
            
            # Create webhook payload matching Railway server expected structure
            webhook_data = {
                "alert_type": "fall",
                "camera_name": camera_name,
                "person_id": str(person_id),
                "location": f"Camera {camera_name}",  # Default location based on camera
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
            
            # Add image data if available
            if image:
                try:
                    # Convert image to base64
                    image_b64 = base64.b64encode(image.data).decode('utf-8')
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    
                    webhook_data["image"] = image_b64
                    webhook_data["image_filename"] = f"fall_alert_{camera_name}_{timestamp_str}.jpg"
                    
                    LOGGER.info(f"� Added image data to webhook ({len(image.data)} bytes)")
                except Exception as e:
                    LOGGER.warning(f"⚠️ Failed to encode image for webhook: {e}")
            
            LOGGER.info(f"🔄 Sending webhook to Railway server")
            LOGGER.info(f"📊 Fall alert: {camera_name} at {timestamp.strftime('%H:%M:%S')} ({confidence:.1%} confidence)")
            LOGGER.info(f"🎯 URL: {self.push_notification_url}")
            
            # Send to Railway server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.push_notification_url,
                    json=webhook_data,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'FallDetectionSystem/1.0',
                        'X-Alert-Type': 'fall',
                        'X-Sensor-Type': 'fall_detection',
                        'Accept': 'application/json'
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    response_text = await response.text()
                    
                    LOGGER.info(f"📡 Railway server response: {response.status}")
                    
                    if response.status == 200:
                        LOGGER.info("✅ Webhook sent successfully to Railway server")
                        LOGGER.info(f"📄 Response: {response_text}")
                        return True
                    else:
                        LOGGER.error(f"❌ Railway server webhook failed with status {response.status}")
                        LOGGER.error(f"📄 Response: {response_text}")
                        
                        # Try to parse response for specific error details
                        try:
                            response_json = json.loads(response_text)
                            if "error" in response_json:
                                LOGGER.error(f"🔍 Server error: {response_json['error']}")
                            if "missing_fields" in response_json:
                                LOGGER.error(f"💡 Missing fields: {response_json['missing_fields']}")
                            if "details" in response_json:
                                LOGGER.error(f"📝 Details: {response_json['details']}")
                        except json.JSONDecodeError:
                            LOGGER.error("📄 Response is not valid JSON")
                        
                        return False
                        
        except ImportError:
            LOGGER.error("❌ aiohttp not installed - install with: pip install aiohttp")
            return False
        except aiohttp.ClientTimeout:
            LOGGER.error("❌ Webhook request to Railway server timed out")
            return False
        except Exception as e:
            LOGGER.error(f"❌ Railway server webhook error: {e}")
            return False
        try:
            import aiohttp
            import json
            
            # Create webhook payload matching rigguardian.com expected structure exactly
            webhook_data = {
                "alert_type": "fall",
                "timestamp": timestamp.isoformat(),
                "camera_name": camera_name,
                "person_id": person_id,
                "confidence": confidence,
                "severity": "critical",
                "location": camera_name,  # Using camera name as location
                "title": "🚨 Fall Alert - Immediate Action Required",
                "message": f"Fall detected on {camera_name} with {confidence:.1%} confidence",
                "requires_immediate_attention": True,
                "notification_type": "web_push"
            }
            
            LOGGER.info(f"🔄 Sending webhook to rigguardian.com with expected structure")
            LOGGER.info(f"📊 Fall alert: {camera_name} at {timestamp.strftime('%H:%M:%S')} ({confidence:.1%} confidence)")
            
            # Debug: Log the payload structure more concisely to avoid truncation
            LOGGER.info(f"� Payload types: alert_type={type(webhook_data['alert_type']).__name__}, timestamp={type(webhook_data['timestamp']).__name__}, camera_name={type(webhook_data['camera_name']).__name__}")
            LOGGER.info(f"🔧 More types: person_id={type(webhook_data['person_id']).__name__}({webhook_data['person_id']}), confidence={type(webhook_data['confidence']).__name__}({webhook_data['confidence']})")
            LOGGER.info(f"🔧 Final types: severity={type(webhook_data['severity']).__name__}, requires_immediate_attention={type(webhook_data['requires_immediate_attention']).__name__}")
            
            # Show a compact version of the payload
            LOGGER.info(f"🔍 Compact payload: alert_type='{webhook_data['alert_type']}', person_id='{webhook_data['person_id']}', confidence={webhook_data['confidence']}")
            LOGGER.info(f"🔍 Full JSON: {json.dumps(webhook_data, separators=(',', ':'))}")  # Compact JSON
            
            # First, try sending to the main endpoint
            success = await self._try_webhook_endpoint(webhook_data, self.push_notification_url, "main")
            if success:
                return True
            
            # Test if endpoint is reachable with a simple GET request
            LOGGER.info("🔄 Testing if rigguardian.com endpoint is reachable...")
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.push_notification_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        LOGGER.info(f"🌐 GET {self.push_notification_url} returned {response.status}")
                        if response.status == 404:
                            LOGGER.error("❌ API endpoint not found - check if /api/fall-alert exists")
                        elif response.status == 405:
                            LOGGER.info("✅ Endpoint exists but only accepts POST (this is expected)")
            except Exception as e:
                LOGGER.error(f"❌ Cannot reach endpoint: {e}")
            
            # If main endpoint fails, try alternative approaches
            LOGGER.info("🔄 Main endpoint failed, trying alternative approaches...")
            
            # Try without emojis in case they're causing encoding issues
            webhook_data_no_emoji = webhook_data.copy()
            webhook_data_no_emoji["title"] = "Fall Alert - Immediate Action Required"
            webhook_data_no_emoji["message"] = f"Fall detected on {camera_name} with {confidence:.1%} confidence"
            
            success = await self._try_webhook_endpoint(webhook_data_no_emoji, self.push_notification_url, "no-emoji")
            if success:
                return True
            
            # Try with string confidence (in case number is causing issues)
            webhook_data_str = webhook_data_no_emoji.copy()
            webhook_data_str["confidence"] = f"{confidence:.3f}"
            
            success = await self._try_webhook_endpoint(webhook_data_str, self.push_notification_url, "string-confidence")
            if success:
                return True
            
            # Try with minimal required fields only
            webhook_data_minimal = {
                "alert_type": "fall",
                "timestamp": timestamp.isoformat(),
                "camera_name": camera_name,
                "person_id": str(person_id),  # Ensure it's a string
                "confidence": confidence,
                "severity": "critical",
                "location": camera_name,
                "title": "Fall Alert",
                "message": "Fall detected",
                "requires_immediate_attention": True,
                "notification_type": "web_push"
            }
            
            success = await self._try_webhook_endpoint(webhook_data_minimal, self.push_notification_url, "minimal")
            if success:
                return True
            
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
    
    async def save_fall_image(self, camera_name: str, person_id: str, confidence: float, image: ViamImage, data_manager=None, vision_service=None):
        """Save fall detection image using data manager camera capture with Fall tag"""
        try:
            LOGGER.info(f"🔄 Triggering fall image capture for camera: {camera_name}")
            LOGGER.info(f"📊 Image size: {len(image.data)} bytes, Person: {person_id}, Confidence: {confidence:.3f}")
            
            # Use data manager to capture from camera with Fall tag
            if data_manager and vision_service:
                try:
                    # Get the camera component name from the vision service
                    # The vision service should have access to the camera it depends on
                    
                    LOGGER.info(f"🏷️ Triggering data manager capture with Fall tag for camera: {camera_name}")
                    
                    # Use data manager's capture functionality with tags
                    # The data manager will capture from the camera component with the Fall tag
                    capture_result = await data_manager.capture(
                        component_name=camera_name,
                        method_name="ReadImage",
                        tags=["Fall"],
                        additional_metadata={
                            "person_id": person_id,
                            "confidence": f"{confidence:.3f}",
                            "event_type": "fall_detected",
                            "vision_service": "yolov8n-pose"
                        }
                    )
                    
                    LOGGER.info(f"✅ Data manager capture completed: {capture_result}")
                    LOGGER.info(f"🎯 Component: {camera_name}, Tag: Fall, Person: {person_id}")
                    
                    return {"status": "success", "method": "data_manager_capture", "result": capture_result}
                    
                except Exception as dm_error:
                    LOGGER.error(f"❌ Data manager capture failed: {dm_error}")
                    LOGGER.info("🔄 Falling back to file-based method")
                    
                    # Fallback: Save to the data manager's capture directory
                    return await self._save_fall_image_to_file(camera_name, person_id, confidence, image)
                    
            else:
                LOGGER.warning("⚠️ No data manager or vision service provided - using file-based fallback")
                return await self._save_fall_image_to_file(camera_name, person_id, confidence, image)
                
        except Exception as e:
            LOGGER.error(f"❌ Error in save_fall_image: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            return {"status": "error", "method": "save_fall_image", "error": str(e)}
    
    async def _save_fall_image_to_file(self, camera_name: str, person_id: str, confidence: float, image: ViamImage):
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
                    "event_type": "fall_detected",
                    "vision_service": "yolov8n-pose"
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
