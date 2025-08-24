"""
After Hours Alert Service using Twilio
Sends SMS alerts, webhooks, and saves images/JSON when a person is detected on a camera between specified hours.
"""

import os
import asyncio
import logging
from datetime import datetime, time as dt_time
from typing import Optional, Dict, Any
import base64
import tempfile
from twilio.rest import Client
from viam.media.video import ViamImage

LOGGER = logging.getLogger(__name__)

class AfterHoursAlerts:
    """Service for sending after-hours person detection alerts via Twilio"""
    
    def __init__(self, config: dict):
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
        env_phones = os.environ.get('TWILIO_TO_PHONES')
        if env_phones:
            self.to_phones = [phone.strip() for phone in env_phones.split(',')]
        else:
            self.to_phones = config.get('twilio_to_phones', '+19738652226')
        self.webhook_url = (
            os.environ.get('TWILIO_WEBHOOK_URL') or 
            config.get('webhook_url')
        )
        self.push_notification_url = (
            os.environ.get('RIGGUARDIAN_WEBHOOK_URL') or
            config.get('rigguardian_webhook_url', 'https://building-sensor-platform-production.up.railway.app/webhook/fall-alert')
        )
        self.after_hours_start = config.get('after_hours_start', '22:00')  # e.g., '22:00'
        self.after_hours_end = config.get('after_hours_end', '06:00')      # e.g., '06:00'
        self.cooldown_seconds = config.get('alert_cooldown_seconds', 300)
        self.last_alert_time = {}  # Track last alert time per camera
        try:
            self.client = Client(self.account_sid, self.auth_token)
            LOGGER.info("✅ Twilio client initialized successfully")
        except Exception as e:
            LOGGER.error(f"❌ Failed to initialize Twilio client: {e}")
            raise

    def is_after_hours(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now()
        start = dt_time.fromisoformat(self.after_hours_start)
        end = dt_time.fromisoformat(self.after_hours_end)
        if start < end:
            return start <= now.time() < end
        else:
            return now.time() >= start or now.time() < end

    def should_send_alert(self, camera_name: str) -> bool:
        now = datetime.now()
        if not self.is_after_hours(now):
            LOGGER.debug(f"Not after hours: {now.time()} not in {self.after_hours_start}-{self.after_hours_end}")
            return False
        last_time = self.last_alert_time.get(camera_name)
        if last_time:
            time_since_last = (now - last_time).total_seconds()
            if time_since_last < self.cooldown_seconds:
                LOGGER.debug(f"Alert cooldown active for camera {camera_name} ({time_since_last:.1f}s < {self.cooldown_seconds}s)")
                return False
        return True

    async def save_image_locally(self, image: ViamImage, camera_name: str) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"after_hours_{camera_name}_{timestamp}.jpg"
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, filename)
            with open(image_path, 'wb') as f:
                f.write(image.data)
            LOGGER.info(f"📸 After-hours image saved: {image_path}")
            return image_path
        except Exception as e:
            LOGGER.error(f"❌ Failed to save image: {e}")
            return ""

    def format_alert_message(self, camera_name: str, timestamp: datetime, image_path: str = "") -> str:
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        message = f"🚨 AFTER HOURS PERSON DETECTED 🚨\n"
        message += f"Camera: {camera_name}\n"
        message += f"Time: {timestamp_str}\n"
        if image_path:
            message += f"Image: {os.path.basename(image_path)}\n"
        message += "\nPlease check the location immediately."
        return message

    async def send_after_hours_alert(self, camera_name: str, image: ViamImage, metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            if not self.should_send_alert(camera_name):
                return False
            timestamp = datetime.now()
            self.last_alert_time[camera_name] = timestamp
            image_path = await self.save_image_locally(image, camera_name)
            message = self.format_alert_message(camera_name, timestamp, image_path)
            success_count = 0
            for phone_number in self.to_phones:
                try:
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.from_phone,
                        to=phone_number
                    )
                    LOGGER.info(f"📱 After-hours alert sent to {phone_number}, SID: {message_obj.sid}")
                    success_count += 1
                except Exception as e:
                    LOGGER.error(f"❌ Failed to send SMS to {phone_number}: {e}")
            push_success = await self.send_push_notification(camera_name, timestamp, metadata, image)
            if push_success:
                LOGGER.info("📱 Push notification sent to rigguardian.com successfully")
            else:
                LOGGER.warning("⚠️ Push notification failed - SMS alert still sent")
            if success_count > 0:
                LOGGER.info(f"✅ After-hours alert sent successfully to {success_count}/{len(self.to_phones)} recipients")
                return True
            else:
                LOGGER.error("❌ Failed to send after-hours alert to any recipients")
                return False
        except Exception as e:
            LOGGER.error(f"❌ Error sending after-hours alert: {e}")
            return False

    async def send_push_notification(self, camera_name: str, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None, image: Optional[ViamImage] = None) -> bool:
        try:
            import aiohttp
            import json
            webhook_data = {
                "alert_type": "after_hours",
                "camera_name": camera_name,
                "location": f"Camera {camera_name}",
                "severity": "warning",
                "title": "After Hours Person Detected",
                "message": f"Person detected on {camera_name} after hours.",
                "requires_immediate_attention": True,
                "notification_type": "after_hours_detection",
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {},
            }
            if image:
                try:
                    image_b64 = base64.b64encode(image.data).decode('utf-8')
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    webhook_data["image"] = image_b64
                    webhook_data["image_filename"] = f"after_hours_{camera_name}_{timestamp_str}.jpg"
                    LOGGER.info(f"� Added image data to webhook ({len(image.data)} bytes)")
                except Exception as e:
                    LOGGER.warning(f"⚠️ Failed to encode image for webhook: {e}")
            LOGGER.info(f"🔄 Sending webhook to Railway server")
            LOGGER.info(f"📊 After-hours alert: {camera_name} at {timestamp.strftime('%H:%M:%S')}")
            LOGGER.info(f"🎯 URL: {self.push_notification_url}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.push_notification_url,
                    json=webhook_data,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'AfterHoursDetectionSystem/1.0',
                        'X-Alert-Type': 'after_hours',
                        'X-Sensor-Type': 'after_hours_detection',
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
                        return False
        except ImportError:
            LOGGER.error("❌ aiohttp not installed - install with: pip install aiohttp")
            return False
        except Exception as e:
            LOGGER.error(f"❌ Railway server webhook error: {e}")
            return False

# Example usage:
# alerts = AfterHoursAlerts(config)
# await alerts.send_after_hours_alert(camera_name, image, metadata={...})
