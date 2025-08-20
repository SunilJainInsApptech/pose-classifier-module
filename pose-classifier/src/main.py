#!/usr/bin/env python3
"""
Pose Classification Service Module
Handles pose classification and fall detection with alerts per camera
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from viam.module.module import Module
from viam.services.generic import Generic
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

# Import your existing fall detection alerts
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fall_detection_alerts import FallDetectionAlerts

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class PoseClassifierService(Generic):
    """
    Pose Classification Service that processes YOLO11 detections
    and triggers fall alerts per camera configuration
    """
    # Use a unique API and model name for the custom service to avoid conflict with Vision API
    from viam.resource.types import API
    API = API("rig-guardian:service:pose-classifier-svc")
    MODEL: Model = Model(ModelFamily("rig-guardian", "pose-classifier"), "pose-classifier-svc")
    
    def __init__(self, name: str):
        super().__init__(name)
        self.pose_classifier = None
        self.fall_detection_alerts = None
        self.enabled_cameras: Dict[str, Dict[str, Any]] = {}
        
    @classmethod
    def new(cls, config, dependencies):
        """Create new pose classification service"""
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service
    
    @classmethod  
    def validate_config(cls, config) -> None:
        """Validate configuration"""
        # Add validation logic here
        pass
    
    def reconfigure(self, config, dependencies):
        """Configure the pose classification service"""
        try:
            attributes = struct_to_dict(config.attributes)
            
            # Load pose classifier
            pose_classifier_path = attributes.get("pose_classifier_path")
            if pose_classifier_path:
                import joblib
                self.pose_classifier = joblib.load(pose_classifier_path)
                LOGGER.info(f"✅ Pose classifier loaded from {pose_classifier_path}")
            
            # Configure fall detection alerts
            alert_config = {
                'fall_confidence_threshold': attributes.get('fall_confidence_threshold', 0.7),
                'alert_cooldown_seconds': attributes.get('alert_cooldown_seconds', 300),
                'rigguardian_webhook_url': attributes.get('railway_webhook_url', 
                    'https://building-sensor-platform-production.up.railway.app/webhook/fall-alert')
            }
            
            # Initialize fall detection alerts if not already done
            if not self.fall_detection_alerts:
                self.fall_detection_alerts = FallDetectionAlerts(alert_config)
                LOGGER.info("✅ Fall detection alerts initialized")
                
        except Exception as e:
            LOGGER.error(f"❌ Failed to configure pose classifier service: {e}")
            raise
    
    async def do_command(self, command: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Handle pose classification commands from smart cameras
        
        Commands:
        - classify_poses: Classify poses from YOLO11 detections
        - register_camera: Register a camera for fall detection
        - get_status: Get service status
        """
        cmd = command.get("command")
        
        if cmd == "classify_poses":
            return await self._classify_poses(command)
        elif cmd == "register_camera":
            return await self._register_camera(command)
        elif cmd == "get_status":
            return await self._get_status()
        else:
            return {"error": f"Unknown command: {cmd}"}
    
    async def _classify_poses(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Classify poses from YOLO11 detections"""
        try:
            camera_name = command.get("camera_name")
            detections = command.get("detections", [])
            image_data = command.get("image_data")  # Base64 encoded image
            
            if not self.pose_classifier:
                return {"error": "Pose classifier not loaded"}
            
            if not detections:
                return {"classified_poses": [], "fall_detected": False}
            
            classified_poses = []
            max_fall_confidence = 0.0
            
            # Process each detection
            for detection in detections:
                # Extract pose keypoints (assuming YOLO11-pose format)
                keypoints = detection.get("keypoints", [])
                if not keypoints:
                    continue
                
                # Use your existing pose classification logic
                # This should match the format expected by your joblib model
                pose_features = self._extract_pose_features(keypoints)
                
                if pose_features is not None:
                    # Classify the pose
                    pose_probs = self.pose_classifier.predict_proba([pose_features])[0]
                    pose_classes = self.pose_classifier.classes_
                    
                    # Create classification result
                    pose_result = {
                        "detection_id": detection.get("id", "unknown"),
                        "bbox": detection.get("bbox"),
                        "keypoints": keypoints,
                        "classification": {
                            class_name: float(prob) 
                            for class_name, prob in zip(pose_classes, pose_probs)
                        }
                    }
                    
                    classified_poses.append(pose_result)
                    
                    # Check for fall
                    fall_confidence = pose_result["classification"].get("fallen", 0.0)
                    max_fall_confidence = max(max_fall_confidence, fall_confidence)
            
            # Check if fall alert should be triggered
            camera_config = self.enabled_cameras.get(camera_name, {})
            fall_threshold = camera_config.get("fall_confidence_threshold", 0.7)
            enable_alerts = camera_config.get("enable_fall_alerts", False)
            
            fall_detected = max_fall_confidence >= fall_threshold
            
            # Send fall alert if needed
            if fall_detected and enable_alerts and self.fall_detection_alerts:
                try:
                    # Decode image data if provided
                    image = None
                    if image_data:
                        import base64
                        from viam.media.video import ViamImage
                        image_bytes = base64.b64decode(image_data)
                        image = ViamImage(data=image_bytes, mime_type="image/jpeg")
                    
                    # Send fall alert
                    await self.fall_detection_alerts.send_fall_alert(
                        camera_name=camera_name,
                        person_id=f"person_{camera_name}",
                        confidence=max_fall_confidence,
                        image=image,
                        metadata={"probabilities": classified_poses[0]["classification"] if classified_poses else {}}
                    )
                    
                    LOGGER.info(f"🚨 Fall alert sent for camera {camera_name} (confidence: {max_fall_confidence:.3f})")
                    
                except Exception as e:
                    LOGGER.error(f"❌ Failed to send fall alert for {camera_name}: {e}")
            
            return {
                "classified_poses": classified_poses,
                "fall_detected": fall_detected,
                "max_fall_confidence": max_fall_confidence,
                "alert_sent": fall_detected and enable_alerts
            }
            
        except Exception as e:
            LOGGER.error(f"❌ Pose classification error: {e}")
            return {"error": str(e)}
    
    async def _register_camera(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Register a camera for fall detection"""
        try:
            camera_name = command.get("camera_name")
            camera_config = command.get("config", {})
            
            self.enabled_cameras[camera_name] = camera_config
            
            LOGGER.info(f"📹 Registered camera {camera_name} with config: {camera_config}")
            
            return {"success": True, "registered_camera": camera_name}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "pose_classifier_loaded": self.pose_classifier is not None,
            "fall_alerts_enabled": self.fall_detection_alerts is not None,
            "registered_cameras": list(self.enabled_cameras.keys()),
            "camera_configs": self.enabled_cameras
        }
    
    def _extract_pose_features(self, keypoints: List[List[float]]) -> Optional[List[float]]:
        """
        Extract pose features from YOLO11 keypoints
        This should match your existing pose feature extraction logic
        """
        try:
            # YOLO11-pose returns 17 keypoints with (x, y, confidence)
            if len(keypoints) < 17:
                return None
            
            # Extract the features in the same format as your training data
            # This is a placeholder - you should use your actual feature extraction
            features = []
            
            for kp in keypoints:
                if len(kp) >= 2:
                    features.extend([kp[0], kp[1]])  # x, y coordinates
                else:
                    features.extend([0.0, 0.0])  # Missing keypoint
            
            # Add any additional features your model expects
            # (angles, distances, ratios, etc.)
            
            return features if len(features) > 0 else None
            
        except Exception as e:
            LOGGER.error(f"❌ Feature extraction error: {e}")
            return None

from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from src.models.pose_classifier import PoseClassifier

if __name__ == "__main__":
    # Register the Vision Service (Viam API)
    Registry.register_resource_creator(
        PoseClassifier.API,
        PoseClassifier.MODEL,
        ResourceCreatorRegistration(PoseClassifier.new, PoseClassifier.validate_config)
    )
    # Register the custom PoseClassifierService (custom API)
    Registry.register_resource_creator(
        PoseClassifierService.API,
        PoseClassifierService.MODEL,
        ResourceCreatorRegistration(PoseClassifierService.new, PoseClassifierService.validate_config)
    )
    Module.run()
