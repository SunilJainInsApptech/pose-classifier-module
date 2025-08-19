#!/usr/bin/env python3
"""
Viam Custom Module Template for Pose Classification and Fall Detection
"""
import asyncio
from viam.module.module import ModuleService
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.components.service import ServiceBase
from viam.logging import getLogger

# Import your pipeline logic
from run_pose_pipeline import main as run_pose_pipeline_main

LOGGER = getLogger(__name__)

class PoseFallDetectionService(ServiceBase):
    """Custom Viam service for pose classification and fall detection."""
    MODEL = "pose_fall_detection"

    @classmethod
    def new_service(cls, name, **kwargs):
        return cls(name)

    async def do_work(self):
        LOGGER.info("Starting pose classification and fall detection pipeline...")
        await run_pose_pipeline_main()

# Register the service with Viam
Registry.register_resource_creator(
    ServiceBase.SUBTYPE,
    PoseFallDetectionService.MODEL,
    ResourceCreatorRegistration(PoseFallDetectionService.new_service)
)

async def main():
    module = ModuleService(
        "pose_fall_detection_module",
        "0.1.0"
    )
    await module.serve()

if __name__ == "__main__":
    asyncio.run(main())
