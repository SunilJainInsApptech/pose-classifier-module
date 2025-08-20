import asyncio
import base64
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.services.vision import VisionClient

# Fill in your connection details
address = "camerasystemnvidia-main.niccosz288.viam.cloud"
api_key = "qj8qcg0093x28jtoi90cs5inashvjnd8"
api_key_id = "db33ed99-42fe-46e4-a403-d9af6729dd2b"
triton_service = "vision-1"  # Name of your Triton vision service
pose_classifier_service = "pose-classifier"  # Name of your pose-classifier service

async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(api_key, api_key_id)
    return await RobotClient.at_address(address, opts)

async def process_camera(robot, camera_name):
    camera = Camera.from_robot(robot, camera_name)
    triton = VisionClient.from_robot(robot, triton_service)
    pose_classifier = robot.resource_by_name(pose_classifier_service)

    while True:
        try:
            # Get image from camera
            img = await camera.get_image()
            img_bytes = img.data
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            # Get detections from Triton service
            detections = await triton.get_detections(img)

            if detections:
                # Prepare command for pose-classifier (custom do_command API)
                command = {
                    "command": "classify_poses",
                    "camera_name": camera_name,
                    "detections": [d.to_dict() for d in detections],
                    "image_data": img_b64
                }
                result = await pose_classifier.do_command(command)
                print(f"[{camera_name}] Result: {result}")
        except Exception as e:
            print(f"[{camera_name}] Error: {e}")
        await asyncio.sleep(1)  # Adjust frame rate as needed

async def get_camera_names(robot):
    # Returns a list of all camera resource names using Camera.SUBTYPE
    return [name.name for name in robot.resource_names_by_subtype(Camera.SUBTYPE)]

async def main():
    robot = await connect()
    camera_names = await get_camera_names(robot)
    print("Detected cameras:", camera_names)
    tasks = [process_camera(robot, cam) for cam in camera_names]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
