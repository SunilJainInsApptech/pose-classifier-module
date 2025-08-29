import asyncio
import os
import time
from viam.rpc.dial import DialOptions, Credentials
from viam.app.viam_client import ViamClient
from viam.utils import create_filter

ROBOT_ADDRESS = os.environ.get('ROBOT_ADDRESS', 'camerasystemnvidia-main.niccosz288.viam.cloud')  # Replace with your robot address
ROBOT_API_KEY = os.environ.get('ROBOT_API_KEY', 'qj8qcg0093x28jtoi90cs5inashvjnd8')
ROBOT_API_KEY_ID = os.environ.get('ROBOT_API_KEY_ID', 'db33ed99-42fe-46e4-a403-d9af6729dd2b')

# Replace with the filename dynamically passed from _save_fall_image_to_file
filename = os.environ.get('FALL_IMAGE_FILENAME', 'your_file_name.jpg')

async def connect() -> ViamClient:
    dial_options = DialOptions(
        credentials=Credentials(
            type="api-key",
            payload=ROBOT_API_KEY,
        ),
        auth_entity=ROBOT_API_KEY_ID
    )
    return await ViamClient.create_from_dial_options(dial_options)

async def main(detections):
    viam_client = await connect()
    data_client = viam_client.data_client

    # Create a filter for the file name
    my_filter = create_filter(file_name=filename)
    binary_metadata, count, last = await data_client.binary_data_by_filter(
        my_filter,
        include_binary_data=False
    )

    # Get the binary data IDs from the metadata results
    my_ids = [obj.metadata.binary_data_id for obj in binary_metadata]

    if not detections:
        print("No detections found")
        return 1

    for detection in detections:
        # Ensure bounding box is big enough to be useful
        if detection['x_max_normalized'] - detection['x_min_normalized'] <= 0.01 or \
           detection['y_max_normalized'] - detection['y_min_normalized'] <= 0.01:
            continue
        bbox_id = await data_client.add_bounding_box_to_image_by_id(
            binary_id=my_ids[0],
            label=detection['class_name'],
            x_min_normalized=detection['x_min_normalized'],
            y_min_normalized=detection['y_min_normalized'],
            x_max_normalized=detection['x_max_normalized'],
            y_max_normalized=detection['y_max_normalized']
        )
        print(f"Added bounding box to image: {bbox_id}")

    viam_client.close()
    return 0

if __name__ == "__main__":
    # Replace with actual detections passed from process_yolo_pose_outputs
    example_detections = [
        {
            'class_name': 'person',
            'x_min_normalized': 0.1,
            'y_min_normalized': 0.2,
            'x_max_normalized': 0.4,
            'y_max_normalized': 0.5
        }
    ]
    asyncio.run(main(example_detections))