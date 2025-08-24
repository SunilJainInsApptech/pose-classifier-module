import os
import json
from viam.rpc.dial import Credentials, DialOptions
from viam.app.data.client import DataClient
from viam.app.data import BoundingBox, Tag

# Load Viam credentials from environment variables
VIAM_LOCATION_SECRET = os.environ.get("VIAM_LOCATION_SECRET")
VIAM_LOCATION_ID = os.environ.get("VIAM_LOCATION_ID")

class ViamDataAnnotator:
    def __init__(self):
        creds = Credentials(
            type='robot-location-secret',
            payload=VIAM_LOCATION_SECRET
        )
        opts = DialOptions(
            credentials=creds,
            auth_entity=VIAM_LOCATION_ID
        )
        self.data_client = DataClient('app.viam.com', opts)

    async def annotate_image(self, filename, bbox, detection_label, pose_label, keypoints, timestamp, camera_name):
        # 1. Find the Binary Data ID for the image
        search_results = await self.data_client.find_binary_data(
            tags=['Fall'],
            mime_type='image/jpeg'
        )
        binary_id = None
        for item in search_results:
            if item.filename == filename:
                binary_id = item.id
                break
        if not binary_id:
            raise Exception("Image not found in Viam Cloud")

        # 2. Add bounding box
        await self.data_client.add_bounding_box_to_image_by_id(binary_id, bbox)

        # 3. Add tags/labels/metadata
        tags = [
            f"{pose_label}",
            f"{detection_label}",
            f"{timestamp}",
            f"{camera_name}",
            f"{json.dumps(keypoints)}"
        ]
        await self.data_client.add_tags_to_binary_data_by_ids(tags=tags, binary_ids=[binary_id])
