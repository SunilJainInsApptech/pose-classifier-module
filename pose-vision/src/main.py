from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.vision import Vision
from models.pose_classifier import PoseClassifier

if __name__ == "__main__":
    module = Module.from_args()
    Registry.register_resource_creator(
        Vision.SUBTYPE,
        PoseClassifier.MODEL,
        ResourceCreatorRegistration(PoseClassifier.new, PoseClassifier.validate_config)
    )
    module.start()
