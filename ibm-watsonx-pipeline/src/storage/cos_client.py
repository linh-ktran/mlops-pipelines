"""IBM Cloud Object Storage client — re-exports from mlops_core."""

from mlops_core.storage import COSClient as _BaseCOSClient

from src.pipeline.config import PipelineConfig


class COSClient(_BaseCOSClient):
    """COS client initialized from PipelineConfig."""

    def __init__(self, config: PipelineConfig):
        super().__init__(
            endpoint=config.cos_endpoint,
            api_key=config.cos_api_key,
            instance_crn=config.cos_instance_crn,
            bucket=config.cos_bucket,
        )
