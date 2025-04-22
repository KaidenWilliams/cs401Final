

from sagemaker.processing import ScriptProcessor
from sagemaker import get_execution_role, Session
import sagemaker

session = sagemaker.Session()
role = get_execution_role()

processor = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310",  # Official PyTorch CPU image
    role=role,
    command=["python3"],
    instance_count=1,
    instance_type="ml.m5.large",
    base_job_name="audio-processing",
    sagemaker_session=session
)

processor.run(
    code="audio_pipeline.py",
    environment={
        "INPUT_PREFIX": "s3://cs401finalpipelineprocessingdata/data/audio_specs/",
        "OUTPUT_PREFIX": "s3://cs401finalpipelineprocessingdata/data"
    },
    wait=True
)