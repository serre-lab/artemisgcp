import kfp
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kubernetes.client.models import V1EnvVar
import kfp.components as comp

project_id = 'acbm-317517'
region = 'US-CENTRAL1'
pipeline_root_path = 'gs://vertex-ai-sdk-pipelines'

def preprocess(json_file: str):
  import os
  import logging
  logging.basicConfig(level=logging.INFO)
  logging.info('The environment variable is: {}'.format(json_file))
  
  return json_file

preprocess_step = comp.create_component_from_func(logg_env_function,
                                                 base_image='gcr.io/deeplearning-platform-release/tf-gpu.1-15')

@kfp.dsl.pipeline(
    name="automl-image-training-v2",
    pipeline_root=pipeline_root_path)
def pipeline(project_id: str, dataset_file: str):
    ds_op = gcc_aip.ImageDatasetCreateOp(
        project=project_id,
        display_name="flowers",
        gcs_source=dataset_file,
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
    )

    preprocess_op = logg_env_function_op(dataset_file)
    
    training_job_run_op = gcc_aip.AutoMLImageTrainingJobRunOp(
        project=project_id,
        display_name="train-iris-automl-mbsdk-1",
        prediction_type="classification",
        model_type="CLOUD",
        base_model=None,
        dataset=ds_op.outputs["dataset"],
        model_display_name="iris-classification-model-mbsdk",
        training_fraction_split=0.6,
        validation_fraction_split=0.2,
        test_fraction_split=0.2,
        budget_milli_node_hours=8000,
    )

    training_job_run_op.after(preprocess_op)

    endpoint_op = gcc_aip.ModelDeployOp(
        project=project_id, model=training_job_run_op.outputs["model"]
    )


compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='image_classif_pipeline.json')

# api_client = AIPlatformClient(project_id=project_id, region=region)
#
# response = api_client.create_run_from_job_spec(
#     'image_classif_pipeline.json',
#     pipeline_root=pipeline_root_path,
#     parameter_values={
#         'project_id': project_id
#     })
