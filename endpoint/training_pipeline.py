import kfp
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kubernetes.client.models import V1EnvVar
import kfp.components as comp

project_id = 'acbm-317517'
region = 'us-central1'
pipeline_root_path = 'gs://vertex-ai-sdk-pipelines'

create_step_train = comp.load_component_from_text("""
name: Train Model
description: trains LSTM model

inputs:
- {name: model_uri, type: String, description: 'URI to Base Model to be trained'}
- {name: annotation_uri, type: String, description: 'URI to Annotations to use for training'}
- {name: embedding_uri, type: String, description: 'URI to Embeddings to use for training'}

implementation:
  container:
    image: gcr.io/acbm-317517/artemisgcp_training:latest
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python, 
      # Path of the program inside the container
      /training/main_training.py,
      --model,
      {inputValue: model_uri},
      --emb, 
      {inputValue: embedding_uri},
      --annotation, 
      {inputValue: annotation_uri},
    ]""")


#KFP pipeline. Needs name and root path where artifacts stored
@kfp.dsl.pipeline(
    name="automl-image-inference-v2",
    pipeline_root=pipeline_root_path)


def pipeline(project_id: str):
    train_step = create_step_train(
        model_uri='gs://acbm_videos/model0.9573332767722087.pth',
        annotation_uri='gs://acbm_videos/Trap2_FC-A-1-12-Postfearret_new_video_2019Y_02M_23D_05h_30m_06s_cam_6394846-0000.json',
        embedding_uri='gs://acbm_videos/Trap2_FC-A-1-12-Postfearret_new_video_2019Y_02M_23D_05h_30m_06s_cam_6394846-0000.p ',
    )

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='training_pipeline.json')

api_client = AIPlatformClient(project_id=project_id, region=region)

response = api_client.create_run_from_job_spec(
    'training_pipeline.json',
    pipeline_root=pipeline_root_path,
    parameter_values={
        'project_id': project_id
    })