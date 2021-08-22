import kfp
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kubernetes.client.models import V1EnvVar
import kfp.components as comp
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile

project_id = 'acbm-317517'
region = 'us-central1'
pipeline_root_path = 'gs://vertex-ai-sdk-pipelines'

create_step_train = comp.load_component_from_text("""
name: Train Model
description: trains LSTM model
inputs:
- {name: model_uri, type: Path, description: 'Path to Base Model to be trained'}
- {name: annotation_bucket, type: String, description: 'Path to Annotations to use for training'}
- {name: embedding_bucket, type: String, description: 'Path to Embeddings to use for training'}
- {name: save_bucket, type: String, description: 'Path to trained model to use for training'}

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
      {inputPath: model_uri},
      --emb, 
      {inputValue: embedding_bucket},
      --annotation, 
      {inputValue: annotation_bucket},
      --save,
      {inputValue: save_bucket},
    ]""")

def download_model(source_blob_model: str, model_file: OutputPath()):
    import subprocess
    subprocess.run(["pip", "install", "google-cloud-storage"])
    from google.cloud import storage
    from urllib.parse import urlparse
 
    client = storage.Client()
    model_url = urlparse(source_blob_model)
    model_bucket = client.bucket(model_url.netloc)
    modelBlob = model_bucket.blob(model_url.path.replace('/',''))
    modelBlob.download_to_filename(model_file)
  
    

download_blob_step = comp.create_component_from_func(
  download_model,
  base_image='gcr.io/google.com/cloudsdktool/cloud-sdk:latest',
)

#KFP pipeline. Needs name and root path where artifacts stored
@kfp.dsl.pipeline(
    name="automl-image-inference-v2",
    pipeline_root=pipeline_root_path)

def pipeline(project_id: str, model_uri: str, annotation_bucket: str, embedding_bucket: str):
    download_blob_op = (download_blob_step(
      model_uri
    ))
    print(pipeline_root_path)
    
    train_step = create_step_train(
        model_uri=download_blob_op.output,
        annotation_bucket=annotation_bucket,
        embedding_bucket=embedding_bucket,
        save_bucket=pipeline_root_path,
    ).add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
    ).set_gpu_limit(1)

    model_upload_op = gcc_aip.ModelUploadOp(
      project=project_id,
      display_name='lstm_trained_model',
      artifact_uri=pipeline_root_path,
      serving_container_image_uri='gcr.io/acbm-317517/artemisgcp_training:latest',
      serving_container_environment_variables={"MODEL_PATH": "{}".format(pipeline_root_path)},
  )
    model_upload_op.after(train_step)
    

    

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='training_pipeline.json')

api_client = AIPlatformClient(project_id=project_id, region=region)

response = api_client.create_run_from_job_spec(
    'training_pipeline.json',
    pipeline_root=pipeline_root_path,
    parameter_values={
        'project_id': project_id,
        'model_uri': 'gs://acbm_videos/model0.9573332767722087.pth',
        'annotation_bucket': 'acbm_videos',
        'embedding_bucket': 'acbm_videos'
    })



#add blob downloader for training process for now. then talk to people about how to separate and pass through