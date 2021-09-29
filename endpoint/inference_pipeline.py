import kfp
from kfp.v2 import compiler
from google.cloud import aiplatform
import kfp.components as comp
from kfp.v2.google.client import AIPlatformClient
from kfp.components import OutputPath
# from kfp.dsl import Output
import os

project_id = 'acbm-317517'
region = 'us-central1'
pipeline_root_path = 'gs://vertex-ai-sdk-pipelines'

def download_model(source_blob_model: str, model_file: OutputPath()):
    import subprocess
    subprocess.run(["pip", "install", "google-cloud-storage"])
    from google.cloud import storage
    from urllib.parse import urlparse
    import re
 
    client = storage.Client()
    model_accuracy = None

    model_exists = False
    for model in client.list_blobs(source_blob_model, prefix='trained_models'):
        res = re.findall("\d+\.\d+", model.name)
        model_exists = True
        
        if model_accuracy == None:
            model_accuracy= float(res[0])
            model_name = model.name
        
        else:
            if model_accuracy > float(res[0]):
                continue
            else:
                model_accuracy = float(res[0])
                model_name = model.name

    print('ran through model')
    print(model_exists)
    if model_exists == True:     
        model_bucket = client.bucket(source_blob_model)
        modelBlob = model_bucket.blob(model_name)
        print('model found: using model found in ' + source_blob_model + '/' + model_name)
        modelBlob.download_to_filename(model_file)
    if model_exists == False:
        print("no model downloading one")
        model_file = "models/"

download_blob_step = comp.create_component_from_func(
  download_model,
  base_image='gcr.io/google.com/cloudsdktool/cloud-sdk:latest',
)

preprocess_component = comp.load_component_from_text("""
name: Get embeddings
description: Run the i3d model to get embeddings

inputs:
- {name: video_uri, type: String, description: 'URI of the video file (GCP bucket)'}
- {name: model_uri, type: String, description: 'URI to Annotations to use for training'}

outputs:
- {name: pickled_output, type: Artifact, description: 'URI to Saved model to use for predictions'}

implementation:
  container:
    image: gcr.io/acbm-317517/i3d-preprocess:latest
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python, 
      # Path of the program inside the container
      testing.py,
      --video_file,
      {inputValue: video_uri},
      --model_folder_name, 
      {inputValue: model_uri},
      --output_folder,
      {outputPath: pickled_output},
    ]""")

inference_component = comp.load_component_from_text("""
name: Run LSTM model
description: Runs the LSTM model on embeddings

inputs:
- {name: embeddings, type: Artifact, description: 'Path for embeddings file'}
- {name: model_folder_name, type: Artifact, description: 'Model uri path'}

outputs:
- {name: Predictions, type: Artifact, description: 'Predictions of pipeline'}

implementation:
  container:
    image: gcr.io/acbm-317517/lstm-inference:latest
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python, 
      # Path of the program inside the container
      dset_main_inf_forshed.py,
      --embname,
      {inputPath: embeddings},
      --model_uri,
      {inputPath: model_folder_name},
      --output_file_name,
      {outputPath: Predictions}
    ]""")

@kfp.dsl.pipeline(
    name="automl-image-inference-v2",
    pipeline_root=pipeline_root_path)
def pipeline(project_id: str, video_file: str, model_uri: str):
    preprocess_op = (preprocess_component(
        video_uri = video_file,
        model_uri= 'models/',
    ).add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
    ).set_gpu_limit(1))

    download_blob_op = (download_blob_step(
      model_uri
    ))

    inference_op = (inference_component(
        embeddings = preprocess_op.output,
        model_folder_name = download_blob_op.output
    ).add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
    ).set_gpu_limit(1))

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='inference_pipeline.json')

# api_client = AIPlatformClient(project_id=project_id, region=region)

# response = api_client.create_run_from_job_spec(
#     'inference_pipeline.json',
#     pipeline_root=pipeline_root_path,
#     enable_caching = False,
#     service_account = 'vertex-ai-pipeline@acbm-317517.iam.gserviceaccount.com',
#     parameter_values={
#         'project_id': project_id,
#         'video_file': 'gs://acbm_videos/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000_tesing.mp4',
#         'model_uri': 'acbm_videos'
#     })