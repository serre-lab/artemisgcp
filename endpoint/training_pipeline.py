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

def print_hello():
    print('Hello')

def check_embeddings_exist(video_file: str) -> str:
    ''' Returns exists if a video_files embeddings exists'''
    from google.cloud import storage

    def parse_url(url: str):
        from urllib.parse import urlparse
        o = urlparse(url)
        return o.netloc, o.path.lstrip('/')

    bucket_name, source_blob_name = parse_url(video_file)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name.replace('.mp4', '.p'))

    if blob.exists():
        return 'Exists'
    else:
        return 'Does not exist'


def upload_embeddings(video_file: str, embeddings: comp.InputArtifact()):
    from google.cloud import storage
    def parse_url(url: str):
        from urllib.parse import urlparse
        o = urlparse(url)
        return o.netloc, o.path.lstrip('/')

    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"
        # The path to your file to upload
        # source_file_name = "local/path/to/file"
        # The ID of your GCS object
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

    bucket_name, source_blob_name = parse_url(video_file)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = source_blob_name.replace('.mp4', '.p')

    upload_blob(bucket_name, embeddings, destination_blob_name)


check_embeddings_component = comp.create_component_from_func(
    check_embeddings_exist, 
    base_image = 'gcr.io/acbm-317517/utils:latest'
    )

upload_component = comp.create_component_from_func(
    upload_embeddings,
    base_image = 'gcr.io/acbm-317517/utils:latest'
)

print_component = comp.create_component_from_func(
    print_hello,
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
      display_name='lstm_trained_model_docker3',
      serving_container_predict_route='/prediction',
      serving_container_health_route='/health',
      serving_container_image_uri='gcr.io/acbm-317517/artemisgcp_training:latest',
      serving_container_environment_variables={"MODEL_PATH": "{}".format(pipeline_root_path)},
  )
    model_upload_op.after(train_step)

    endpoint_create_op = gcc_aip.EndpointCreateOp(
        project=project_id,
        display_name="lstm_treained_model_endpoint3",
    )

    model_deploy_op = gcc_aip.ModelDeployOp( 
        project=project_id,
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_upload_op.outputs["model"],
        deployed_model_display_name="lstm_trained_model_deploy3",
        machine_type="n1-standard-4",
    )
    
# def pipeline(project_id: str):
    
#     with kfp.dsl.ParallelFor(
#         ['gs://acbm_videos/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000.mp4', 'gs://acbm_videos/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000_tesing.mp4']
#         ) as video:
#         check_embeddings_op = check_embeddings_component(video)
#         check_embeddings_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
#         with kfp.dsl.Condition(check_embeddings_op.output != 'Exists'):
#             preprocess_op = (preprocess_component(
#                 video_uri = video,
#                 model_uri= 'models/',
#             ).add_node_selector_constraint(
#                 'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
#             ).set_gpu_limit(1))
#             upload_op = upload_component(video, preprocess_op.output)

#     print_op = print_component()
#     print_op.after(upload_op)
    

    # check_embeddings_op = check_embeddings_component(
    #     video_file)

    # with kfp.dsl.Condition(check_embeddings_op.output != 'Exists'):
    #     preprocess_op = (preprocess_component(
    #         video_uri = video_file,
    #         model_uri= 'models/',
    #     ).add_node_selector_constraint(
    #         'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
    #     ).set_gpu_limit(1))
    

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='training_pipeline.json')

# api_client = AIPlatformClient(project_id=project_id, region=region)

# response = api_client.create_run_from_job_spec(
#     'training_pipeline.json',
#     pipeline_root=pipeline_root_path,
#     parameter_values={
#         'project_id': project_id,
#         'model_uri': 'gs://acbm_videos/model0.9573332767722087.pth',
#         'annotation_bucket': 'acbm_videos',
#         'embedding_bucket': 'acbm_videos'
#     })



#add blob downloader for training process for now. then talk to people about how to separate and pass through
# with open("training_pipeline.yaml", 'r') as yaml_in, open("training_pipeline.json", "w") as json_out:
#     yaml_object = yaml.safe_load(yaml_in) # yaml_object will be a list or a dict
#     json.dump(yaml_object, json_out)
